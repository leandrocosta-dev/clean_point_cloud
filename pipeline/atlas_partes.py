"""Atlas de partes: da a LLM (ou a uma pessoa) "olhos" sobre a nuvem para decidir a limpeza regiao por regiao.

    python pipeline/atlas_partes.py segmentar <projeto> --saida D:/Projects/X_atlas [--scans id ...] [--poses poses.json]
    python pipeline/atlas_partes.py ver       D:/Projects/X_atlas/<scan> --regioes 12 15 [--isolar]
    python pipeline/atlas_partes.py ver       D:/Projects/X_atlas/<scan> --vista obliqua_030 --celula C4
    python pipeline/atlas_partes.py dividir   D:/Projects/X_atlas/<scan> --regiao 3 --partes 4
    python pipeline/atlas_partes.py sugerir   D:/Projects/X_atlas --saida rotulos.json
    python pipeline/atlas_partes.py previa    D:/Projects/X_atlas --rotulos rotulos.json
    python pipeline/atlas_partes.py aplicar   <projeto> D:/Projects/X_atlas --rotulos rotulos.json --saida <projeto_novo>

Por que existe: `componentes.py` so separa o que esta SOLTO; lixo encostado na peca (massinha, apoio, fiapos) cai no
mesmo componente e a decisao por componente nao limpa nada (veredito do usuario no Projeto 9). Aqui a nuvem de cada
scan e dividida em REGIOES pequenas: retalhos de superficie lisa (crescimento por normal: a emenda peca x apoio e uma
quina e separa os dois) e GRUPOS rugosos (fiapos, massinha, detalhe fino) cortados em blocos de --bloco mm. Cada regiao recebe medidas e aparece numerada em
imagens sombreadas, entao quem decide enxerga a forma e le a evidencia que a imagem nao mostra.

O que `segmentar` grava por scan (pasta <saida>/<scan>/):
  atlas.npz          voxels (centroide, normal, curvatura, regiao, componente, quadros, altura sobre o plano)
  regioes.json       planos de apoio candidatos, vistas e a tabela das maiores regioes: tipo (lisa/fragmento), area,
                     dimensoes, planaridade, rugosidade, altura sobre o plano, quadros que a veem, componente solto ou
                     nao, vizinhas (contato, angulo, juncao concava/convexa), confirmacao por outros scans (--poses)
                     e indicios em texto. `regioes_todas.json` traz a cauda completa.
  vistas/<vista>_sombra.png    forma sombreada (luz + realce de borda por profundidade), grade A-H x 1-8, escala em mm
  vistas/<vista>_regioes.png   mesma vista, uma cor por regiao, id escrito sobre cada regiao visivel
Decisao em `rotulos.json` (classes: peca | apoio | suporte | residuo | ruido | duvida):
    {"<scan>": {"padrao": "duvida", "poeira": "ruido",
                "componentes": {"1": "apoio"}, "grupos": {"1": "residuo"},
                "regioes": {"2": "peca", "17": {"classe": "residuo", "motivo": "massinha na abertura"}},
                "cortes": [{"plano": 0, "abaixo_mm": 1.0, "classe": "apoio"}], "proteger": [2]}}
  Precedencia: corte (menos regioes em `proteger`) > regiao > caixa > grupo > componente > padrao; "caixas":
  [{"min": [x, y, z], "max": [x, y, z], "classe": "peca", "grupo": 1}] rotula por volume (opcionalmente so um grupo).
  "ilhas": "ruido" = do que fica, so o maior componente conexo permanece (usar depois de ver a `previa`; nao serve
  para peca em pedacos separados). So `peca` e `duvida` ficam;
  `duvida` deixa o resultado PARCIAL (codigo 2): nao aprovar sem resolver. `previa` mostra o que sai (vermelho) e o
  que esta em duvida (amarelo) ANTES de mexer nos quadros; `aplicar` apaga os pixels nos frames de uma copia nova.
Limites: as imagens sao navegacao (voxels); a aplicacao usa todos os pixels dos quadros. Confirmacao por outros scans
mede observacao repetida, nao pertencimento a peca (suporte fixo tambem repete). Regiao lisa pode juntar peca e
residuo quando a emenda e suave: olhar de perto com `ver` e separar com `dividir`.
O projeto de origem nunca e alterado; destinos de `aplicar` nao podem existir.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import geom_gpu
import gpu
import numpy as np
import revo_projeto as RP
from comum import configurar_log, salvar_json_atomico
from recursos import baixa_prioridade
from revo_projeto import RevoScan, gravar_revo

_LOG = configurar_log(__name__)
CLASSES = ("peca", "apoio", "suporte", "residuo", "ruido", "duvida")
MANTER = ("peca", "duvida")
COD = {c: i for i, c in enumerate(CLASSES)}
COLUNAS, LINHAS = "ABCDEFGH", 8


# ---------------------------------------------------------------- leitura dos quadros
def scans_do_projeto(proj, filtro=None):
    revo = next(f for f in os.listdir(proj) if f.endswith(".revo"))
    with open(os.path.join(proj, revo), encoding="utf-8") as fh:
        nos = json.load(fh)["nodes"]
    ids = [n["guid"] for n in nos if n.get("type") != 5 and os.path.isdir(os.path.join(proj, "data", n["guid"]))]
    return revo, nos, [i for i in ids if not filtro or i in filtro]


def abrir_scan(proj, s, tmp):
    scan = RevoScan(os.path.join(proj, "data", s), tmp)
    for modo in sorted({int(x) for x, p in zip(scan.modo, scan.tem_prof) if p}):
        scan.calibrar(modo)
    return scan


def origem_grade(voxel):
    """Origem fixa das chaves de voxel (cobre +-2^20 voxels por eixo), igual em segmentar e aplicar."""
    return np.full(3, -(2**20) * float(voxel))


def acumular(scan, voxel, passo_quadros=1, passo_pixel=1):
    """Voxeliza o scan inteiro em fluxo. Devolve cupy: chaves ordenadas, centroides, observacoes, quadros que viram o
    voxel e posicao media da camera (para orientar as normais)."""
    m = gpu.xp()
    org = origem_grade(voxel)
    lotes, pend = [], 0

    def consolidar():
        nonlocal lotes, pend
        if len(lotes) > 1:
            u, inv, _ = gpu.agrupar(m.concatenate([x[0] for x in lotes]))
            lotes = [(u, gpu.somar_por_grupo(m.concatenate([x[1] for x in lotes]), inv, len(u)))]
        pend = len(lotes[0][0]) if lotes else 0

    for k in range(0, len(scan.ids), passo_quadros):
        if not scan.tem_prof[k]:
            continue
        P, _ = scan.pontos(k, passo=passo_pixel, na_gpu=True)
        if P is None or not len(P):
            continue
        u, inv, cnt = gpu.agrupar(gpu.chaves_voxel(P, voxel, org))
        S = gpu.somar_por_grupo(P, inv, len(u))
        cam = gpu.para_gpu(scan.T[k][:3, 3], m.float64)
        c = cnt.astype(m.float64)[:, None]
        lotes.append((u, m.concatenate([S, c, m.ones_like(c), c * cam[None, :]], axis=1)))
        pend += len(u)
        if pend > 6_000_000:
            consolidar()
    consolidar()
    if not lotes:
        return None
    u, S = lotes[0]
    cnt = S[:, 3]
    return dict(chaves=u, C=S[:, :3] / cnt[:, None], obs=cnt, quadros=S[:, 4], cam=S[:, 5:8] / cnt[:, None])


def voxel_automatico(scan):
    """Voxel pela extensao da cena: ~diagonal/600, entre 0,3 e 2 mm (peca de 60 mm -> 0,3; para-choque -> 2)."""
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    m = gpu.xp()
    B = m.vstack([scan.pontos(k, passo=16, na_gpu=True)[0] for k in ks[:: max(1, len(ks) // 40)]])
    lo, hi = m.percentile(B, 1, axis=0), m.percentile(B, 99, axis=0)
    diag = float(m.linalg.norm(hi - lo))
    return float(np.clip(round(diag / 600, 2), 0.3, 2.0))


# ---------------------------------------------------------------- planos de apoio
def _area_ocupada(S, n, d, limiar, celula):
    m = gpu.xp()
    ng = gpu.para_gpu(n, m.float64)
    Q = S[m.abs(S @ ng + d) < limiar]
    if not len(Q):
        return 0.0
    _, u, v = base_do_plano(n)
    ij = m.floor(m.stack([Q @ gpu.para_gpu(u), Q @ gpu.para_gpu(v)], axis=1) / celula).astype(m.int64)
    return float(len(gpu.agrupar(ij)[0])) * celula * celula


def base_do_plano(n):
    n = np.asarray(n, float)
    n = n / np.linalg.norm(n)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    return n, u, np.cross(n, u)


def planos_candidatos(C, voxel, n_max=3):
    """Ate `n_max` planos por RANSAC sequencial, com area ocupada e fracao de voxels: quem decide qual e a mesa e a
    LLM (a face plana da peca tambem aparece aqui). Ordenados por area. Normal apontando para o lado com mais dado."""
    m = gpu.xp()
    lim = max(0.6, 1.5 * voxel)
    S = C[:: max(1, len(C) // 400_000)]
    resto, out = S, []
    for i in range(n_max):
        if len(resto) < 500:
            break
        n, d, _ = gpu.ransac_plano(resto, limiar=lim, iteracoes=2000, semente=i)
        n = np.asarray(n, float)
        esc = np.linalg.norm(n)
        n, d = n / esc, float(d) / esc
        h = S @ gpu.para_gpu(n) + d
        if float((h > 3 * lim).mean()) < float((h < -3 * lim).mean()):
            n, d = -n, -d
            h = -h
        dentro = m.abs(h) < lim
        out.append(
            dict(
                normal=n.round(6).tolist(),
                d=round(d, 4),
                limiar_mm=round(lim, 2),
                fracao_voxels=round(float(dentro.mean()), 4),
                area_mm2=round(_area_ocupada(S, n, d, lim, max(2.0, 4 * voxel)), 0),
                acima_frac=round(float((h > 3 * lim).mean()), 4),
                abaixo_frac=round(float((h < -3 * lim).mean()), 4),
            )
        )
        resto = resto[m.abs(resto @ gpu.para_gpu(n) + d) >= lim]
    out.sort(key=lambda p: -p["area_mm2"])
    for i, p in enumerate(out):
        p["id"] = i
    return out


# ---------------------------------------------------------------- segmentacao em regioes
def segmentar_regioes(
    C, N, curv, voxel, angulo=18.0, degrau=0.5, curv_max=0.06, min_lisa=25, min_frag=8, bloco=4.0, conf=None
):
    """Rotulo por voxel: 0 = poeira; 1.. = regioes ordenadas por tamanho. Retalhos lisos crescem por normal; o que
    sobra (rugoso) forma GRUPOS por proximidade e cada grupo e cortado em blocos de `bloco` mm (e, com `conf`, entre
    confirmado/nao confirmado por outros scans), para que detalhe fino da peca e residuo encostado nao fiquem na
    mesma regiao. Devolve (rot cupy int32, tipo numpy (R+1,), grupo numpy (R+1,) com 0 = sem grupo, pares (i, j))."""
    m = gpu.xp()
    n = len(C)
    _, J = gpu.Indice(C, 1.8 * voxel).knn(C, 12)
    I = m.repeat(m.arange(n, dtype=m.int64), J.shape[1])
    J = J.reshape(-1)
    ok = (J >= 0) & (J != I)
    I, J = I[ok], J[ok]
    D = C[J] - C[I]
    cosn = (N[I] * N[J]).sum(1)
    liso = (
        (cosn > np.cos(np.radians(angulo)))
        & (m.abs((D * N[I]).sum(1)) < degrau * voxel)
        & (m.abs((D * N[J]).sum(1)) < degrau * voxel)
        & (curv[I] < curv_max)
        & (curv[J] < curv_max)
    )
    raiz = geom_gpu.rotulos_por_arestas(n, I[liso], J[liso])
    _, inv = m.unique(raiz, return_inverse=True)
    inv = inv.reshape(-1)
    tam = m.bincount(inv)
    rot = m.where(tam[inv] >= min_lisa, inv + 1, 0).astype(m.int64)  # provisorio: 0 = sobra
    # quinas e bordas: sobra encostada num retalho liso e a menos de 0,75 voxel do plano dele entra no retalho. O plano
    # e o do voxel ORIGINAL do retalho (`ref`), herdado em cadeia: sem isso a mesa subiria pela parede da peca.
    ref = m.arange(n, dtype=m.int64)
    for _ in range(4):
        rj = ref[J]
        cand = (rot[I] == 0) & (rot[J] > 0) & (m.abs(((C[I] - C[rj]) * N[rj]).sum(1)) < 0.75 * voxel)
        if not bool(cand.any()):
            break
        ic, jc = I[cand], J[cand]
        dist = m.linalg.norm(C[jc] - C[ic], axis=1)
        ordem = m.lexsort(m.stack([dist, ic]))
        ic, jc = ic[ordem], jc[ordem]
        prim = m.ones(len(ic), bool)
        prim[1:] = ic[1:] != ic[:-1]
        novo, nref = rot.copy(), ref.copy()
        novo[ic[prim]] = rot[jc[prim]]
        nref[ic[prim]] = ref[jc[prim]]
        rot, ref = novo, nref
    # grupos rugosos: sobras ligadas entre si por proximidade; cada grupo e cortado em blocos
    sobra = rot == 0
    idx = m.cumsum(sobra) - 1
    par = sobra[I] & sobra[J]
    ns = int(sobra.sum())
    grupo_v = m.zeros(n, m.int64)  # grupo por voxel (0 = liso ou poeira)
    bruto = rot.copy()
    if ns:
        rz = geom_gpu.rotulos_por_arestas(ns, idx[I[par]], idx[J[par]])
        _, fi = m.unique(rz, return_inverse=True)
        fi = fi.reshape(-1)
        ft = m.bincount(fi)
        ordem_g = m.argsort(-ft, kind="stable")
        posto_g = m.empty(len(ft), m.int64)
        posto_g[ordem_g] = m.arange(len(ft))
        gv = m.where(ft[fi] >= min_frag, posto_g[fi] + 1, 0)  # 1 = maior grupo
        grupo_v[sobra] = gv
        tem = m.flatnonzero(grupo_v > 0)
        if len(tem):
            cel = m.floor(C[tem] / bloco).astype(m.int64)
            cb = (conf[tem] >= 1).astype(m.int64) if conf is not None else m.zeros(len(tem), m.int64)
            _, bi, _ = gpu.agrupar(m.concatenate([grupo_v[tem, None], cel, cb[:, None]], axis=1))
            bruto[tem] = int(rot.max()) + 1 + bi
    # renumera por tamanho (1 = maior); 0 continua poeira
    u, inv2, cnt = m.unique(bruto, return_inverse=True, return_counts=True)
    inv2 = inv2.reshape(-1)
    tem_zero = bool(u[0] == 0)
    c2 = cnt.copy()
    if tem_zero:
        c2[0] = -1
    ordem = m.argsort(-c2, kind="stable")
    posto = m.empty(len(u), m.int64)
    posto[ordem] = m.arange(len(u))
    final = m.where(bruto == 0, 0, posto[inv2] + 1) if tem_zero else posto[inv2] + 1
    R = int(final.max()) if n else 0
    grupo = m.zeros(R + 1, m.int64)
    grupo[final] = grupo_v
    grupo[0] = 0
    grupo = gpu.para_cpu(grupo)
    tipo = np.where(grupo > 0, "fragmento", "lisa")
    tipo[0] = "poeira"
    return final.astype(m.int32), tipo, grupo, (I, J)


def _extremos(valores, grupos, ng):
    m = gpu.xp()
    lo = m.full((ng,) + valores.shape[1:], m.inf)
    hi = m.full((ng,) + valores.shape[1:], -m.inf)
    m.minimum.at(lo, grupos, valores)
    m.maximum.at(hi, grupos, valores)
    return lo, hi


def tabela_regioes(at, tipo, pares, planos, voxel):
    """Medidas por regiao (somas por grupo na GPU). `at`: dict cupy com C, N, curv, rot, comp, obs, quadros, conf."""
    m = gpu.xp()
    C, N, rot = at["C"], at["N"], at["rot"].astype(m.int64)
    R = int(rot.max())
    ng = R + 1
    n = gpu.para_cpu(m.bincount(rot, minlength=ng)).astype(np.int64)
    S = gpu.para_cpu(gpu.somar_por_grupo(C, rot, ng))
    cen = S / np.maximum(n, 1)[:, None]
    Cc = C - gpu.para_gpu(cen)[rot]
    prod = m.stack(
        [Cc[:, 0] ** 2, Cc[:, 1] ** 2, Cc[:, 2] ** 2, Cc[:, 0] * Cc[:, 1], Cc[:, 0] * Cc[:, 2], Cc[:, 1] * Cc[:, 2]], 1
    )
    cv = gpu.para_cpu(gpu.somar_por_grupo(prod, rot, ng)) / np.maximum(n, 1)[:, None]
    M = np.empty((ng, 3, 3))
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = cv[:, 0], cv[:, 1], cv[:, 2]
    M[:, 0, 1] = M[:, 1, 0] = cv[:, 3]
    M[:, 0, 2] = M[:, 2, 0] = cv[:, 4]
    M[:, 1, 2] = M[:, 2, 1] = cv[:, 5]
    ev = np.sqrt(np.clip(np.linalg.eigvalsh(M)[:, ::-1], 0, None))  # desvios nos 3 eixos principais, maior primeiro
    Nok = m.nan_to_num(N)
    nm = gpu.para_cpu(gpu.somar_por_grupo(Nok, rot, ng)) / np.maximum(n, 1)[:, None]
    curv = gpu.para_cpu(gpu.somar_por_grupo(m.nan_to_num(at["curv"]), rot, ng)) / np.maximum(n, 1)
    obs = gpu.para_cpu(gpu.somar_por_grupo(at["obs"], rot, ng))
    qd = gpu.para_cpu(gpu.somar_por_grupo(at["quadros"], rot, ng)) / np.maximum(n, 1)
    lo, hi = (gpu.para_cpu(x) for x in _extremos(C, rot, ng))
    comp = at["comp"].astype(m.int64)
    comp_reg = np.zeros(ng, np.int64)
    comp_reg[gpu.para_cpu(rot)] = gpu.para_cpu(comp)  # regiao lisa/fragmento vive num unico componente
    n_comp = gpu.para_cpu(m.bincount(comp))
    hs = None
    if planos:
        p0 = planos[0]
        h = C @ gpu.para_gpu(p0["normal"], m.float64) + p0["d"]
        hlo, hhi = (gpu.para_cpu(x) for x in _extremos(h, rot, ng))
        hmed = gpu.para_cpu(gpu.somar_por_grupo(h, rot, ng)) / np.maximum(n, 1)
        noplano = gpu.para_cpu(gpu.somar_por_grupo((m.abs(h) < p0["limiar_mm"]).astype(m.float64), rot, ng))
        hs = (hlo, hmed, hhi, noplano / np.maximum(n, 1))
    conf = None
    if at.get("conf") is not None:
        c1 = gpu.para_cpu(gpu.somar_por_grupo((at["conf"] >= 1).astype(m.float64), rot, ng)) / np.maximum(n, 1)
        c2 = gpu.para_cpu(gpu.somar_por_grupo((at["conf"] >= 2).astype(m.float64), rot, ng)) / np.maximum(n, 1)
        conf = (c1, c2)
    # adjacencia entre regioes
    I, J = pares
    ri, rj = rot[I], rot[J]
    dif = (ri != rj) & (ri > 0) & (rj > 0)
    viz = {}
    if bool(dif.any()):
        I2, J2, ri, rj = I[dif], J[dif], ri[dif], rj[dif]
        u, inv, cnt = gpu.agrupar(ri * ng + rj)
        cosm = gpu.somar_por_grupo((Nok[I2] * Nok[J2]).sum(1), inv, len(u)) / cnt
        conc = gpu.somar_por_grupo(((C[J2] - C[I2]) * Nok[I2]).sum(1), inv, len(u)) / cnt
        u, cnt, cosm, conc = (gpu.para_cpu(x) for x in (u, cnt, cosm, conc))
        for k, c, cs, cc in zip(u, cnt, cosm, conc):
            a, b = divmod(int(k), ng)
            ang = float(np.degrees(np.arccos(np.clip(cs, -1, 1))))
            junc = (
                "lisa"
                if ang < 15
                else ("concava" if cc > 0.05 * voxel else "convexa" if cc < -0.05 * voxel else "quina")
            )
            viz.setdefault(a, []).append((int(c), b, round(ang, 0), junc))
    maior_lisa = next((r for r in range(1, ng) if tipo[r] == "lisa"), 1)
    linhas = []
    for r in range(1, ng):
        e = ev[r]
        dims = (2 * np.sqrt(3) * e).round(1)  # extensao equivalente (distribuicao uniforme) nos eixos principais
        planar = float((e[1] - e[2]) / e[0]) if e[0] > 0 else 0.0
        lin = dict(
            id=r,
            tipo=str(tipo[r]),
            grupo=int(at["grupo"][r]),
            voxels=int(n[r]),
            area_mm2=round(float(n[r]) * voxel * voxel, 1),
            dimensoes_mm=dims.tolist(),
            centroide=cen[r].round(2).tolist(),
            caixa_min=lo[r].round(2).tolist(),
            caixa_max=hi[r].round(2).tolist(),
            normal_media=nm[r].round(3).tolist(),
            coerencia_normais=round(float(np.linalg.norm(nm[r])), 3),
            planaridade=round(planar, 3),
            espessura_rms_mm=round(float(e[2]), 3),
            curvatura_media=round(float(curv[r]), 4),
            obs_por_voxel=round(float(obs[r] / max(n[r], 1)), 1),
            quadros_por_voxel=round(float(qd[r]), 1),
            componente=int(comp_reg[r]),
            componente_voxels=int(n_comp[comp_reg[r]]),
            solto=bool(comp_reg[r] != comp_reg[maior_lisa]),
        )
        if hs is not None:
            lin["altura_plano0_mm"] = [round(float(hs[0][r]), 2), round(float(hs[1][r]), 2), round(float(hs[2][r]), 2)]
            lin["fracao_no_plano0"] = round(float(hs[3][r]), 3)
        if conf is not None:
            lin["confirmado_por_1_outro_scan"] = round(float(conf[0][r]), 3)
            lin["confirmado_por_2_outros_scans"] = round(float(conf[1][r]), 3)
        vz = sorted(viz.get(r, []), reverse=True)[:6]
        lin["vizinhas"] = [dict(id=b, contatos=c, angulo_graus=a, juncao=j) for c, b, a, j in vz]
        lin["indicios"] = indicios(lin)
        linhas.append(lin)
    return linhas


def indicios(lin):
    """Frases curtas a partir das medidas: ajudam a leitura, NAO sao a decisao."""
    out = []
    if lin["solto"]:
        out.append("componente separado da maior regiao lisa")
    if lin["tipo"] == "fragmento":
        out.append("sem superficie lisa (fiapo, migalha, borda ruidosa ou detalhe fino)")
    if lin.get("fracao_no_plano0", 0) > 0.8:
        out.append("deitada no plano 0 (candidata a mesa/placa se o plano 0 for o apoio)")
    elif "altura_plano0_mm" in lin and abs(lin["altura_plano0_mm"][0]) < 1.0 and lin["altura_plano0_mm"][2] > 3:
        out.append("encosta no plano 0 e sobe (peca apoiada ou suporte)")
    if lin["planaridade"] > 0.9 and lin["espessura_rms_mm"] < 0.3 and lin["area_mm2"] > 200:
        out.append("chapa plana e fina")
    if lin["quadros_por_voxel"] < 3:
        out.append("vista em poucos quadros (transitorio ou borda de cobertura)")
    if lin.get("confirmado_por_1_outro_scan") is not None:
        c = lin["confirmado_por_1_outro_scan"]
        if c < 0.15:
            out.append("quase sem confirmacao em outros scans (muda de lugar entre poses, ou so este scan olha ali)")
        elif c > 0.8:
            out.append("confirmada por outros scans (repete entre poses; suporte fixo tambem repete)")
    conc = [v for v in lin["vizinhas"] if v["juncao"] == "concava" and v["contatos"] >= 20]
    if conc:
        out.append("juncao concava com " + ", ".join(str(v["id"]) for v in conc[:3]) + " (objeto encostado em outro)")
    return out


# ---------------------------------------------------------------- render (z-buffer de pontos na GPU)
def paleta(n):
    import colorsys

    cores = np.empty((n, 3))
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        cores[i] = colorsys.hsv_to_rgb(h, 0.55 + 0.4 * ((i * 7) % 3) / 2, 0.95 - 0.3 * ((i * 5) % 4) / 3)
    return cores


def referencial(C, planos):
    """Eixos (ex, ey, ez) das vistas: ez = normal do plano 0 quando ele pesa (>= 8% dos voxels), senao o menor eixo
    do PCA; ex = maior extensao no plano perpendicular."""
    m = gpu.xp()
    S = C[:: max(1, len(C) // 200_000)]
    c0 = S.mean(0)
    w, V = np.linalg.eigh(gpu.para_cpu(m.cov((S - c0).T)))
    if planos and planos[0]["fracao_voxels"] >= 0.08:
        ez = np.asarray(planos[0]["normal"], float)
    else:
        ez = V[:, 0]
    ex = V[:, 2] - (V[:, 2] @ ez) * ez
    if np.linalg.norm(ex) < 1e-6:
        ex = V[:, 1] - (V[:, 1] @ ez) * ez
    ex /= np.linalg.norm(ex)
    return ex, np.cross(ez, ex), ez


def vistas_padrao(ex, ey, ez):
    out = [dict(nome="cima", direcao=(-ez).tolist(), cima=ey.tolist())]
    for az in (30, 120, 210, 300):
        a, e = np.radians(az), np.radians(35)
        d = -(np.cos(e) * np.cos(a) * ex + np.cos(e) * np.sin(a) * ey + np.sin(e) * ez)
        out.append(dict(nome=f"obliqua_{az:03d}", direcao=d.round(6).tolist(), cima=ez.tolist()))
    out.append(dict(nome="baixo", direcao=ez.tolist(), cima=ey.tolist()))
    return out


def _eixos_camera(direcao, cima):
    d = np.asarray(direcao, float)
    d /= np.linalg.norm(d)
    c = np.asarray(cima, float)
    lado = np.cross(d, c)
    if np.linalg.norm(lado) < 1e-6:
        lado = np.cross(d, [1.0, 0, 0] if abs(d[0]) < 0.9 else [0, 1.0, 0])
    lado /= np.linalg.norm(lado)
    return lado, np.cross(lado, d), d


def rasterizar(P, direcao, cima, voxel, tam=1400, janela=None, borda=46):
    """Projecao ortografica com z-buffer. Devolve dict: win (H, W) cupy int64 = indice do ponto visivel (-1 fundo),
    z (H, W) profundidade, ppm, origem da janela e eixos. `janela` = (x0, x1, y0, y1) em mm no plano da camera."""
    m = gpu.xp()
    lado, sobe, d = _eixos_camera(direcao, cima)
    x, y, z = P @ gpu.para_gpu(lado), P @ gpu.para_gpu(sobe), P @ gpu.para_gpu(d)
    if janela is None:  # 0,3% de cada lado fica fora: poeira distante nao pode encolher a peca na imagem
        q = gpu.para_cpu(m.percentile(m.stack([x, y]), gpu.para_gpu(np.array([0.3, 99.7])), axis=1))
        fx, fy = 0.04 * (q[1, 0] - q[0, 0]) + 2 * voxel, 0.04 * (q[1, 1] - q[0, 1]) + 2 * voxel
        janela = (q[0, 0] - fx, q[1, 0] + fx, q[0, 1] - fy, q[1, 1] + fy)
    x0, x1, y0, y1 = janela
    ext = max(x1 - x0, y1 - y0, 1e-6)
    ppm = (tam - 2 * borda) / ext
    W, H = int((x1 - x0) * ppm) + 2 * borda + 1, int((y1 - y0) * ppm) + 2 * borda + 1
    px = m.rint((x - x0) * ppm).astype(m.int64) + borda
    py = m.rint((y1 - y) * ppm).astype(m.int64) + borda
    r = int(min(12, max(0, np.ceil(0.75 * voxel * ppm))))  # discos se sobrepoem: a superficie nao deixa ver o fundo
    offs = [(a, b) for a in range(-r, r + 1) for b in range(-r, r + 1) if a * a + b * b <= (r + 0.5) ** 2]
    zb = m.full(W * H, m.inf, m.float32)
    z32 = z.astype(m.float32)
    ii = m.arange(len(P), dtype=m.int64)
    dentro0 = (px >= -r) & (px < W + r) & (py >= -r) & (py < H + r)
    ii, px, py, z32 = ii[dentro0], px[dentro0], py[dentro0], z32[dentro0]
    for a, b in offs:
        qx, qy = px + a, py + b
        ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
        m.minimum.at(zb, (qy * W + qx)[ok], z32[ok])
    win = m.full(W * H, -1, m.int64)
    for a, b in offs:
        qx, qy = px + a, py + b
        ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
        idx = (qy * W + qx)[ok]
        ganhou = z32[ok] <= zb[idx]
        win[idx[ganhou]] = ii[ok][ganhou]
    return dict(
        win=win.reshape(H, W), z=zb.reshape(H, W), ppm=ppm, janela=janela, borda=borda, eixos=(lado, sobe, d), W=W, H=H
    )


def sombrear(rs, N, voxel, cor=None):
    """Imagem (H, W, 3) uint8 numpy: Lambert com luz de cima-esquerda + luz frontal + escurecimento de borda por
    profundidade (da nocao de frente/fundo). `cor` (n, 3) em 0..1 por ponto, ou None para cinza."""
    m = gpu.xp()
    win, z = rs["win"], rs["z"]
    lado, sobe, d = (gpu.para_gpu(v, m.float64) for v in rs["eixos"])
    vis = win >= 0
    Nw = N[m.maximum(win, 0)]
    sem_normal = ~m.isfinite(Nw).all(-1)
    Nw = m.nan_to_num(Nw)
    Nw = m.where(((Nw * d).sum(-1) > 0)[..., None], -Nw, Nw)  # normal virada para a camera
    luz = -d - 0.55 * lado + 0.65 * sobe
    luz /= m.linalg.norm(luz)
    s = 0.22 + 0.55 * m.clip((Nw * luz).sum(-1), 0, 1) + 0.28 * m.clip(-(Nw * d).sum(-1), 0, 1)
    resp = m.zeros_like(z)
    for dy, dx in ((0, 2), (0, -2), (2, 0), (-2, 0)):
        nb = m.roll(z, (dy, dx), axis=(0, 1))
        ok = m.isfinite(nb) & vis
        resp += m.where(ok, m.maximum(z - nb, 0), 0)
    s = m.where(sem_normal, 0.6, s) * m.exp(-resp / (4 * 6.0 * voxel)).clip(0.35, 1)
    base = m.full(win.shape + (3,), 0.80) if cor is None else cor[m.maximum(win, 0)]
    img = m.where(vis[..., None], base * (0.35 + 0.75 * s[..., None]), 1.0)
    return gpu.para_cpu((m.clip(img, 0, 1) * 255).astype(m.uint8))


def _fonte(tam):
    from PIL import ImageFont

    for nome in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(nome, tam)
        except OSError:
            continue
    return ImageFont.load_default()


def anotar(img, rs, titulo, voxel, rotulos=None, grade=True):
    """Grade A-H x 1-8, barra de escala, titulo e ids das regioes. `rotulos` = lista (texto, coluna_px, linha_px,
    cor) ja ordenada por prioridade; ids que nao cabem sem sobrepor ficam de fora. Devolve (PIL.Image, ids escritos)."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    W, H = im.size
    f1, f2 = _fonte(17), _fonte(14)
    if grade:
        for i in range(1, len(COLUNAS)):
            dr.line([(W * i / len(COLUNAS), 0), (W * i / len(COLUNAS), H)], fill=(0, 90, 200, 55), width=1)
        for i in range(1, LINHAS):
            dr.line([(0, H * i / LINHAS), (W, H * i / LINHAS)], fill=(0, 90, 200, 55), width=1)
        for i, c in enumerate(COLUNAS):
            dr.text((W * (i + 0.5) / len(COLUNAS) - 5, 2), c, fill=(0, 70, 170, 255), font=f2)
        for i in range(LINHAS):
            dr.text((3, H * (i + 0.5) / LINHAS - 8), str(i + 1), fill=(0, 70, 170, 255), font=f2)
    escritos, caixas = [], []
    for texto, cx, cy, cor in rotulos or []:
        bb = dr.textbbox((0, 0), texto, font=f1)
        w, h = bb[2] - bb[0] + 6, bb[3] - bb[1] + 6
        cx0, cy0 = int(np.clip(cx - w / 2, 0, W - w)), int(np.clip(cy - h / 2, 18, H - h))
        cx1, cy1 = cx0 + w, cy0 + h
        if any(not (cx1 < a or cx0 > c or cy1 < b or cy0 > d) for a, b, c, d in caixas):
            continue
        caixas.append((cx0, cy0, cx1, cy1))
        dr.rectangle(
            [cx0, cy0, cx1, cy1], fill=(255, 255, 255, 225), outline=tuple(int(255 * v) for v in cor) + (255,), width=2
        )
        dr.text((cx0 + 3 - bb[0], cy0 + 3 - bb[1]), texto, fill=(0, 0, 0, 255), font=f1)
        escritos.append(texto)
    # barra de escala: valor redondo perto de 1/5 da largura
    alvo = (W / 5) / rs["ppm"]
    passo = 10 ** np.floor(np.log10(alvo))
    mm = min((1, 2, 5, 10), key=lambda k: abs(k * passo - alvo)) * passo
    x0, y0 = 24, H - 20
    dr.rectangle([x0 - 4, y0 - 20, x0 + mm * rs["ppm"] + 70, y0 + 8], fill=(255, 255, 255, 220))
    dr.line([(x0, y0), (x0 + mm * rs["ppm"], y0)], fill=(0, 0, 0, 255), width=3)
    dr.text((x0 + mm * rs["ppm"] + 6, y0 - 9), f"{mm:g} mm", fill=(0, 0, 0, 255), font=f2)
    bb = dr.textbbox((0, 0), titulo, font=f2)
    dr.rectangle([20, 18, 26 + bb[2], 24 + bb[3] + 16], fill=(255, 255, 255, 220))
    dr.text((23, 21), titulo, fill=(0, 0, 0, 255), font=f2)
    dr.text((23, 21 + bb[3] + 2), f"voxel {voxel:g} mm; {rs['ppm']:.1f} px/mm", fill=(60, 60, 60, 255), font=f2)
    return im, escritos


def ancoras(rs, rot_cpu, cores, min_px=120, max_rotulos=90):
    """Onde escrever o id de cada regiao visivel: o pixel da regiao mais proximo do centro dos pixels dela."""
    win = gpu.para_cpu(rs["win"])
    vis = win >= 0
    ids = np.zeros(win.shape, np.int64)
    ids[vis] = rot_cpu[win[vis]]
    yy, xx = np.nonzero(ids > 0)
    r = ids[yy, xx]
    ordem = np.argsort(r, kind="stable")
    yy, xx, r = yy[ordem], xx[ordem], r[ordem]
    u, ini, cnt = np.unique(r, return_index=True, return_counts=True)
    out, px = [], {}
    for k in np.argsort(-cnt):
        px[int(u[k])] = int(cnt[k])
        if cnt[k] < min_px or len(out) >= max_rotulos:
            continue
        ys, xs = yy[ini[k] : ini[k] + cnt[k]], xx[ini[k] : ini[k] + cnt[k]]
        j = np.argmin((ys - ys.mean()) ** 2 + (xs - xs.mean()) ** 2)
        out.append((str(int(u[k])), int(xs[j]), int(ys[j]), cores[int(u[k]) % len(cores)]))
    return out, px, ids


def carregar_atlas(pasta):
    """atlas.npz -> dict com arrays cupy e os escalares; regioes.json -> dict."""
    m = gpu.xp()
    z = np.load(os.path.join(pasta, "atlas.npz"))
    at = {k: gpu.para_gpu(z[k]) for k in ("chaves", "C", "N", "curv", "rot", "comp", "obs", "quadros")}
    at["C"] = at["C"].astype(m.float64)
    at["N"] = at["N"].astype(m.float64)
    at["conf"] = gpu.para_gpu(z["conf"]) if "conf" in z.files else None
    at["voxel"] = float(z["voxel"])
    at["tipo"] = z["tipo"]
    at["grupo"] = z["grupo"]
    with open(os.path.join(pasta, "regioes.json"), encoding="utf-8") as fh:
        info = json.load(fh)
    return at, info


def gravar_atlas(pasta, at):
    extra = {} if at.get("conf") is None else {"conf": gpu.para_cpu(at["conf"]).astype(np.int8)}
    np.savez_compressed(
        os.path.join(pasta, "atlas.npz"),
        chaves=gpu.para_cpu(at["chaves"]),
        C=gpu.para_cpu(at["C"]).astype(np.float32),
        N=gpu.para_cpu(at["N"]).astype(np.float32),
        curv=gpu.para_cpu(at["curv"]).astype(np.float32),
        rot=gpu.para_cpu(at["rot"]).astype(np.int32),
        comp=gpu.para_cpu(at["comp"]).astype(np.int32),
        obs=gpu.para_cpu(at["obs"]).astype(np.float32),
        quadros=gpu.para_cpu(at["quadros"]).astype(np.float32),
        voxel=at["voxel"],
        tipo=np.asarray(at["tipo"]),
        grupo=np.asarray(at["grupo"], np.int64),
        **extra,
    )


def desenhar_vistas(pasta, at, vistas, tam=1400):
    """Grava vistas/<nome>_sombra.png e _regioes.png; devolve {vista: {arquivos, ids escritos, pixels por regiao}}."""
    m = gpu.xp()
    os.makedirs(os.path.join(pasta, "vistas"), exist_ok=True)
    rot_cpu = gpu.para_cpu(at["rot"]).astype(np.int64)
    cores = paleta(64)
    cor = gpu.para_gpu(cores)[at["rot"].astype(m.int64) % 64]
    cor = m.where((at["rot"] == 0)[:, None], 0.55, cor)
    resumo = {}
    for v in vistas:
        rs = rasterizar(at["C"], v["direcao"], v["cima"], at["voxel"], tam)
        a = os.path.join(pasta, "vistas", v["nome"] + "_sombra.png")
        b = os.path.join(pasta, "vistas", v["nome"] + "_regioes.png")
        anotar(sombrear(rs, at["N"], at["voxel"]), rs, v["nome"] + " (forma)", at["voxel"])[0].save(a)
        anc, px, _ = ancoras(rs, rot_cpu, cores)
        im, escritos = anotar(sombrear(rs, at["N"], at["voxel"], cor), rs, v["nome"] + " (regioes)", at["voxel"], anc)
        im.save(b)
        resumo[v["nome"]] = dict(
            sombra=a, regioes=b, ids_escritos=[int(x) for x in escritos], maiores_px={k: px[k] for k in list(px)[:40]}
        )
    return resumo


def resumo_json(scan, at, linhas, planos, vistas, desenho, parametros, top=150):
    m = gpu.xp()
    rot = at["rot"]
    n = len(rot)
    comp_n = gpu.para_cpu(m.bincount(at["comp"].astype(m.int64)))
    cauda = linhas[top:]
    grupos = {}
    for lin in linhas:
        if lin["grupo"]:
            g = grupos.setdefault(
                lin["grupo"], dict(id=lin["grupo"], regioes=[], voxels=0, lo=lin["caixa_min"], hi=lin["caixa_max"])
            )
            g["regioes"].append(lin["id"])
            g["voxels"] += lin["voxels"]
            g["lo"] = np.minimum(g["lo"], lin["caixa_min"]).tolist()
            g["hi"] = np.maximum(g["hi"], lin["caixa_max"]).tolist()
    grupos = sorted(grupos.values(), key=lambda g: -g["voxels"])[:60]
    for g in grupos:
        g["caixa_min"], g["caixa_max"] = g.pop("lo"), g.pop("hi")
        g["n_regioes"] = len(g["regioes"])
        g["regioes"] = g["regioes"][:80]
    por_comp = {}
    for lin in cauda:
        c = por_comp.setdefault(lin["componente"], dict(regioes=0, voxels=0))
        c["regioes"] += 1
        c["voxels"] += lin["voxels"]
    return dict(
        scan=scan,
        parametros=parametros,
        voxel_mm=at["voxel"],
        voxels=n,
        regioes=len(linhas),
        poeira_fracao=round(float((rot == 0).mean()), 4),
        componentes=[dict(id=i, voxels=int(c), fracao=round(float(c / n), 4)) for i, c in enumerate(comp_n[:15])],
        planos=planos,
        grupos_rugosos=grupos,
        vistas=vistas,
        imagens=desenho,
        classes=list(CLASSES),
        regioes_listadas=min(top, len(linhas)),
        tabela=linhas[:top],
        cauda=dict(
            regioes=len(cauda),
            voxels=sum(x["voxels"] for x in cauda),
            por_componente={str(k): v for k, v in sorted(por_comp.items(), key=lambda kv: -kv[1]["voxels"])[:20]},
            arquivo="regioes_todas.json",
        ),
    )


def confirmacao(C, outros, tol):
    """Por voxel: em quantos OUTROS scans (ja no referencial deste) ha ponto a <= tol."""
    m = gpu.xp()
    conf = m.zeros(len(C), m.int8)
    for Q in outros:
        _, j = gpu.vizinho_mais_proximo(Q, C, dist_max=tol)
        conf += (j >= 0).astype(m.int8)
    return conf


def montar_atlas(ac, voxel, a, outros=None):
    """Do acumulado do scan ate o atlas completo (normais, regioes, componentes, tabela)."""
    C = ac["C"]
    N, curv = gpu.normais_pca(C, raio=a.raio_normal * voxel, orientar_para=ac["cam"] - C, com_curvatura=True)
    conf = confirmacao(C, outros, a.tol_confirma) if outros else None
    rot, tipo, grupo, pares = segmentar_regioes(
        C, N, curv, voxel, a.angulo, a.degrau, a.curv_max, a.min_lisa, a.min_frag, a.bloco, conf
    )
    comp, _ = gpu.componentes_conexos_voxel(C, voxel, 26)
    at = dict(chaves=ac["chaves"], C=C, N=N, curv=curv, rot=rot, comp=comp, obs=ac["obs"], quadros=ac["quadros"])
    at.update(voxel=voxel, tipo=tipo, grupo=grupo, conf=conf)
    planos = planos_candidatos(C, voxel)
    return at, tabela_regioes(at, tipo, pares, planos, voxel), planos


def cmd_segmentar(a):
    proj = os.path.abspath(a.projeto)
    _, _, scans = scans_do_projeto(proj, a.scans)
    if not scans:
        sys.exit("nenhum scan encontrado")
    os.makedirs(a.saida, exist_ok=True)
    poses = None
    if a.poses:
        with open(a.poses, encoding="utf-8") as fh:
            poses = {k: np.asarray(v, float) for k, v in json.load(fh).items()}
    m = gpu.xp()
    grossos = {}
    with tempfile.TemporaryDirectory() as tmp:
        if poses:  # nuvem reduzida de cada scan com pose, para a confirmacao cruzada
            _, _, todos = scans_do_projeto(proj)
            for s in todos:
                if s in poses:
                    sc = abrir_scan(proj, s, tmp)
                    grossos[s] = acumular(sc, a.voxel or voxel_automatico(sc), 2, 3)["C"]
                    sc.c.close()
        for s in scans:
            t0 = time.time()
            scan = abrir_scan(proj, s, tmp)
            voxel = a.voxel or voxel_automatico(scan)
            ac = acumular(scan, voxel, a.passo_quadros, a.passo_pixel)
            scan.c.close()
            if ac is None:
                _LOG.warning("%s: sem quadros com profundidade", s)
                continue
            outros = None
            if poses and s in poses:
                inv = np.linalg.inv(poses[s])
                outros = [geom_gpu.transformar(Q, inv @ poses[o], saida_gpu=True) for o, Q in grossos.items() if o != s]
            at, linhas, planos = montar_atlas(ac, voxel, a, outros)
            d = os.path.join(a.saida, s)
            os.makedirs(d, exist_ok=True)
            vistas = vistas_padrao(*referencial(at["C"], planos))
            desenho = desenhar_vistas(d, at, vistas, a.tamanho)
            gravar_atlas(d, at)
            prm = {k: v for k, v in vars(a).items() if k not in ("f", "scans", "projeto", "saida")}
            prm["projeto"] = proj
            salvar_json_atomico(os.path.join(d, "regioes_todas.json"), linhas, indent=0)
            salvar_json_atomico(
                os.path.join(d, "regioes.json"), resumo_json(s, at, linhas, planos, vistas, desenho, prm)
            )
            lisas = sum(1 for x in linhas if x["tipo"] == "lisa")
            _LOG.info(
                "%s: voxel %.2f mm, %s voxels, %d regioes (%d lisas, %d fragmentos), poeira %.1f%%, %.1f s",
                s,
                voxel,
                f"{len(at['rot']):,}",
                len(linhas),
                lisas,
                len(linhas) - lisas,
                100 * float((at["rot"] == 0).mean()),
                time.time() - t0,
            )
            del at, ac
            m.get_default_memory_pool().free_all_blocks() if gpu.em_gpu() else None
    _LOG.info("saida: %s", os.path.abspath(a.saida))


# ---------------------------------------------------------------- ver de perto
def _janela_celula(rs, celula):
    col, lin = COLUNAS.index(celula[0].upper()), int(celula[1:]) - 1
    if not 0 <= lin < LINHAS:
        raise ValueError(f"celula invalida: {celula}")
    x0, x1, y0, y1 = rs["janela"]
    W, H, b, ppm = rs["W"], rs["H"], rs["borda"], rs["ppm"]
    px0, px1 = W * col / len(COLUNAS), W * (col + 1) / len(COLUNAS)
    py0, py1 = H * lin / LINHAS, H * (lin + 1) / LINHAS
    return (x0 + (px0 - b) / ppm, x0 + (px1 - b) / ppm, y1 - (py1 - b) / ppm, y1 - (py0 - b) / ppm)


def cmd_ver(a):
    m = gpu.xp()
    pasta = os.path.abspath(a.atlas)
    at, info = carregar_atlas(pasta)
    C, N, rot = at["C"], at["N"], at["rot"].astype(m.int64)
    voxel = at["voxel"]
    cores = paleta(64)
    saida = a.saida or os.path.join(pasta, "zoom")
    os.makedirs(saida, exist_ok=True)
    vistas = {v["nome"]: v for v in info["vistas"]}
    res = dict(arquivos=[])
    if a.celula:
        if a.vista not in vistas:
            sys.exit(f"--vista deve ser uma de {sorted(vistas)}")
        v = vistas[a.vista]
        geral = rasterizar(C, v["direcao"], v["cima"], voxel, info["parametros"]["tamanho"])
        jan = _janela_celula(geral, a.celula)
        lado, sobe, _ = geral["eixos"]
        x, y = C @ gpu.para_gpu(lado), C @ gpu.para_gpu(sobe)
        mg = a.margem
        sel = (x >= jan[0] - mg) & (x <= jan[1] + mg) & (y >= jan[2] - mg) & (y <= jan[3] + mg)
        jan = (jan[0] - mg, jan[1] + mg, jan[2] - mg, jan[3] + mg)
        rs = rasterizar(C[sel], v["direcao"], v["cima"], voxel, a.tamanho, jan)
        sub = gpu.para_cpu(rot[sel])
        cor = gpu.para_gpu(cores)[rot[sel] % 64]
        cor = m.where((rot[sel] == 0)[:, None], 0.55, cor)
        nome = f"{a.vista}_{a.celula.upper()}"
        anotar(sombrear(rs, N[sel], voxel), rs, nome + " (forma)", voxel, grade=False)[0].save(
            os.path.join(saida, nome + "_sombra.png")
        )
        anc, px, _ = ancoras(rs, sub, cores, min_px=40)
        im, esc = anotar(sombrear(rs, N[sel], voxel, cor), rs, nome + " (regioes)", voxel, anc, grade=False)
        im.save(os.path.join(saida, nome + "_regioes.png"))
        res["arquivos"] = [os.path.join(saida, nome + "_sombra.png"), os.path.join(saida, nome + "_regioes.png")]
        res["regioes_visiveis_px"] = {str(k): v2 for k, v2 in list(px.items())[:40]}
        res["ids_escritos"] = [int(x) for x in esc]
    else:
        alvo = m.zeros(len(C), bool)
        for r in a.regioes or []:
            alvo |= rot == r
        for c in a.componentes or []:
            alvo |= at["comp"] == c
        if a.grupos:
            alvo |= gpu.para_gpu(np.isin(at["grupo"], a.grupos))[rot]
        if not bool(alvo.any()):
            sys.exit("selecao vazia: passe --regioes, --grupos, --componentes ou --vista + --celula")
        lo, hi = C[alvo].min(0) - a.margem, C[alvo].max(0) + a.margem
        perto = alvo if a.isolar else m.all((C >= lo) & (C <= hi), axis=1)
        Cs, Ns, rs_rot, alvo_s = C[perto], N[perto], rot[perto], alvo[perto]
        cor = m.where(alvo_s[:, None], gpu.para_gpu(cores)[rs_rot % 64], 0.78)
        nm = gpu.para_cpu(m.nan_to_num(N[alvo]).mean(0))
        ez = np.asarray(info["vistas"][0]["direcao"]) * -1
        if np.linalg.norm(nm) < 0.2:
            nm = ez
        nm = nm / np.linalg.norm(nm)
        lado, _, _ = _eixos_camera(-nm, ez)
        sobe = np.cross(lado, -nm)
        dirs = [("frente", -nm)]
        for nome_v, ang in (("esquerda", -55), ("direita", 55), ("acima", None), ("abaixo", None)):
            if ang is not None:
                t = np.radians(ang)
                dirs.append((nome_v, -(np.cos(t) * nm + np.sin(t) * lado)))
            else:
                sgn = 1 if nome_v == "acima" else -1
                t = np.radians(55)
                dirs.append((nome_v, -(np.cos(t) * nm + sgn * np.sin(t) * sobe)))
        dirs.append(("verso", nm))
        sub = np.where(gpu.para_cpu(alvo_s), gpu.para_cpu(rs_rot), 0)
        etiqueta = (
            "_".join(str(r) for r in (a.regioes or [])[:4])
            or ("g" + "_".join(str(g) for g in a.grupos[:4]) if a.grupos else "")
            or "comp" + "_".join(str(c) for c in a.componentes[:4])
        )
        for nome_v, d in dirs:
            rs = rasterizar(
                Cs, d, sobe if nome_v in ("frente", "esquerda", "direita", "verso") else nm, voxel, a.tamanho
            )
            anc, px, _ = ancoras(rs, sub, cores, min_px=40)
            arq = os.path.join(saida, f"regiao_{etiqueta}_{nome_v}.png")
            anotar(sombrear(rs, Ns, voxel, cor), rs, f"regioes {etiqueta}: {nome_v}", voxel, anc, grade=False)[0].save(
                arq
            )
            res["arquivos"].append(arq)
        res["pontos_no_recorte"] = int(perto.sum())
        res["voxels_selecionados"] = int(alvo.sum())
        res["caixa_mm"] = dict(min=gpu.para_cpu(lo).round(2).tolist(), max=gpu.para_cpu(hi).round(2).tolist())
        with open(os.path.join(pasta, "regioes_todas.json"), encoding="utf-8") as fh:
            todas = {x["id"]: x for x in json.load(fh)}
        res["medidas"] = [todas[r] for r in (a.regioes or []) if r in todas]
        if a.grupos:  # tabela curta dos blocos: onde estao e em quem encostam
            res["blocos"] = [
                dict(
                    id=x["id"],
                    grupo=x["grupo"],
                    voxels=x["voxels"],
                    centroide=x["centroide"],
                    altura_plano0_mm=x.get("altura_plano0_mm"),
                    quadros_por_voxel=x["quadros_por_voxel"],
                    confirmado=x.get("confirmado_por_1_outro_scan"),
                    encosta_em_lisas=[
                        [v["id"], v["contatos"]] for v in x["vizinhas"] if todas[v["id"]]["tipo"] == "lisa"
                    ][:3],
                )
                for x in todas.values()
                if x["grupo"] in a.grupos
            ]
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return res


# ---------------------------------------------------------------- dividir uma regiao
def cmd_dividir(a):
    m = gpu.xp()
    pasta = os.path.abspath(a.atlas)
    at, info = carregar_atlas(pasta)
    C, N, rot = at["C"], at["N"], at["rot"].astype(m.int64)
    voxel = at["voxel"]
    sel = m.flatnonzero(rot == a.regiao)
    if not len(sel):
        sys.exit(f"regiao {a.regiao} nao existe")
    Q = C[sel]
    if a.plano:
        p, n = np.asarray(a.plano[:3], float), np.asarray(a.plano[3:], float)
        parte = ((Q - gpu.para_gpu(p)) @ gpu.para_gpu(n / np.linalg.norm(n)) > 0).astype(m.int64)
    else:
        # k-means em posicao + normal (a normal pesa `--peso-normal` x a extensao da regiao): separa faces/saliencias
        esc = float(m.linalg.norm(Q.std(0)))
        F = m.concatenate([Q - Q.mean(0), m.nan_to_num(N[sel]) * esc * a.peso_normal], axis=1)
        rng = np.random.default_rng(0)
        cen = F[gpu.para_gpu(rng.choice(len(F), 1))]
        for _ in range(a.partes - 1):  # k-means++ deterministico: o mais distante dos centros atuais
            dmin = m.min(((F[:, None, :] - cen[None, :, :]) ** 2).sum(-1), axis=1)
            cen = m.concatenate([cen, F[int(dmin.argmax())][None, :]])
        for _ in range(40):
            parte = m.argmin(((F[:, None, :] - cen[None, :, :]) ** 2).sum(-1), axis=1)
            novo = m.stack([F[parte == k].mean(0) if bool((parte == k).any()) else cen[k] for k in range(a.partes)])
            if float(m.abs(novo - cen).max()) < 1e-4:
                break
            cen = novo
    base = int(rot.max())
    tam = gpu.para_cpu(m.bincount(parte))
    maior = int(np.argmax(tam))
    novos, tipo, grupo = {}, list(at["tipo"]), list(at["grupo"])
    for k in range(len(tam)):
        if k == maior or tam[k] == 0:
            continue
        base += 1
        rot[sel[parte == k]] = base
        novos[base] = int(tam[k])
        tipo.append(tipo[a.regiao])
        grupo.append(grupo[a.regiao])
    at["rot"], at["tipo"], at["grupo"] = rot.astype(m.int32), np.asarray(tipo), np.asarray(grupo)
    _, I = gpu.Indice(C, 1.8 * voxel).knn(C, 12)
    Ii = m.repeat(m.arange(len(C), dtype=m.int64), I.shape[1])
    I = I.reshape(-1)
    ok = (I >= 0) & (I != Ii)
    linhas = tabela_regioes(at, at["tipo"], (Ii[ok], I[ok]), info["planos"], voxel)
    linhas.sort(key=lambda x: x["id"])
    desenho = desenhar_vistas(pasta, at, info["vistas"], info["parametros"]["tamanho"])
    gravar_atlas(pasta, at)
    salvar_json_atomico(os.path.join(pasta, "regioes_todas.json"), linhas, indent=0)
    ordenadas = sorted(linhas, key=lambda x: -x["voxels"])
    salvar_json_atomico(
        os.path.join(pasta, "regioes.json"),
        resumo_json(info["scan"], at, ordenadas, info["planos"], info["vistas"], desenho, info["parametros"]),
    )
    res = dict(regiao=a.regiao, fica_com=int(tam[maior]), novas={str(k): v for k, v in novos.items()})
    print(json.dumps(res, indent=1))
    return res


# ---------------------------------------------------------------- decisao -> classe por voxel
def _classe(v):
    c = v["classe"] if isinstance(v, dict) else v
    if c not in COD:
        raise ValueError(f"classe desconhecida: {c!r} (use {', '.join(CLASSES)})")
    return COD[c]


def classes_por_voxel(at, info, dec):
    """Codigo de classe (indice em CLASSES) por voxel, pela precedencia corte > regiao > caixa > grupo > componente > padrao."""
    m = gpu.xp()
    rot, comp = at["rot"].astype(m.int64), at["comp"].astype(m.int64)
    R = int(rot.max())
    lut = np.full(R + 1, _classe(dec.get("padrao", "duvida")), np.int8)
    lut[0] = _classe(dec.get("poeira", "ruido"))
    cls = gpu.para_gpu(lut)[rot]
    for c, v in (dec.get("componentes") or {}).items():
        cls = m.where((comp == int(c)) & (rot > 0), _classe(v), cls)
    for g, v in (dec.get("grupos") or {}).items():
        cls = m.where(gpu.para_gpu(at["grupo"] == int(g))[rot], _classe(v), cls)
    for cx in dec.get("caixas") or []:
        dentro = m.all(
            (at["C"] >= gpu.para_gpu(cx["min"], m.float64)) & (at["C"] <= gpu.para_gpu(cx["max"], m.float64)), axis=1
        )
        if cx.get("grupo") is not None:  # caixa restrita a um grupo rugoso
            dentro &= gpu.para_gpu(at["grupo"] == int(cx["grupo"]))[rot]
        cls = m.where(dentro & (rot > 0), _classe(cx), cls)
    lut2 = np.full(R + 1, -1, np.int8)
    for r, v in (dec.get("regioes") or {}).items():
        if not 0 < int(r) <= R:
            raise ValueError(f"regiao {r} nao existe (1..{R})")
        lut2[int(r)] = _classe(v)
    l2 = gpu.para_gpu(lut2)[rot]
    cls = m.where(l2 >= 0, l2, cls)
    protegidas = m.zeros(R + 1, bool)
    protegidas[gpu.para_gpu(np.asarray(dec.get("proteger") or [], np.int64))] = True
    for corte in dec.get("cortes") or []:
        p = corte["plano"]
        if isinstance(p, int):
            p = info["planos"][p]["normal"] + [info["planos"][p]["d"]]
        h = at["C"] @ gpu.para_gpu(p[:3], m.float64) + float(p[3])
        fora = (h < float(corte["abaixo_mm"])) & ~protegidas[rot]
        cls = m.where(fora, _classe(corte.get("classe", "apoio")), cls)
    if dec.get("ilhas"):  # do que fica, so o maior componente conexo; as ilhas recebem a classe pedida
        fica = m.flatnonzero((cls == COD["peca"]) | (cls == COD["duvida"]))
        if len(fica):
            ilha, _ = gpu.componentes_conexos_voxel(
                at["C"][fica], at["voxel"] * float(dec.get("ilhas_voxels", 1.0)), 26
            )
            cls[fica[ilha > 0]] = _classe(dec["ilhas"])
    return cls.astype(m.int8)


def _ler_rotulos(caminho):
    with open(caminho, encoding="utf-8") as fh:
        return json.load(fh)


def resumo_classes(cls, at):
    m = gpu.xp()
    n = gpu.para_cpu(m.bincount(cls.astype(m.int64), minlength=len(CLASSES)))
    o = gpu.para_cpu(m.bincount(cls.astype(m.int64), weights=at["obs"].astype(m.float64), minlength=len(CLASSES)))
    return {
        c: dict(voxels=int(n[i]), fracao_observacoes=round(float(o[i] / max(o.sum(), 1)), 4))
        for i, c in enumerate(CLASSES)
    }


def cmd_sugerir(a):
    """Rascunho de rotulos a partir das medidas (ponto de partida para a revisao visual, nao decisao final):
    com --poses no atlas, regiao confirmada por >= 2 outros scans em >= --conf-peca dos voxels = peca; confirmada por
    1 outro em <= --conf-lixo = apoio (deitada no plano 0) ou residuo; o resto = duvida, inclusive regiao confirmada
    mas deitada no plano 0 (pode ser a mesa repetida entre scans de mesma pose ou a face plana da peca). Sem confirmacao medida, so o
    plano 0 vira apoio e todo o resto fica em duvida."""
    raiz = os.path.abspath(a.atlas)
    saida = {}
    for s in sorted(os.listdir(raiz)):
        if not os.path.exists(os.path.join(raiz, s, "regioes_todas.json")):
            continue
        if a.por_componente:
            saida[s] = dict(
                padrao="residuo",
                poeira="ruido",
                componentes={"0": "peca"},
                motivo="rascunho de `sugerir --por-componente`: maior componente = peca; REVISAR com `previa`",
            )
            _LOG.info("%s: rascunho por componente (componente 0 = peca, resto = residuo)", s)
            continue
        with open(os.path.join(raiz, s, "regioes_todas.json"), encoding="utf-8") as fh:
            linhas = json.load(fh)
        reg, cont = {}, dict.fromkeys(CLASSES, 0)
        for x in linhas:
            c2, c1 = x.get("confirmado_por_2_outros_scans"), x.get("confirmado_por_1_outro_scan")
            no_plano = x.get("fracao_no_plano0", 0) > 0.8
            if c2 is None:
                cl = "apoio" if no_plano else "duvida"
            elif c2 >= a.conf_peca:  # mesa fixa em relacao a peca tambem repete entre scans: decidir olhando
                cl = "duvida" if no_plano else "peca"
            elif c1 <= a.conf_lixo:
                cl = "apoio" if no_plano else "residuo"
            else:
                cl = "duvida"
            reg[str(x["id"])] = cl
            cont[cl] += x["voxels"]
        saida[s] = dict(
            padrao="duvida", poeira="ruido", regioes=reg, motivo="rascunho de `sugerir`: revisar com `previa`"
        )
        if a.ilhas:
            saida[s]["ilhas"] = "ruido"
        if a.corte_placa_mm > 0:
            with open(os.path.join(raiz, s, "regioes.json"), encoding="utf-8") as fh:
                planos = json.load(fh)["planos"]
            p0 = planos[0] if planos else None
            if p0 and p0["area_mm2"] >= a.placa_area_min and p0["fracao_voxels"] >= 0.2:
                saida[s]["cortes"] = [dict(plano=0, abaixo_mm=a.corte_placa_mm, classe="apoio")]
                _LOG.info(
                    "%s: plano 0 com %.0f mm2 tratado como placa de apoio (corte a %.1f mm)",
                    s,
                    p0["area_mm2"],
                    a.corte_placa_mm,
                )
            else:
                _LOG.info("%s: sem placa de apoio pelo criterio de area (plano 0: %s mm2)", s, p0 and p0["area_mm2"])
        _LOG.info("%s: voxels por classe sugerida %s", s, {k: v for k, v in cont.items() if v})
    salvar_json_atomico(a.saida, saida, indent=0)
    _LOG.info("rascunho: %s", os.path.abspath(a.saida))
    return saida


def cmd_previa(a):
    m = gpu.xp()
    dec = _ler_rotulos(a.rotulos)
    saida = {}
    for s, d in dec.items():
        pasta = os.path.join(os.path.abspath(a.atlas), s)
        at, info = carregar_atlas(pasta)
        cls = classes_por_voxel(at, info, d)
        cor_cls = np.array([[0.78, 0.78, 0.78]] + [[0.90, 0.15, 0.12]] * 4 + [[0.98, 0.80, 0.10]])
        cor = gpu.para_gpu(cor_cls)[cls.astype(m.int64)]
        os.makedirs(os.path.join(pasta, "previa"), exist_ok=True)
        arqs = []
        fica = (cls == COD["peca"]) | (cls == COD["duvida"])
        for v in info["vistas"]:
            rs = rasterizar(at["C"], v["direcao"], v["cima"], at["voxel"], info["parametros"]["tamanho"])
            arq = os.path.join(pasta, "previa", v["nome"] + "_previa.png")
            anotar(
                sombrear(rs, at["N"], at["voxel"], cor),
                rs,
                v["nome"] + ": cinza fica, vermelho sai, amarelo duvida",
                at["voxel"],
            )[0].save(arq)
            rs2 = rasterizar(at["C"][fica], v["direcao"], v["cima"], at["voxel"], info["parametros"]["tamanho"])
            arq2 = os.path.join(pasta, "previa", v["nome"] + "_resultado.png")
            anotar(sombrear(rs2, at["N"][fica], at["voxel"]), rs2, v["nome"] + ": o que fica", at["voxel"])[0].save(
                arq2
            )
            arqs += [arq, arq2]
        rc = resumo_classes(cls, at)
        comp_f, tam_f = (
            gpu.componentes_conexos_voxel(at["C"][fica], at["voxel"], 26) if bool(fica.any()) else (None, [])
        )
        saida[s] = dict(
            classes=rc,
            pendente=rc["duvida"]["voxels"] > 0,
            componentes_que_ficam=[int(x) for x in gpu.para_cpu(tam_f)[:8]] if comp_f is not None else [],
            imagens=arqs,
        )
    print(json.dumps(saida, indent=1, ensure_ascii=False))
    return saida


def cmd_aplicar(a):
    m = gpu.xp()
    proj, atlas, out = os.path.abspath(a.projeto), os.path.abspath(a.atlas), os.path.abspath(a.saida)
    if os.path.exists(out):
        sys.exit(f"destino ja existe: {out}")
    if os.path.commonpath([proj, out]) == proj:
        sys.exit("o destino nao pode ficar dentro do projeto de origem")
    revo, nos, scans = scans_do_projeto(proj)
    dec = _ler_rotulos(a.rotulos)
    faltam = [s for s in dec if s not in scans]
    if faltam:
        sys.exit(f"scans dos rotulos que nao estao no projeto: {faltam}")
    os.makedirs(os.path.join(out, "data"))
    rel, pendente = {}, False
    original = RP.mascara_manter
    try:
        for s in scans:
            src = os.path.join(proj, "data", s)
            if s not in dec:
                shutil.copytree(src, os.path.join(out, "data", s))
                _LOG.info("%s: sem rotulos, copiado intacto", s)
                continue
            at, info = carregar_atlas(os.path.join(atlas, s))
            voxel = at["voxel"]
            cls = classes_por_voxel(at, info, dec[s])
            fica_v = (cls == COD["peca"]) | (cls == COD["duvida"])
            chaves, org = at["chaves"], origem_grade(voxel)
            ind = gpu.Indice(at["C"], 1.5 * voxel)
            cont = dict(fora_do_atlas=0)

            def mascara(Pw, seg, chaves=chaves, fica_v=fica_v, ind=ind, cont=cont, voxel=voxel, org=org):
                na_gpu = RP._eh_cupy(Pw)
                Pg = gpu.para_gpu(Pw, m.float64)
                k = gpu.chaves_voxel(Pg, voxel, org)
                pos = m.clip(m.searchsorted(chaves, k), 0, len(chaves) - 1)
                achou = chaves[pos] == k
                ok = achou & fica_v[pos]
                if not bool(achou.all()):  # pixel que a amostragem do atlas nao voxelizou: herda do voxel mais proximo
                    falta = m.flatnonzero(~achou)
                    _, j = ind.mais_proximo(Pg[falta])
                    ok[falta] = (j >= 0) & fica_v[m.maximum(j, 0)]
                    cont["fora_do_atlas"] += int((j < 0).sum())
                return ok if na_gpu else gpu.para_cpu(ok)

            RP.mascara_manter = mascara
            log = []
            with tempfile.TemporaryDirectory() as tmp:
                scan = abrir_scan(proj, s, tmp)
                r = RP.limpar_scan(
                    scan,
                    {},
                    os.path.join(out, "data", s),
                    log=lambda *x, log=log: log.append(" ".join(map(str, x))),
                    pixels=False,
                    ilhas=False,
                    min_mantido=0.0,
                    sem_ply_antigos=True,
                    zerar_vazios=True,
                    min_pix_quadro=a.min_pix_quadro,
                )
                scan.c.close()
            rc = resumo_classes(cls, at)
            pendente |= rc["duvida"]["voxels"] > 0
            rel[s] = dict(
                pontos_antes=r["antes"],
                pontos_depois=r["depois"],
                fracao_mantida=round(r["depois"] / max(r["antes"], 1), 4),
                quadros_esvaziados=r["vazios"],
                integridade=r["integridade"],
                pixels_fora_do_atlas_removidos=cont["fora_do_atlas"],
                classes=rc,
                pendente=rc["duvida"]["voxels"] > 0,
            )
            _LOG.info(
                "%s: %s -> %s pontos (%.1f%% mantidos); %d quadros esvaziados; duvida %s voxels",
                s,
                f"{r['antes']:,}",
                f"{r['depois']:,}",
                100 * rel[s]["fracao_mantida"],
                r["vazios"],
                f"{rc['duvida']['voxels']:,}",
            )
    finally:
        RP.mascara_manter = original
    gravar_revo(os.path.join(proj, revo), out, os.path.splitext(revo)[0], [n["guid"] for n in nos])
    shutil.copy2(a.rotulos, os.path.join(out, "rotulos_aplicados.json"))
    salvar_json_atomico(
        os.path.join(out, "atlas_aplicacao.json"), dict(origem=proj, atlas=atlas, scans=rel, parcial=pendente)
    )
    _LOG.info("gravado: %s%s", out, " | PARCIAL: ha regioes em duvida" if pendente else "")
    return 2 if pendente else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("segmentar", help="divide cada scan em regioes, mede e desenha")
    p.add_argument("projeto")
    p.add_argument("--saida", required=True, help="pasta do atlas (fora do projeto)")
    p.add_argument("--scans", nargs="+", help="guids; padrao = todos")
    p.add_argument("--poses", help="JSON {guid: 4x4 scan->referencia}: mede confirmacao por outros scans")
    p.add_argument("--voxel", type=float, default=0.0, help="mm; 0 = automatico pela extensao da cena (0,3 a 2)")
    p.add_argument("--passo-quadros", type=int, default=1)
    p.add_argument("--passo-pixel", type=int, default=1)
    p.add_argument("--raio-normal", type=float, default=2.5, help="raio da normal, em voxels")
    p.add_argument("--angulo", type=float, default=18.0, help="graus entre normais vizinhas dentro de um retalho liso")
    p.add_argument("--degrau", type=float, default=0.5, help="degrau maximo entre vizinhos, em voxels")
    p.add_argument("--curv-max", type=float, default=0.06, help="curvatura acima disso e quina: nao cresce por ela")
    p.add_argument("--min-lisa", type=int, default=25, help="voxels minimos de um retalho liso")
    p.add_argument("--min-frag", type=int, default=8, help="voxels minimos de um grupo rugoso; menor = poeira")
    p.add_argument("--bloco", type=float, default=4.0, help="mm; lado dos blocos em que um grupo rugoso e cortado")
    p.add_argument("--tol-confirma", type=float, default=0.6, help="mm para contar ponto de outro scan")
    p.add_argument("--tamanho", type=int, default=1400, help="lado maior das imagens, px")
    p.set_defaults(f=cmd_segmentar)
    p = sub.add_parser("ver", help="imagens de perto de regioes ou de uma celula da grade")
    p.add_argument("atlas", help="pasta <saida>/<scan>")
    p.add_argument("--regioes", nargs="+", type=int)
    p.add_argument("--grupos", nargs="+", type=int, help="grupos rugosos (cada bloco do grupo aparece com seu id)")
    p.add_argument("--componentes", nargs="+", type=int)
    p.add_argument("--vista")
    p.add_argument("--celula", help="celula da grade da vista, ex. C4")
    p.add_argument("--margem", type=float, default=4.0, help="mm em volta da selecao")
    p.add_argument("--isolar", action="store_true", help="mostra so a selecao (sem o entorno)")
    p.add_argument("--tamanho", type=int, default=1200)
    p.add_argument("--saida", help="pasta das imagens; padrao <atlas>/zoom")
    p.set_defaults(f=cmd_ver)
    p = sub.add_parser("dividir", help="separa uma regiao em partes (k-means posicao+normal, ou plano)")
    p.add_argument("atlas")
    p.add_argument("--regiao", type=int, required=True)
    p.add_argument("--partes", type=int, default=2)
    p.add_argument("--peso-normal", type=float, default=0.5)
    p.add_argument("--plano", nargs=6, type=float, metavar=("PX", "PY", "PZ", "NX", "NY", "NZ"))
    p.set_defaults(f=cmd_dividir)
    p = sub.add_parser("sugerir", help="rascunho de rotulos pelas medidas (revisar antes de aplicar)")
    p.add_argument("atlas", help="pasta do atlas (com uma subpasta por scan)")
    p.add_argument("--saida", required=True, help="rotulos.json a gravar")
    p.add_argument("--conf-peca", type=float, default=0.7)
    p.add_argument("--conf-lixo", type=float, default=0.15)
    p.add_argument("--ilhas", action="store_true", help="do que fica, so o maior componente conexo")
    p.add_argument(
        "--por-componente",
        action="store_true",
        help="rascunho sem medir confirmacao: o maior componente conexo de cada scan e a peca e o resto e residuo "
        "(1 ou 2 scans; REVISAR nas imagens de `previa`: apoio encostado na peca vem junto)",
    )
    p.add_argument(
        "--corte-placa-mm",
        type=float,
        default=0.0,
        help="> 0: corta tudo ate essa altura do plano 0 quando ele e a placa de apoio (criterio de area)",
    )
    p.add_argument(
        "--placa-area-min",
        type=float,
        default=3000.0,
        help="mm2: plano 0 maior que isso (e >= 20%% dos voxels) e placa; peca com face plana grande pede valor maior",
    )
    p.set_defaults(f=cmd_sugerir)
    p = sub.add_parser("previa", help="mostra o efeito dos rotulos sem tocar nos quadros")
    p.add_argument("atlas", help="pasta do atlas (com uma subpasta por scan)")
    p.add_argument("--rotulos", required=True)
    p.set_defaults(f=cmd_previa)
    p = sub.add_parser("aplicar", help="apaga dos quadros (copia nova) o que nao e peca")
    p.add_argument("projeto")
    p.add_argument("atlas")
    p.add_argument("--rotulos", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--min-pix-quadro", type=int, default=200)
    p.set_defaults(f=cmd_aplicar)
    a = ap.parse_args(argv)
    baixa_prioridade()
    return a.f(a)


if __name__ == "__main__":
    _r = main()
    sys.exit(_r if isinstance(_r, int) else 0)
