"""Ciclo fusao -> uniao -> analise -> correcao: tira de cada scan os trechos que levam defeito para a uniao.

    python pipeline/ciclo_uniao.py <projeto_limpo> --poses poses.json --saida D:/Projects/X_ciclo [--rodadas 3]

O que faz, por rodada:
  1. FUSAO de cada scan (propria, GPU): centroides de --voxel mm dos quadros, com normal voltada para a camera.
  2. UNIAO: todos no referencial comum pelas poses ({guid: 4x4 scan->referencia}, ex. Merge Alignment do Revo).
  3. ANALISE: em cada ponto de cada scan, a altura media local (raio --raio, ao longo da normal) da superficie de
     cada OUTRO scan que cobre o ponto (>= --viz-min vizinhos, normal de mesmo lado). Se >= --min-scans outros scans
     concordam entre si (a menos de --acordo mm) e este scan esta a mais de --tol mm deles, o trecho e DEFEITO deste
     scan: e ele que engrossa/deforma a uniao ali, e os outros cobrem melhor.
  4. CORRECAO: manchas de defeito (maioria local, area >= --area-min mm2) saem da superficie do scan; a rodada
     seguinte mede de novo. Onde nao ha maioria (so 1 outro scan, ou os outros discordam) NADA sai: vira pendencia.
No fim os pixels dos quadros que caem nos voxels com defeito sao apagados numa COPIA nova (poses intactas), a fusao
propria e refeita a partir dos quadros gravados e a analise e repetida (antes x depois no relatorio e nas imagens).
Scan ruim INTEIRO (pouca cobertura; no P9 o 093808, 176 quadros, 9,6% fora do consenso contra 0,8-3,9% dos outros)
nao se remenda: `--descartar-ruins` (ou `--descartar guid`) tira o scan antes da analise, para ele nem votar. `--unir
NOME` grava no fim `<saida>/unido`, um scan unico com os quadros dos scans aceitos ja alinhados, pronto para o Revo.
Travas: scan com mais de --frac-max dos pontos cobertos em defeito nao e cortado (indica POSE errada, nao defeito
local); ponto visto so por este scan nunca sai; nenhum furo e fechado.
Saida: <saida>/projeto (copia com os quadros corrigidos), relatorio.json, imagens/<scan>_antes.png e _depois.png
(cinza = concorda, amarelo = desvio > tol/2, vermelho = defeito removido, azul = so este scan cobre).
Com --fusoes PASTA (<guid>.ply = fusao do Revo de cada scan) a analise roda sobre as fusoes REAIS: pixel sai quando o
ponto fundido com defeito mais proximo (<= --alcance) esta mais perto que o ponto fundido bom; o "depois" do
relatorio e previsao e so a nova fusao no Revo confirma.
Pose: `vies_de_pose` = mediana COM SINAL do desvio ao consenso por celula; acima de --vies-max a pose vira pendencia
(erro de pose pequeno nao vira "defeito", vira borrao na fusao). Com --duvida-atlas/--duvida-rotulos as regioes que a
limpeza do atlas deixou em duvida sao decididas aqui: confirmada por >= 2 scans vira peca; o resto continua em duvida
(magenta nas imagens) e so e pendencia acima de --duvida-max-pct da superficie do scan.
Limite: a fusao propria por centroides nao e a do Revo (no P9 so 60% dos pontos ruins da fusao do Revo tinham
superficie propria por perto); ela serve para LOCALIZAR defeito sem abrir o Revo. Preferir --fusoes quando houver. Codigo 2 = ha pendencias (pose suspeita, pose com vies ou duvida acima do limite).
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import atlas_partes as AP
import gpu
import numpy as np
import revo_projeto as RP
from comum import configurar_log, salvar_json_atomico
from recursos import baixa_prioridade

_LOG = configurar_log(__name__)


def fundir_scan(proj, s, tmp, voxel, min_obs, T):
    """Superficie propria do scan: centroides de `voxel` mm com >= min_obs observacoes. Devolve dict cupy com
    chaves (grade do scan), P e N no referencial comum, obs."""
    m = gpu.xp()
    scan = AP.abrir_scan(proj, s, tmp)
    ac = AP.acumular(scan, voxel)
    scan.c.close()
    if ac is None:
        return None
    ok = ac["obs"] >= min_obs
    C = ac["C"][ok]
    N = gpu.normais_pca(C, raio=3.0 * voxel, orientar_para=ac["cam"][ok] - C)
    bom = m.isfinite(N).all(1)
    R, t = gpu.para_gpu(T[:3, :3].T, m.float64), gpu.para_gpu(T[:3, 3], m.float64)
    return dict(chaves=ac["chaves"][ok][bom], P=C[bom] @ R + t, N=N[bom] @ R, obs=ac["obs"][ok][bom])


def altura_local(alvo, P, N, raio, hmax, assinado, k=64, ind=None):
    """Altura media (ao longo de N) da superficie de `alvo` no cilindro de raio `raio` e meia-altura `hmax` em torno
    de cada P; n de vizinhos e cosseno medio. A busca ao longo da normal acha a superficie do outro scan mesmo
    quando ela esta deslocada mais que `raio` (defeito grande). `assinado`: as normais dos dois lados tem sentido
    confiavel, entao so vale vizinho de MESMA face (parede fina nao se confunde com o verso). `ind`: indice ja
    montado sobre alvo["P"] com alcance >= hypot(raio, hmax). Tudo cupy."""
    m = gpu.xp()
    off = m.zeros(len(P))
    cnt = m.zeros(len(P), m.int32)
    cs = m.zeros(len(P))
    if not len(alvo["P"]) or not len(P):
        return off, cnt, cs
    if ind is None:
        ind = gpu.Indice(alvo["P"], float(np.hypot(raio, hmax)))
    for a, b in gpu.blocos(len(P), bytes_por_item=k * 120):
        _, j = ind.knn(P[a:b], k)
        jj = m.maximum(j, 0)
        v = alvo["P"][jj] - P[a:b, None, :]
        d = (v * N[a:b, None, :]).sum(-1)
        c = (alvo["N"][jj] * N[a:b, None, :]).sum(-1)
        val = (j >= 0) & ((v * v).sum(-1) - d * d <= raio * raio) & (m.abs(d) <= hmax)
        if assinado:
            val &= c > 0
        else:
            c = m.abs(c)
        n = val.sum(1)
        off[a:b] = m.where(val, d, 0).sum(1) / m.maximum(n, 1)
        cs[a:b] = m.where(val, c, 0).sum(1) / m.maximum(n, 1)
        cnt[a:b] = n
    return off, cnt, cs


class Consenso:
    """Analise do consenso entre scans com memoria: a altura de cada scan sobre cada outro (par s, t) e calculada
    uma vez; quando um scan perde pontos (`remover`) so e refeita a parte das consultas que tinha um ponto removido
    ao alcance. O resultado e identico ao de recalcular tudo (a vizinhanca de quem nao tinha ponto removido por
    perto nao muda), mas as rodadas seguintes custam uma fracao da primeira."""

    def __init__(self, sup, a, vivos=None):
        m = gpu.xp()
        self.sup, self.a = sup, a
        self.vivos = vivos if vivos is not None else {s: m.ones(len(x["P"]), bool) for s, x in sup.items()}
        self._par, self._atual, self._ind = {}, {}, {}

    def _hmax(self, s, t):
        a = self.a
        ass = s in a.orientados and t in a.orientados
        return (a.raio if t == s or not ass else a.alcance_normal), (s in a.orientados if t == s else ass)

    def _alvo(self, t, alcance):
        if t not in self._atual:
            self._atual[t] = {k: v[self.vivos[t]] for k, v in self.sup[t].items()}
        if (t, alcance) not in self._ind and len(self._atual[t]["P"]):
            self._ind[t, alcance] = gpu.Indice(self._atual[t]["P"], alcance)
        return self._atual[t], self._ind.get((t, alcance))

    def _calc(self, s, t, sel=None):
        hmax, ass = self._hmax(s, t)
        alvo, ind = self._alvo(t, float(np.hypot(self.a.raio, hmax)))
        P, N = self.sup[s]["P"], self.sup[s]["N"]
        if sel is not None:
            P, N = P[sel], N[sel]
        return altura_local(alvo, P, N, self.a.raio, hmax, ass, ind=ind)

    def par(self, s, t):
        if (s, t) not in self._par:
            self._par[s, t] = list(self._calc(s, t))
        return self._par[s, t]

    def remover(self, t, mascara):
        """Tira de `t` os pontos de `mascara` (bool sobre todos os pontos de t) e atualiza os pares ja calculados."""
        m = gpu.xp()
        sai = mascara & self.vivos[t]
        if not bool(sai.any()):
            return 0
        Prem = self.sup[t]["P"][sai]
        self.vivos[t] = self.vivos[t] & ~sai
        self._atual.pop(t, None)
        for k in [k for k in self._ind if k[0] == t]:
            del self._ind[k]
        for s, t2 in list(self._par):
            if t2 != t:
                continue
            hmax, _ = self._hmax(s, t)
            alc = float(np.hypot(self.a.raio, hmax)) * 1.001 + 1e-9
            sel = m.flatnonzero(gpu.vizinho_mais_proximo(Prem, self.sup[s]["P"], dist_max=alc)[1] >= 0)
            if len(sel):
                for velho, novo in zip(self._par[s, t], self._calc(s, t, sel), strict=True):
                    velho[sel] = novo
        return int(sai.sum())

    def descartar(self, t):
        """Scan sai inteiro: deixa de ser analisado e de votar."""
        self.sup.pop(t, None)
        self.vivos.pop(t, None)
        self._atual.pop(t, None)
        self._ind = {k: v for k, v in self._ind.items() if k[0] != t}
        self._par = {k: v for k, v in self._par.items() if t not in k}

    def analisar(self, alvos=None):
        """Por scan (`alvos`, padrao todos): desvio em relacao ao consenso dos outros, usando so os pontos vivos.
        Devolve {s: dict(desvio, n_cobrem, consenso, defeito, cobre, devs, de)}; `cobre` e `devs` = matrizes
        (ponto, outro scan) de cobertura e de desvio; `de` = ids dos outros scans na ordem das colunas."""
        m = gpu.xp()
        a, ids, out = self.a, list(self.sup), {}
        for s in alvos or ids:
            P, vivo = self.sup[s]["P"], self.vivos[s]
            proprio = self.par(s, s)[0]
            de = [t for t in ids if t != s]
            devs, cobre = [], []
            for t in de:
                off, cnt, cs = self.par(s, t)
                devs.append(off - proprio)
                cobre.append((cnt >= a.viz_min) & (cs > a.cos_min))
            D, K = m.stack(devs, 1), m.stack(cobre, 1)
            n_cob = K.sum(1)
            med = m.nanmedian(m.where(K, D, m.nan), axis=1)
            med = m.where(n_cob > 0, med, 0.0)
            concordam = (K & (m.abs(D - med[:, None]) < a.acordo)).sum(1)
            consenso = concordam >= a.min_scans
            bruto = consenso & (m.abs(med) > a.tol) & vivo
            # mancha: maioria dos vizinhos (2 voxels) tambem em defeito; depois area minima por componente
            defeito = m.zeros(len(P), bool)
            if bool(bruto.any()):
                r = 2.0 * a.voxel
                tot = gpu.contar_no_raio(P[vivo], P, r)
                ruim = gpu.contar_no_raio(P[bruto], P, r)
                cand = vivo & consenso & (m.abs(med) > 0.5 * a.tol) & (ruim >= 0.5 * m.maximum(tot, 1))
                if bool(cand.any()):
                    ic = m.flatnonzero(cand)
                    comp, tam = gpu.componentes_conexos_voxel(P[ic], 2.0 * a.voxel, 26)
                    defeito[ic[tam[comp] * a.voxel * a.voxel >= a.area_min]] = True
            out[s] = dict(desvio=med, n_cobrem=n_cob, consenso=consenso, defeito=defeito, cobre=K, devs=D, de=de)
        return out


def analisar(sup, vivos, a, alvos=None):
    """Analise avulsa (sem memoria entre chamadas); ver Consenso.analisar."""
    return Consenso(sup, a, vivos).analisar(alvos)


def vies_por_celula(P, desvio, sel, cel, min_n=200):
    """Desvio COM SINAL em relacao ao consenso, mediana por celula de `cel` mm (so pontos `sel`). Pose errada aparece
    como celulas inteiras deslocadas para o mesmo lado, mesmo quando o desvio e pequeno demais para virar defeito."""
    from metricas import _mediana_por_grupo

    m = gpu.xp()
    Ps, r = P[sel], desvio[sel]
    if len(Ps) < min_n:
        return dict(celulas=0)
    org = m.floor(Ps.min(0) / cel) * cel
    _, inv, cnt = gpu.agrupar(gpu.chaves_voxel(Ps, cel, org))
    med = _mediana_por_grupo(r, inv, len(cnt))[cnt >= min_n]
    if not len(med):
        return dict(celulas=0)
    return dict(
        celula_mm=round(float(cel), 2),
        celulas=int(len(med)),
        vies_global_mm=round(float(m.median(r)), 4),
        vies_min_mm=round(float(med.min()), 4),
        vies_max_mm=round(float(med.max()), 4),
        vies_abs_max_mm=round(float(m.abs(med).max()), 4),
    )


def escolher_descarte(pct, minimo, fator):
    """Scan que sai inteiro: o de maior %% fora do consenso, se passa de `minimo` e de `fator` x a mediana dos outros.
    Devolve (scan, mediana dos outros) ou None."""
    pior = max(pct, key=pct.get)
    resto = float(np.median([v for s, v in pct.items() if s != pior]))
    if pct[pior] < minimo or pct[pior] < fator * max(resto, 1e-6):
        return None
    return pior, resto


def ruido_entre_scans(an, vivos):
    """Mediana e p90 de |desvio| nos pontos com consenso de todos os scans juntos (mm)."""
    m = gpu.xp()
    ad = m.concatenate([m.abs(x["desvio"][x["consenso"] & vivos[s]]) for s, x in an.items()])
    if not len(ad):
        return None
    return dict(p50_mm=round(float(m.percentile(ad, 50)), 4), p90_mm=round(float(m.percentile(ad, 90)), 4))


def duvidas_do_atlas(atlas, dec, s, P, T):
    """Para cada ponto P (referencial comum) do scan `s`: id da regiao do atlas que ficou em DUVIDA na limpeza
    (0 = nao e duvida). Devolve (regiao por ponto, ids das regioes deitadas no plano 0, normal do plano 0 no
    referencial comum ou None)."""
    m = gpu.xp()
    pasta = os.path.join(atlas, s)
    if s not in dec or not os.path.exists(os.path.join(pasta, "atlas.npz")):
        return None
    at, info = AP.carregar_atlas(pasta)
    cls = AP.classes_por_voxel(at, info, dec[s])
    Ti = np.linalg.inv(T)
    Pl = P @ gpu.para_gpu(Ti[:3, :3].T) + gpu.para_gpu(Ti[:3, 3])
    k = gpu.chaves_voxel(Pl, at["voxel"], AP.origem_grade(at["voxel"]))
    pos = m.clip(m.searchsorted(at["chaves"], k), 0, len(at["chaves"]) - 1)
    eh = (at["chaves"][pos] == k) & (cls[pos] == AP.COD["duvida"])
    reg = m.where(eh, at["rot"][pos].astype(m.int64), 0)
    deitadas = set()
    arq = os.path.join(pasta, "regioes_todas.json")
    if os.path.exists(arq):
        with open(arq, encoding="utf-8") as fh:
            deitadas = {int(x["id"]) for x in json.load(fh) if x.get("fracao_no_plano0", 0) > 0.8}
    n0 = None
    if info.get("planos"):
        n0 = T[:3, :3] @ np.asarray(info["planos"][0]["normal"], float)
    return reg, deitadas, n0


def resolver_duvidas(x, reg, deitadas, outro_apoio, a):
    """Decide as regioes em duvida de um scan pelo consenso: regiao com >= a.duvida_peca dos pontos confirmados
    (>= a.min_scans outros scans concordam entre si e com este scan dentro de a.tol) vira PECA. Regiao deitada no
    plano de apoio so conta scans de OUTRO apoio (`outro_apoio` = mascara das colunas de x["de"]): a mesa se repete
    entre scans da mesma pose de repouso. O que nao se confirma continua em duvida (nao e apagado: pode ser trecho
    que so este scan viu). Devolve (mascara dos pontos ainda em duvida, resumo)."""
    m = gpu.xp()
    ids = [int(r) for r in gpu.para_cpu(m.unique(reg)) if r > 0]
    pend = m.zeros(len(reg), bool)
    res = dict(regioes=len(ids), viraram_peca=0, pontos=int((reg > 0).sum()), pontos_confirmados=0)
    cols = gpu.para_gpu(np.asarray(outro_apoio, bool))
    conf_todos = x["consenso"] & (m.abs(x["desvio"]) <= a.tol)
    conf_outro = conf_todos
    if not bool(cols.all()):
        if int(cols.sum()) >= a.min_scans:
            K, D = x["cobre"][:, cols], x["devs"][:, cols]
            n = K.sum(1)
            med = m.where(n > 0, m.nanmedian(m.where(K, D, m.nan), axis=1), 0.0)
            conf_outro = ((K & (m.abs(D - med[:, None]) < a.acordo)).sum(1) >= a.min_scans) & (m.abs(med) <= a.tol)
        else:
            conf_outro = m.zeros(len(reg), bool)
    for r in ids:
        sel = reg == r
        f = float((conf_outro if r in deitadas else conf_todos)[sel].mean())
        if f >= a.duvida_peca:
            res["viraram_peca"] += 1
            res["pontos_confirmados"] += int(sel.sum())
        else:
            pend |= sel
    res["pontos_pendentes"] = int(pend.sum())
    return pend, res


def resumo(an, vivos, a):
    m = gpu.xp()
    r = {}
    for s, x in an.items():
        v = vivos[s]
        cons = x["consenso"] & v
        ad = m.abs(x["desvio"][cons])
        r[s] = dict(
            pontos=int(v.sum()),
            com_consenso=int(cons.sum()),
            so_este_scan=int(((x["n_cobrem"] == 0) & v).sum()),
            desvio_p50_mm=round(float(m.percentile(ad, 50)), 4) if len(ad) else None,
            desvio_p90_mm=round(float(m.percentile(ad, 90)), 4) if len(ad) else None,
            acima_tol_pct=round(100 * float((ad > a.tol).mean()), 3) if len(ad) else None,
            sem_maioria=int(((x["n_cobrem"] > 0) & ~x["consenso"] & v).sum()),
        )
    return r


def desenhar(s, sup, vivos, an, removido, caminho, a, duvida=None):
    m = gpu.xp()
    P, N = sup["P"], sup["N"]
    ad = m.abs(an["desvio"])
    cor = m.full((len(P), 3), 0.80)
    cor = m.where((an["n_cobrem"] == 0)[:, None], gpu.para_gpu(np.array([0.45, 0.62, 0.95])), cor)
    cor = m.where((an["consenso"] & (ad > 0.5 * a.tol))[:, None], gpu.para_gpu(np.array([0.98, 0.78, 0.10])), cor)
    cor = m.where((an["consenso"] & (ad > a.tol))[:, None], gpu.para_gpu(np.array([0.98, 0.50, 0.10])), cor)
    if duvida is not None:
        cor = m.where(duvida[:, None], gpu.para_gpu(np.array([0.85, 0.20, 0.85])), cor)
    if removido is not None:
        cor = m.where(removido[:, None], gpu.para_gpu(np.array([0.90, 0.12, 0.10])), cor)
    S = P[:: max(1, len(P) // 100_000)]
    w, V = np.linalg.eigh(gpu.para_cpu(m.cov((S - S.mean(0)).T)))
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    arqs = []
    for nome, d in (("frente", V[:, 0]), ("verso", -V[:, 0])):
        rs = AP.rasterizar(P, d, V[:, 2], a.voxel, 1400)
        arq = caminho.replace(".png", f"_{nome}.png")
        AP.anotar(AP.sombrear(rs, N, a.voxel, cor), rs, f"{s} {os.path.basename(caminho)[:-4]} {nome}", a.voxel)[
            0
        ].save(arq)
        arqs.append(arq)
    return arqs


def aplicar(proj, out, poses, sup, removido, a, excluir=()):
    """Apaga dos quadros (copia nova) os pixels cujo voxel da grade do scan esta em `removido`."""
    m = gpu.xp()
    revo, nos, scans = AP.scans_do_projeto(proj)
    os.makedirs(os.path.join(out, "data"))
    rel = {}
    original = RP.mascara_manter
    org = AP.origem_grade(a.voxel)
    scans = [s for s in scans if s not in excluir]  # scan descartado nao entra na copia
    try:
        for s in scans:
            src = os.path.join(proj, "data", s)
            if s not in sup or not bool(removido[s].any()):
                shutil.copytree(src, os.path.join(out, "data", s))
                rel[s] = dict(pixels_antes=None, pixels_depois=None, copiado=True)
                continue
            Ti = np.linalg.inv(poses[s])
            Pl = sup[s]["P"] @ gpu.para_gpu(Ti[:3, :3].T) + gpu.para_gpu(Ti[:3, 3])  # de volta ao scan
            ruim = Pl[removido[s]]
            tem_chaves = "chaves" in sup[s]
            chaves_ruins = m.sort(sup[s]["chaves"][removido[s]]) if tem_chaves else None
            alcance = 2.0 * a.voxel if tem_chaves else a.alcance
            ind_bom = gpu.Indice(Pl[~removido[s]], alcance)
            ind_ruim = gpu.Indice(ruim, alcance)

            def mascara(Pw, seg, chaves_ruins=chaves_ruins, ind_bom=ind_bom, ind_ruim=ind_ruim):
                na_gpu = RP._eh_cupy(Pw)
                Pg = gpu.para_gpu(Pw, m.float64)
                fora = m.zeros(len(Pg), bool)
                if chaves_ruins is not None:
                    k = gpu.chaves_voxel(Pg, a.voxel, org)
                    pos = m.clip(m.searchsorted(chaves_ruins, k), 0, len(chaves_ruins) - 1)
                    fora = chaves_ruins[pos] == k
                # pixel em voxel ralo (fora da superficie fundida): sai se o defeito esta mais perto que a parte boa
                db, _ = ind_bom.mais_proximo(Pg)
                dr, _ = ind_ruim.mais_proximo(Pg)
                fora |= dr < db
                ok = ~fora
                return ok if na_gpu else gpu.para_cpu(ok)

            RP.mascara_manter = mascara
            with tempfile.TemporaryDirectory() as tmp:
                scan = AP.abrir_scan(proj, s, tmp)
                r = RP.limpar_scan(
                    scan,
                    {},
                    os.path.join(out, "data", s),
                    log=lambda *x: None,
                    pixels=False,
                    ilhas=False,
                    min_mantido=0.0,
                    sem_ply_antigos=True,
                    zerar_vazios=True,
                    min_pix_quadro=200,
                )
                scan.c.close()
            rel[s] = dict(
                pixels_antes=r["antes"],
                pixels_depois=r["depois"],
                apagado_pct=round(100 * (1 - r["depois"] / max(r["antes"], 1)), 3),
                quadros_esvaziados=r["vazios"],
                integridade=r["integridade"],
            )
    finally:
        RP.mascara_manter = original
    RP.gravar_revo(
        os.path.join(proj, revo), out, os.path.splitext(revo)[0], [n["guid"] for n in nos if n["guid"] not in excluir]
    )
    return rel


def ler_fusao(caminho, T):
    """Fusao pronta do scan (ex. `Fused ... .ply` do Revo, no referencial do scan) como superficie da analise."""
    from metricas import ler_nuvem

    m = gpu.xp()
    P, N = ler_nuvem(caminho, normais=True)
    P = gpu.para_gpu(P, m.float64)
    orientado = N is not None and len(N) == len(P)
    N = gpu.para_gpu(N, m.float64) if orientado else gpu.normais_pca(P, k=20)
    bom = m.isfinite(N).all(1)
    R, t = gpu.para_gpu(T[:3, :3].T, m.float64), gpu.para_gpu(T[:3, 3], m.float64)
    return dict(P=P[bom] @ R + t, N=N[bom] @ R, obs=m.ones(int(bom.sum()))), orientado


def fundir_todos(proj, scans, poses, a, fusoes=None):
    sup = {}
    with tempfile.TemporaryDirectory() as tmp:
        for s in scans:
            arq = os.path.join(fusoes, s + ".ply") if fusoes else None
            if arq and os.path.exists(arq):
                x, orientado = ler_fusao(arq, poses[s])
                (a.orientados.add if orientado else a.orientados.discard)(s)
            elif fusoes:
                sys.exit(f"falta a fusao de {s} em {fusoes} (<guid>.ply no referencial do scan)")
            else:
                x = fundir_scan(proj, s, tmp, a.voxel, a.min_obs, poses[s])
                a.orientados.add(s)
            if x is not None and len(x["P"]):
                sup[s] = x
    return sup


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", help="projeto ja limpo (so a peca)")
    ap.add_argument("--poses", required=True, help="JSON {guid: 4x4 scan->referencia}")
    ap.add_argument("--saida", required=True, help="pasta nova")
    ap.add_argument("--scans", nargs="+", help="padrao: todos os que tem pose")
    ap.add_argument("--fusoes", help="pasta com <guid>.ply (fusao do Revo de cada scan, referencial do scan)")
    ap.add_argument(
        "--alcance", type=float, default=0.45, help="mm: com --fusoes, pixel sai se o defeito esta mais perto"
    )
    ap.add_argument("--voxel", type=float, default=0.15, help="mm da fusao propria (ou espacamento das --fusoes)")
    ap.add_argument("--min-obs", type=int, default=4, help="observacoes minimas por voxel da fusao")
    ap.add_argument("--raio", type=float, default=0.4, help="mm da media local de altura")
    ap.add_argument(
        "--alcance-normal",
        type=float,
        default=1.2,
        help="mm ao longo da normal para achar a superficie dos outros scans (so entre fusoes com normal orientada)",
    )
    ap.add_argument("--viz-min", type=int, default=6, help="vizinhos de outro scan para contar como cobertura")
    ap.add_argument("--cos-min", type=float, default=0.6, help="cosseno minimo entre normais (mesma face)")
    ap.add_argument("--tol", type=float, default=0.15, help="mm de desvio do consenso para ser defeito")
    ap.add_argument("--acordo", type=float, default=0.10, help="mm: outros scans concordam entre si ate isso")
    ap.add_argument("--min-scans", type=int, default=2, help="outros scans concordantes exigidos")
    ap.add_argument("--area-min", type=float, default=0.5, help="mm2 minimos de uma mancha de defeito")
    ap.add_argument("--frac-max", type=float, default=0.30, help="acima disso o scan nao e cortado (pose suspeita)")
    ap.add_argument("--rodadas", type=int, default=3)
    ap.add_argument("--descartar", nargs="+", help="scans que saem inteiros (nao entram nem votam)")
    ap.add_argument(
        "--descartar-ruins",
        action="store_true",
        help="descarta sozinho o scan cuja fracao fora do consenso passa de --descarte-min e de --descarte-fator x a mediana",
    )
    ap.add_argument("--descarte-min", type=float, default=6.0, help="%% minimo fora do consenso para descartar")
    ap.add_argument("--descarte-fator", type=float, default=2.5, help="vezes a mediana dos outros scans")
    ap.add_argument(
        "--antes",
        help="projeto de ANTES do consenso_visibilidade (o `projeto` e o de depois): bloco que so um scan tem e que a "
        "visibilidade cortou em >= --ilha-frac sai inteiro",
    )
    ap.add_argument("--ilha-frac", type=float, default=0.3)
    ap.add_argument("--ilha-min", type=int, default=20, help="pontos minimos do bloco")
    ap.add_argument("--ilhas-uniao", action="store_true", help="tira o que, na uniao, fica solto do corpo da peca")
    ap.add_argument(
        "--motivos", help="descartados.json de uma triagem anterior: preserva o motivo medido de cada descarte"
    )
    ap.add_argument(
        "--duvida-atlas", help="pasta do atlas usado na limpeza (com --duvida-rotulos): resolve as regioes em duvida"
    )
    ap.add_argument("--duvida-rotulos", help="rotulos.json aplicado na limpeza")
    ap.add_argument(
        "--duvida-peca", type=float, default=0.5, help="fracao confirmada para a regiao em duvida virar peca"
    )
    ap.add_argument(
        "--duvida-max-pct",
        type=float,
        default=2.0,
        help="%% da superficie do scan que pode continuar em duvida sem virar pendencia",
    )
    ap.add_argument(
        "--vies-celula", type=float, default=0.0, help="mm da celula do vies de pose; 0 = min(30, maior dimensao / 3)"
    )
    ap.add_argument(
        "--vies-max", type=float, default=0.12, help="mm de vies com sinal numa celula para a pose virar pendencia"
    )
    ap.add_argument("--so-triagem", action="store_true", help="so grava <saida>/descartados.json e termina")
    ap.add_argument("--unir", metavar="NOME", help="grava tambem <saida>/unido: um scan com todos os aceitos alinhados")
    a = ap.parse_args(argv)
    a.orientados = set()
    baixa_prioridade()
    m = gpu.xp()
    proj, saida = os.path.abspath(a.projeto), os.path.abspath(a.saida)
    if os.path.exists(saida) and os.listdir(saida):
        sys.exit(f"destino ja existe: {saida}")
    if os.path.commonpath([proj, saida]) == proj:
        sys.exit("o destino nao pode ficar dentro do projeto de origem")
    with open(a.poses, encoding="utf-8") as fh:
        poses = {k: np.asarray(v, float) for k, v in json.load(fh).items()}
    _, _, todos = AP.scans_do_projeto(proj)
    scans = [s for s in todos if s in poses and (not a.scans or s in a.scans)]
    if len(scans) < 3:
        sys.exit("preciso de pelo menos 3 scans com pose para haver maioria")
    t0 = time.time()
    sup = fundir_todos(proj, scans, poses, a, a.fusoes)
    _LOG.info(
        "fusao propria de %d scans: %s pontos, %.0f s",
        len(sup),
        f"{sum(len(x['P']) for x in sup.values()):,}",
        time.time() - t0,
    )
    # scan ruim inteiro (pouca cobertura, rastreio ruim): descartar em vez de remendar; ele tambem nao vota
    descartados = {}
    motivos = {}
    if a.motivos:
        with open(a.motivos, encoding="utf-8") as fh:
            motivos = json.load(fh)
    for s in a.descartar or []:
        if s in sup:
            descartados[s] = motivos.get(s) or dict(motivo="pedido em --descartar")
            sup.pop(s)
    con = Consenso(sup, a)
    while a.descartar_ruins and len(sup) > 3:
        r0 = resumo(con.analisar(), con.vivos, a)
        pct = {s: r0[s]["acima_tol_pct"] or 0.0 for s in sup}
        escolha = escolher_descarte(pct, a.descarte_min, a.descarte_fator)
        if escolha is None:
            break
        pior, resto = escolha
        descartados[pior] = dict(
            motivo="fora do consenso muito acima dos outros scans",
            acima_tol_pct=pct[pior],
            mediana_dos_outros_pct=resto,
        )
        _LOG.info("%s DESCARTADO: %.1f%% fora do consenso (mediana dos outros %.1f%%)", pior, pct[pior], resto)
        con.descartar(pior)
    scans = [s for s in scans if s in sup]
    if a.so_triagem:  # so decide quais scans saem inteiros (para as etapas anteriores do fluxo nao usarem o scan ruim)
        salvar_json_atomico(os.path.join(saida, "descartados.json"), descartados)
        if len(sup) >= 3:
            salvar_json_atomico(
                os.path.join(saida, "triagem.json"),
                dict(descartados=descartados, ruido_entre_scans=ruido_entre_scans(con.analisar(), con.vivos)),
            )
        _LOG.info("triagem: %d scan(s) descartado(s): %s", len(descartados), sorted(descartados))
        return 0
    if len(sup) < 3:
        sys.exit("sobraram menos de 3 scans depois dos descartes")
    vivos = con.vivos  # o Consenso e o dono das mascaras: toda remocao passa por con.remover
    removido = {s: m.zeros(len(x["P"]), bool) for s, x in sup.items()}

    def tirar(s, mascara):
        removido[s] |= mascara & vivos[s]
        return con.remover(s, mascara)

    # ilhas de um scan so: bloco que nenhum outro scan tem e que a limpeza por visibilidade ja cortou em parte e
    # objeto estranho (suporte, massinha); o miolo que ninguem chegou a ver atraves sai junto
    ilhas = {}
    if a.antes:
        supA = fundir_todos(os.path.abspath(a.antes), scans, poses, a)
        for s in sup:
            if s not in supA:
                continue
            PA = supA[s]["P"]
            outros = m.concatenate([sup[t]["P"] for t in sup if t != s])
            orf = gpu.vizinho_mais_proximo(outros, PA, dist_max=a.raio)[1] < 0
            sumiu = ~m.isin(supA[s]["chaves"], sup[s]["chaves"])
            idx = m.flatnonzero(orf)
            if not len(idx):
                continue
            comp, tam = gpu.componentes_conexos_voxel(PA[idx], 2.0 * a.voxel, 26)
            cortado = m.bincount(comp, weights=sumiu[idx].astype(m.float64), minlength=len(tam))
            ruim = (cortado >= a.ilha_frac * tam) & (tam >= a.ilha_min)
            sai = m.isin(sup[s]["chaves"], supA[s]["chaves"][idx[ruim[comp] & ~sumiu[idx]]])
            tirar(s, sai)
            ilhas[s] = dict(blocos=int(ruim.sum()), pontos_removidos=int(sai.sum()), orfaos=int(len(idx)))
            _LOG.info(
                "%s: %d bloco(s) so deste scan, cortados em parte pela visibilidade; saem mais %s pontos",
                s,
                int(ruim.sum()),
                f"{int(sai.sum()):,}",
            )
        del supA
    an = con.analisar()
    antes, an0 = resumo(an, vivos, a), an
    ruido = ruido_entre_scans(an, vivos)
    # pose: vies com sinal por celula (desvio pequeno e sistematico nao vira "defeito", vira borrao na fusao)
    todos_p = m.concatenate([x["P"] for x in sup.values()])
    ext = float(gpu.para_cpu(todos_p.max(0) - todos_p.min(0)).max())
    del todos_p
    cel = a.vies_celula or min(30.0, max(ext / 3.0, 5.0))
    vies = {s: vies_por_celula(sup[s]["P"], an[s]["desvio"], an[s]["consenso"] & vivos[s], cel) for s in sup}
    pose_vies = sorted(s for s, v in vies.items() if v.get("celulas") and v["vies_abs_max_mm"] > a.vies_max)
    for s in pose_vies:
        _LOG.info(
            "%s: POSE COM VIES: celula de %.0f mm deslocada %.3f mm (limite %.3f)",
            s,
            cel,
            vies[s]["vies_abs_max_mm"],
            a.vies_max,
        )
    # regioes que a limpeza deixou em duvida: o consenso decide (confirmada vira peca; o resto segue em duvida)
    duvida, duvida_rel = {}, {}
    if a.duvida_atlas and a.duvida_rotulos:
        with open(a.duvida_rotulos, encoding="utf-8") as fh:
            dec = json.load(fh)
        info_d = {s: duvidas_do_atlas(os.path.abspath(a.duvida_atlas), dec, s, sup[s]["P"], poses[s]) for s in sup}
        for s, d in info_d.items():
            if d is None:
                continue
            reg, deitadas, n0 = d
            outro = [
                n0 is None or info_d[t] is None or info_d[t][2] is None or float(n0 @ info_d[t][2]) < 0.95
                for t in an[s]["de"]
            ]
            duvida[s], duvida_rel[s] = resolver_duvidas(an[s], m.where(vivos[s], reg, 0), deitadas, outro, a)
            duvida_rel[s]["pendente_pct"] = round(
                100 * duvida_rel[s]["pontos_pendentes"] / max(int(vivos[s].sum()), 1), 3
            )
            _LOG.info(
                "%s: duvida em %d regioes (%s pontos): %d viraram peca; seguem em duvida %s pontos (%.2f%%)",
                s,
                duvida_rel[s]["regioes"],
                f"{duvida_rel[s]['pontos']:,}",
                duvida_rel[s]["viraram_peca"],
                f"{duvida_rel[s]['pontos_pendentes']:,}",
                duvida_rel[s]["pendente_pct"],
            )
    suspeitos, hist = set(), []
    for rodada in range(1, a.rodadas + 1):
        if rodada > 1:
            an = con.analisar()
        novos = 0
        # do pior scan para o melhor, UM por vez: o consenso de cada um ja desconta o que os anteriores perderam,
        # entao dois scans nunca sao cortados no mesmo lugar um "contra" o outro (o trecho continua coberto)
        ordem = sorted(an, key=lambda s: -float((an[s]["defeito"] & vivos[s]).sum()) / max(len(vivos[s]), 1))
        for s in ordem:
            x = con.analisar([s])[s]
            cob = int((x["consenso"] & vivos[s]).sum())
            d = x["defeito"] & vivos[s]
            if cob and float(d.sum()) / cob > a.frac_max:
                suspeitos.add(s)
                continue
            novos += tirar(s, d)
        hist.append(dict(rodada=rodada, pontos_removidos=novos))
        _LOG.info("rodada %d: %s pontos de superficie marcados como defeito", rodada, f"{novos:,}")
        if novos < 50:
            break
    soltos = {}
    if a.ilhas_uniao:  # na UNIAO, o que nao esta ligado ao corpo da peca (maior componente) sai do scan que o trouxe
        ids = list(sup)
        X = m.concatenate([sup[s]["P"][vivos[s]] for s in ids])
        comp, _ = gpu.componentes_conexos_voxel(X, 2.0 * a.voxel, 26)
        ini = 0
        for s in ids:
            k = int(vivos[s].sum())
            fora_s = comp[ini : ini + k] > 0
            ini += k
            mask = m.zeros(len(vivos[s]), bool)
            mask[m.flatnonzero(vivos[s])[fora_s]] = True
            tirar(s, mask)
            soltos[s] = int(fora_s.sum())
        _LOG.info("ilhas soltas na uniao: %s pontos removidos", f"{sum(soltos.values()):,}")
    # cobertura do que saiu: quantos OUTROS scans ainda tem superficie no lugar de cada ponto removido
    cobertura = {}
    for s in sup:
        if not bool(removido[s].any()):
            continue
        n = m.zeros(int(removido[s].sum()), m.int32)
        for t in sup:
            if t != s:
                _, cnt, cs = con.par(s, t)
                n += ((cnt[removido[s]] >= a.viz_min) & (cs[removido[s]] > a.cos_min)).astype(m.int32)
        cobertura[s] = dict(
            sem_nenhum_outro_pct=round(100 * float((n == 0).mean()), 3),
            com_1_outro_pct=round(100 * float((n == 1).mean()), 3),
            com_2_ou_mais_pct=round(100 * float((n >= 2).mean()), 3),
        )
    duvida_pend = sorted(s for s, r in duvida_rel.items() if r["pendente_pct"] > a.duvida_max_pct)
    imagens = {}
    for s in sup:
        imagens[s] = desenhar(
            s, sup[s], vivos[s], an0[s], removido[s], os.path.join(saida, "imagens", f"{s}_antes.png"), a, duvida.get(s)
        )
    rel_aplic = aplicar(proj, os.path.join(saida, "projeto"), poses, sup, removido, a, set(descartados))
    unido = None
    if a.unir:  # um scan so com os quadros de todos os scans aceitos, ja no referencial comum: e o que se funde no Revo
        import uniao_scans as U

        base = min(sup, key=lambda s: float(np.abs(poses[s] - np.eye(4)).sum()))
        n = U.unir_varios(
            os.path.join(saida, "projeto"),
            {s: poses[s].tolist() for s in sup},
            base,
            os.path.join(saida, "unido"),
            a.unir,
        )
        unido = dict(
            pasta=os.path.join(saida, "unido"), nome=a.unir, base=base, quadros=n["total"], por_scan=n["por_scan"]
        )
        _LOG.info("scan unido: %s (%d quadros de %d scans)", unido["pasta"], n["total"], len(sup))
    if a.fusoes:  # previsao: as mesmas fusoes sem os pontos removidos; a medida real exige refundir no Revo
        sup2 = {s: {k: v[vivos[s]] for k, v in x.items()} for s, x in sup.items()}
    else:  # verificacao: fusao propria refeita a partir dos quadros GRAVADOS
        sup2 = fundir_todos(os.path.join(saida, "projeto"), scans, poses, a)
    vivos2 = {s: m.ones(len(x["P"]), bool) for s, x in sup2.items()}
    an2 = Consenso(sup2, a, vivos2).analisar()
    depois = resumo(an2, vivos2, a)
    for s in sup2:
        imagens[s] += desenhar(
            s, sup2[s], vivos2[s], an2[s], None, os.path.join(saida, "imagens", f"{s}_depois.png"), a
        )
    por_scan = {}
    for s in sup:
        por_scan[s] = dict(
            antes=antes[s],
            depois=depois.get(s),
            superficie_removida_pct=round(100 * float(removido[s].mean()), 3),
            pose_suspeita=s in suspeitos,
            vies_de_pose=vies.get(s),
            duvida=duvida_rel.get(s),
            cobertura_do_removido=cobertura.get(s),
            quadros=rel_aplic.get(s),
            imagens=imagens[s],
        )
        _LOG.info(
            "%s: acima de %.2f mm %s%% -> %s%% | p90 %s -> %s mm | superficie removida %.2f%%%s",
            s,
            a.tol,
            antes[s]["acima_tol_pct"],
            depois[s]["acima_tol_pct"] if s in depois else "?",
            antes[s]["desvio_p90_mm"],
            depois[s]["desvio_p90_mm"] if s in depois else "?",
            por_scan[s]["superficie_removida_pct"],
            " | POSE SUSPEITA: nao cortado" if s in suspeitos else "",
        )
    pend = bool(suspeitos or pose_vies or duvida_pend)
    salvar_json_atomico(
        os.path.join(saida, "relatorio.json"),
        dict(
            origem=proj,
            parametros={k: (sorted(v) if isinstance(v, set) else v) for k, v in vars(a).items()},
            depois_e="previsao (refundir no Revo para medir)"
            if a.fusoes
            else "fusao propria refeita dos quadros gravados",
            rodadas=hist,
            scans=por_scan,
            descartados=descartados,
            ilhas_de_um_scan=ilhas,
            soltos_na_uniao=soltos,
            unido=unido,
            ruido_entre_scans=ruido,
            pendencias=dict(
                pose_suspeita=sorted(suspeitos), pose_com_vies=pose_vies, duvida_acima_do_limite=duvida_pend
            ),
            parcial=pend,
            tempo_s=round(time.time() - t0, 1),
        ),
    )
    _LOG.info("gravado: %s (%.0f s)%s", saida, time.time() - t0, " | PARCIAL" if pend else "")
    return 2 if pend else 0


if __name__ == "__main__":
    sys.exit(main())
