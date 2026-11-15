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
Limite: a fusao propria por centroides nao e a do Revo (no P9 so 60% dos pontos ruins da fusao do Revo tinham
superficie propria por perto); ela serve para LOCALIZAR defeito sem abrir o Revo. Preferir --fusoes quando houver. Codigo 2 = ha pendencias (pose suspeita ou divergencia sem maioria).
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


def altura_local(alvo, P, N, raio, hmax, assinado, k=64):
    """Altura media (ao longo de N) da superficie de `alvo` no cilindro de raio `raio` e meia-altura `hmax` em torno
    de cada P; n de vizinhos e cosseno medio. A busca ao longo da normal acha a superficie do outro scan mesmo
    quando ela esta deslocada mais que `raio` (defeito grande). `assinado`: as normais dos dois lados tem sentido
    confiavel, entao so vale vizinho de MESMA face (parede fina nao se confunde com o verso). Tudo cupy."""
    m = gpu.xp()
    off = m.zeros(len(P))
    cnt = m.zeros(len(P), m.int32)
    cs = m.zeros(len(P))
    if not len(alvo["P"]) or not len(P):
        return off, cnt, cs
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


def analisar(sup, vivos, a, alvos=None):
    """Por scan (`alvos`, padrao todos): desvio em relacao ao consenso dos outros, usando so os pontos vivos.
    Devolve {s: dict(desvio, n_cobrem, consenso, defeito)}."""
    m = gpu.xp()
    ids = list(sup)
    atual = {s: {k: v[vivos[s]] for k, v in sup[s].items()} for s in ids}
    out = {}
    for s in alvos or ids:
        P, N = sup[s]["P"], sup[s]["N"]
        proprio, _, _ = altura_local(atual[s], P, N, a.raio, a.raio, s in a.orientados)
        devs, cobre = [], []
        for t in ids:
            if t == s:
                continue
            ass = s in a.orientados and t in a.orientados
            off, cnt, cs = altura_local(atual[t], P, N, a.raio, a.alcance_normal if ass else a.raio, ass)
            devs.append(off - proprio)
            cobre.append((cnt >= a.viz_min) & (cs > a.cos_min))
        D, K = m.stack(devs, 1), m.stack(cobre, 1)
        n_cob = K.sum(1)
        med = m.nanmedian(m.where(K, D, m.nan), axis=1)
        med = m.where(n_cob > 0, med, 0.0)
        concordam = (K & (m.abs(D - med[:, None]) < a.acordo)).sum(1)
        consenso = concordam >= a.min_scans
        bruto = consenso & (m.abs(med) > a.tol) & vivos[s]
        # mancha: maioria dos vizinhos (2 voxels) tambem em defeito; depois area minima por componente
        defeito = m.zeros(len(P), bool)
        if bool(bruto.any()):
            r = 2.0 * a.voxel
            tot = gpu.contar_no_raio(P[vivos[s]], P, r)
            ruim = gpu.contar_no_raio(P[bruto], P, r)
            cand = vivos[s] & consenso & (m.abs(med) > 0.5 * a.tol) & (ruim >= 0.5 * m.maximum(tot, 1))
            if bool(cand.any()):
                ic = m.flatnonzero(cand)
                comp, tam = gpu.componentes_conexos_voxel(P[ic], 2.0 * a.voxel, 26)
                defeito[ic[tam[comp] * a.voxel * a.voxel >= a.area_min]] = True
        out[s] = dict(desvio=med, n_cobrem=n_cob, consenso=consenso, defeito=defeito)
    return out


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


def desenhar(s, sup, vivos, an, removido, caminho, a):
    m = gpu.xp()
    P, N = sup["P"], sup["N"]
    ad = m.abs(an["desvio"])
    cor = m.full((len(P), 3), 0.80)
    cor = m.where((an["n_cobrem"] == 0)[:, None], gpu.para_gpu(np.array([0.45, 0.62, 0.95])), cor)
    cor = m.where((an["consenso"] & (ad > 0.5 * a.tol))[:, None], gpu.para_gpu(np.array([0.98, 0.78, 0.10])), cor)
    cor = m.where((an["consenso"] & (ad > a.tol))[:, None], gpu.para_gpu(np.array([0.98, 0.50, 0.10])), cor)
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
    for s in a.descartar or []:
        if s in sup:
            descartados[s] = dict(motivo="pedido em --descartar")
            sup.pop(s)
    while a.descartar_ruins and len(sup) > 3:
        v0 = {s: m.ones(len(x["P"]), bool) for s, x in sup.items()}
        r0 = resumo(analisar(sup, v0, a), v0, a)
        pct = {s: r0[s]["acima_tol_pct"] or 0.0 for s in sup}
        pior = max(pct, key=pct.get)
        resto = float(np.median([v for s, v in pct.items() if s != pior]))
        if pct[pior] < a.descarte_min or pct[pior] < a.descarte_fator * max(resto, 1e-6):
            break
        descartados[pior] = dict(
            motivo="fora do consenso muito acima dos outros scans",
            acima_tol_pct=pct[pior],
            mediana_dos_outros_pct=resto,
        )
        _LOG.info("%s DESCARTADO: %.1f%% fora do consenso (mediana dos outros %.1f%%)", pior, pct[pior], resto)
        sup.pop(pior)
    scans = [s for s in scans if s in sup]
    if a.so_triagem:  # so decide quais scans saem inteiros (para as etapas anteriores do fluxo nao usarem o scan ruim)
        salvar_json_atomico(os.path.join(saida, "descartados.json"), descartados)
        _LOG.info("triagem: %d scan(s) descartado(s): %s", len(descartados), sorted(descartados))
        return 0
    if len(sup) < 3:
        sys.exit("sobraram menos de 3 scans depois dos descartes")
    vivos = {s: m.ones(len(x["P"]), bool) for s, x in sup.items()}
    removido = {s: m.zeros(len(x["P"]), bool) for s, x in sup.items()}
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
            removido[s] |= sai
            vivos[s] &= ~sai
            ilhas[s] = dict(blocos=int(ruim.sum()), pontos_removidos=int(sai.sum()), orfaos=int(len(idx)))
            _LOG.info(
                "%s: %d bloco(s) so deste scan, cortados em parte pela visibilidade; saem mais %s pontos",
                s,
                int(ruim.sum()),
                f"{int(sai.sum()):,}",
            )
    suspeitos, antes, an0, hist = set(), None, None, []
    for rodada in range(1, a.rodadas + 1):
        an = analisar(sup, vivos, a)
        if antes is None:
            antes, an0 = resumo(an, vivos, a), an
        novos = 0
        # do pior scan para o melhor, UM por vez: o consenso de cada um ja desconta o que os anteriores perderam,
        # entao dois scans nunca sao cortados no mesmo lugar um "contra" o outro (o trecho continua coberto)
        ordem = sorted(an, key=lambda s: -float((an[s]["defeito"] & vivos[s]).sum()) / max(len(vivos[s]), 1))
        for s in ordem:
            x = analisar(sup, vivos, a, [s])[s]
            cob = int((x["consenso"] & vivos[s]).sum())
            d = x["defeito"] & vivos[s]
            if cob and float(d.sum()) / cob > a.frac_max:
                suspeitos.add(s)
                continue
            removido[s] |= d
            vivos[s] &= ~d
            novos += int(d.sum())
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
            iv = m.flatnonzero(vivos[s])[fora_s]
            removido[s][iv] = True
            vivos[s][iv] = False
            soltos[s] = int(fora_s.sum())
        _LOG.info("ilhas soltas na uniao: %s pontos removidos", f"{sum(soltos.values()):,}")
    # cobertura do que saiu: quantos OUTROS scans ainda tem superficie no lugar de cada ponto removido
    cobertura = {}
    atual = {s: {k: v[vivos[s]] for k, v in sup[s].items()} for s in sup}
    for s in sup:
        if not bool(removido[s].any()):
            continue
        Pr, Nr = sup[s]["P"][removido[s]], sup[s]["N"][removido[s]]
        n = m.zeros(len(Pr), m.int32)
        for t in sup:
            if t != s:
                ass = s in a.orientados and t in a.orientados
                _, cnt, cs = altura_local(atual[t], Pr, Nr, a.raio, a.alcance_normal if ass else a.raio, ass)
                n += ((cnt >= a.viz_min) & (cs > a.cos_min)).astype(m.int32)
        cobertura[s] = dict(
            sem_nenhum_outro_pct=round(100 * float((n == 0).mean()), 3),
            com_1_outro_pct=round(100 * float((n == 1).mean()), 3),
            com_2_ou_mais_pct=round(100 * float((n >= 2).mean()), 3),
        )
    imagens = {}
    for s in sup:
        imagens[s] = desenhar(
            s, sup[s], vivos[s], an0[s], removido[s], os.path.join(saida, "imagens", f"{s}_antes.png"), a
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
    an2 = analisar(sup2, vivos2, a)
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
    pend = bool(suspeitos)
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
            pendencias=dict(pose_suspeita=sorted(suspeitos)),
            parcial=pend,
            tempo_s=round(time.time() - t0, 1),
        ),
    )
    _LOG.info("gravado: %s (%.0f s)%s", saida, time.time() - t0, " | PARCIAL" if pend else "")
    return 2 if pend else 0


if __name__ == "__main__":
    sys.exit(main())
