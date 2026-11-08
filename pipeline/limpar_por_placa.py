"""Limpa os quadros de um projeto .revo quando a peca esta APOIADA/SUSPENSA sobre uma placa-gabarito.

    python pipeline/limpar_por_placa.py <pasta_do_projeto> --saida <pasta_nova> [--scans id ...]

Por scan: ajusta o plano da placa (RANSAC na amostra dos quadros), mantem so o que esta a mais de
--altura-min mm acima dela e fica com o(s) maior(es) componente(s) conectado(s) (grade de --voxel mm).
Parafusos, suportes baixos, a propria placa, nuvens soltas e estruturas pequenas saem. Em seguida
apaga os pixels de cada quadro (como limpar_raw) e remove a profundidade dos quadros que ficaram
sem peca (senao a placa voltaria na fusao do Revo). O original nunca e tocado e o destino nao pode existir.
Nao usa os .ply antigos (fuse/mesh/mark) porque trazem a placa.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
import gpu
import revo_projeto as RP
from apoio_validado import carregar_planos, validar_perda
from recursos import baixa_prioridade
from revo_projeto import RevoScan, gravar_revo, plano_ransac


def _mascara(P, seg):
    """Mascara da placa (GPU): P numpy ou cupy, saida do mesmo tipo. A grade `manter` fica em cupy no dicionario."""
    m = gpu.xp()
    na_gpu = RP._eh_cupy(P)
    P = gpu.para_gpu(P, m.float64)
    c = seg["placa"]
    h = P @ gpu.para_gpu(c["n"]) + c["d"]
    idx = m.floor((P - gpu.para_gpu(c["lo"])) / c["voxel"]).astype(m.int64)
    manter = gpu.para_gpu(c["manter"])
    forma = gpu.para_gpu(manter.shape)
    dentro = ((idx >= 0) & (idx < forma)).all(axis=1)
    ic = m.clip(idx, 0, forma - 1)
    ok = manter[ic[:, 0], ic[:, 1], ic[:, 2]] & dentro & (h > c["altura"])
    return ok if na_gpu else gpu.para_cpu(ok)


def amostra_mundo(scan, passo_quadro, passo_pixel):
    """Amostra dos quadros no referencial do mundo, float32, em cupy (os pontos nascem na GPU)."""
    m = gpu.xp()
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    return m.vstack([scan.pontos(k, passo=passo_pixel, na_gpu=True)[0].astype(m.float32) for k in ks[::passo_quadro]])


def _area_plano(P, n, d, thr=2.0, celula=2.0):
    """Area (mm2) ocupada pelos pontos a menos de thr do plano, em celulas de celula mm (GPU)."""
    m = gpu.xp()
    P = gpu.para_gpu(P)
    Q = P[m.abs(P @ gpu.para_gpu(n) + d) < thr]
    if len(Q) == 0:
        return 0.0
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c = m.floor(m.stack([Q @ gpu.para_gpu(u), Q @ gpu.para_gpu(v)], axis=1) / celula).astype(m.int64)
    c = c - c.min(0)
    return float(len(gpu.agrupar(c[:, 0] * (int(c[:, 1].max()) + 1) + c[:, 1])[0])) * celula * celula


def escolher_plano(S, a, log):
    """'dominante': plano com mais pontos. 'extenso': entre os 4 maiores planos (RANSAC sequencial),
    o de maior AREA ocupada: a placa e larga e esparsa; a face plana de uma peca pequena tem muitos
    pontos mas pouca area e engana o criterio dominante. Tudo na GPU (S cupy ou numpy)."""
    m = gpu.xp()
    S = gpu.para_gpu(S)
    if a.plano == "dominante":
        return plano_ransac(S, thr=2.0, it=2000)
    resto, cands = S, []
    for i in range(4):
        if len(resto) < 1000:
            break
        n, d, _ = plano_ransac(resto, thr=2.0, it=2000, seed=i)
        n = n / np.linalg.norm(n)
        frac = float((m.abs(S @ gpu.para_gpu(n) + d) < 2.0).mean())
        cands.append((_area_plano(S, n, d), n, d, frac))
        resto = resto[m.abs(resto @ gpu.para_gpu(n) + d) >= 2.0]
    log("   planos candidatos (area mm2 / % pontos): " + ", ".join(f"{c[0]:.0f}/{c[3] * 100:.0f}%" for c in cands))
    _, n, d, frac = max(cands, key=lambda c: c[0])
    # orienta a normal para o lado onde estao os objetos (como plano_ransac)
    h = S @ gpu.para_gpu(n) + d
    if float((h > 6.0).mean()) < float((h < -6.0).mean()):
        n, d = -n, -d
    return n, d, frac


def construir_regiao(A, a, log, plano_apoio=None):
    """Grade da peca acima da placa (GPU: bincount, rotulagem e morfologia em cupy/cupyx)."""
    m = gpu.xp()
    nd = RP._nd()
    A = gpu.para_gpu(A)
    if plano_apoio is None:
        n, d, frac = escolher_plano(A[:: max(1, len(A) // 150000)].astype(m.float64), a, log)
    else:
        n, d = np.asarray(plano_apoio["n"]), float(plano_apoio["d"])
        frac = float((m.abs(A @ gpu.para_gpu(n) + d) < 2.0).mean())
        log("   usando plano de apoio medido antes da limpeza (sem reajustar na face da peca)")
    n = n / np.linalg.norm(n)
    h = A @ gpu.para_gpu(n) + d
    log(
        f"   placa: {frac * 100:.0f}% da amostra no plano (normal {np.round(n, 3).tolist()}); "
        f"{float((h > a.altura_min).mean()) * 100:.1f}% dos pontos acima de {a.altura_min:g} mm"
    )
    if plano_apoio is None and frac < 0.15:
        raise RuntimeError("nao achei uma placa dominante (menos de 15% dos pontos no plano)")
    if plano_apoio is not None and float((h < -0.5).mean()) > 0.05:
        raise ValueError("plano de apoio atravessa mais de 5% da selecao; confira IDs, pose e orientacao")
    acima = A[h > a.altura_min].astype(m.float64)
    validar_perda(len(A), len(acima), getattr(a, "perda_max", 0.3))
    v = a.voxel
    g = m.floor(acima / v).astype(m.int64)
    lo = g.min(0) - 3
    g -= lo
    dims = [int(x) for x in gpu.para_cpu(g.max(0) + 4)]
    cont = m.bincount((g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2], minlength=dims[0] * dims[1] * dims[2]).reshape(
        dims
    )
    vol = cont > 0
    denso = cont >= a.min_pts_voxel  # pontes de ruido (poucos pontos por voxel) nao ligam componentes
    lab, nl = nd.label(denso, structure=RP._uns(3, 3, 3))
    nl = int(nl)
    tam = gpu.para_cpu(m.bincount(lab[denso], minlength=nl + 1))
    tam[0] = 0
    ordem = np.argsort(-tam)[: a.componentes]
    log(
        "   componentes acima da placa (voxels densos): "
        + ", ".join(f"{int(tam[i])}{'*' if i in ordem else ''}" for i in np.argsort(-tam)[:6])
        + f"  (* = mantidos; voxel {v:g} mm, >= {a.min_pts_voxel} pontos)"
    )
    nucleo = m.isin(lab, gpu.para_gpu(ordem))
    # voxels esparsos encostados no nucleo sao borda da propria peca; esparsos soltos ou densos de outro componente nao entram
    manter = nucleo | (RP._morf("binary_dilation", nucleo, structure=RP._uns(3, 3, 3)) & vol & ~denso)
    if a.margem_vox > 0:  # (iterations=0 no scipy repetiria ate encher o volume)
        manter = RP._morf("binary_dilation", manter, structure=RP._uns(3, 3, 3), iterations=a.margem_vox)
    return {"n": n, "d": float(d), "lo": gpu.para_cpu(lo) * v, "voxel": v, "manter": manter, "altura": a.altura_min}


def diagnostico(A, seg, caminho):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = gpu.xp()
    c = seg["placa"]
    Ag = gpu.para_gpu(A, m.float64)
    ok = gpu.para_cpu(_mascara(Ag, seg))
    n = c["n"]
    u = np.cross(n, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    P = Ag
    h = gpu.para_cpu(P @ gpu.para_gpu(n) + c["d"])  # projecoes na GPU; o grafico (matplotlib) e CPU
    pu, pv = gpu.para_cpu(P @ gpu.para_gpu(u)), gpu.para_cpu(P @ gpu.para_gpu(v))
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    for a_, (x, y, t) in zip(ax, [(pu, pv, "topo"), (pu, h, "lateral"), (pv, h, "frontal")]):
        a_.scatter(x[~ok], y[~ok], s=0.15, c="#c33", label="removido")
        a_.scatter(x[ok], y[ok], s=0.15, c="#222", label="mantido")
        a_.set_aspect("equal")
        a_.set_title(t)
    ax[0].legend(markerscale=15)
    plt.tight_layout()
    plt.savefig(caminho, dpi=60)
    plt.close()


def processar(proj, s, saida, a, diag_dir):
    t0 = time.time()
    log = [f"== {s}"]
    pasta = os.path.join(proj, "data", s)
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(pasta, tmp)
        try:
            for m in sorted(set(int(m) for m, p in zip(scan.modo, scan.tem_prof) if p)):
                K, e = scan.calibrar(m)
                log.append(f"   calibracao modo {m}: foco {K[0]:.0f}/{K[1]:.0f} | erro {e:.3f} mm")
            A = amostra_mundo(scan, a.amostra_quadros, a.amostra_passo)
            log.append(f"   amostra: {len(A):,} pontos de {int(scan.tem_prof.sum())} quadros com profundidade")
            plano = getattr(a, "_planos_apoio", {}).get(s)
            if getattr(a, "planos_apoio", None) and plano is None:
                raise ValueError(f"scan {s} sem plano de apoio no JSON")
            seg = {"placa": construir_regiao(A, a, lambda m: log.append(m), plano_apoio=plano)}
            amostra_mantida = int(_mascara(A, seg).sum())
            validar_perda(len(A), amostra_mantida, a.perda_max)
            if diag_dir:
                diagnostico(
                    gpu.para_cpu(A[:: max(1, len(A) // 400000)]), seg, os.path.join(diag_dir, f"{s}_antes_depois.png")
                )
            RP.mascara_manter = _mascara  # limpar_scan e o filtro do raw_preview usam esta funcao
            r = RP.limpar_scan(
                scan,
                seg,
                os.path.join(saida, "data", s),
                log=lambda *x: log.append(" ".join(map(str, x))),
                pixels=not a.sem_pixels,
                ilhas=False,
                min_mantido=0.0,
                sem_ply_antigos=True,
                zerar_vazios=True,
                min_pix_quadro=a.min_pix_quadro,
            )
            validar_perda(r["antes"], r["depois"], a.perda_max)
            r["plano_apoio"] = {
                "n": np.asarray(seg["placa"]["n"]).tolist(),
                "d": seg["placa"]["d"],
                "fornecido": plano is not None,
            }
        finally:
            scan.c.close()
    log.append(f"   concluido em {time.time() - t0:.0f}s")
    return r, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True, help="pasta nova do projeto limpo (nao pode ter conteudo)")
    ap.add_argument("--scans", nargs="+", help="guids (padrao: todos do .revo)")
    ap.add_argument(
        "--altura-min",
        type=float,
        default=16.0,
        help="mm acima do plano da placa; abaixo disso e placa, parafuso ou suporte baixo",
    )
    ap.add_argument("--voxel", type=float, default=6.0, help="mm da grade usada para achar os componentes conectados")
    ap.add_argument("--componentes", type=int, default=1, help="quantos componentes (do maior para o menor) sao a peca")
    ap.add_argument(
        "--min-pts-voxel",
        type=int,
        default=3,
        help="pontos minimos da amostra para um voxel contar na ligacao dos componentes",
    )
    ap.add_argument(
        "--margem-vox", type=int, default=0, help="voxels de folga ao redor da peca (margem = voxel x este valor)"
    )
    ap.add_argument(
        "--min-pix-quadro", type=int, default=200, help="quadro com menos pixels de peca perde a profundidade"
    )
    ap.add_argument("--amostra-quadros", type=int, default=2, help="usa 1 a cada N quadros para achar a peca")
    ap.add_argument("--amostra-passo", type=int, default=8, help="usa 1 a cada N pixels por quadro para achar a peca")
    ap.add_argument("--sem-pixels", action="store_true", help="nao filtra pixels voadores nem manchas 2D")
    ap.add_argument(
        "--plano",
        choices=["dominante", "extenso"],
        default="dominante",
        help="como achar a placa: plano com mais pontos, ou o de maior area entre 4 candidatos "
        "(use 'extenso' quando a peca tem face plana grande e o plano dominante cai nela)",
    )
    ap.add_argument("--diagnostico", help="pasta onde gravar a imagem antes/depois (fora da entrega)")
    ap.add_argument(
        "--planos-apoio", help="JSON de planos medidos no original, antes da limpeza; aceita IDs remapeados"
    )
    ap.add_argument(
        "--perda-max",
        type=float,
        default=0.3,
        help="fracao maxima de pontos removidos por scan (padrao 0,30); exceder gera parcial e copia intacta",
    )
    a = ap.parse_args()
    if not 0 < a.perda_max < 1:
        ap.error("--perda-max deve estar entre 0 e 1")
    baixa_prioridade()
    proj = os.path.abspath(a.projeto)
    a._planos_apoio = carregar_planos(a.planos_apoio, proj) if a.planos_apoio else {}
    revo = next(f for f in os.listdir(proj) if f.endswith(".revo"))
    nome = os.path.splitext(revo)[0] + "_limpo"
    saida = os.path.abspath(a.saida)
    if os.path.exists(saida) and (not os.path.isdir(saida) or os.listdir(saida)):
        sys.exit(f"destino ja existe: {saida}")
    nos = json.load(open(os.path.join(proj, revo), encoding="utf-8"))["nodes"]
    todos = [n["guid"] for n in nos]
    scans = a.scans or [n["guid"] for n in nos if n.get("type") != 5]  # tipo 5 = pasta de marcadores, nao e scan
    os.makedirs(os.path.join(saida, "data"), exist_ok=True)
    if a.diagnostico:
        os.makedirs(a.diagnostico, exist_ok=True)
    rel = {}
    for s in scans:
        try:
            r, log = processar(proj, s, saida, a, a.diagnostico)
            rel[s] = r
        except Exception as e:  # scan que nao deu para limpar: copia intacto e avisa
            log = [f"== {s}", f"   ERRO: {type(e).__name__}: {e}; copiado sem alteracao"]
            shutil.rmtree(os.path.join(saida, "data", s), ignore_errors=True)
            shutil.copytree(os.path.join(proj, "data", s), os.path.join(saida, "data", s))
            rel[s] = {"erro": str(e)}
        print("\n".join(log), flush=True)
    for s in todos:
        if s not in scans:
            shutil.copytree(os.path.join(proj, "data", s), os.path.join(saida, "data", s))
    gravar_revo(os.path.join(proj, revo), saida, nome, todos)
    with open(os.path.join(saida, "relatorio_limpeza_placa.json"), "w", encoding="utf-8") as arquivo:
        json.dump(rel, arquivo, indent=2)
    if a.diagnostico:
        with open(os.path.join(a.diagnostico, "relatorio_limpeza.json"), "w", encoding="utf-8") as arquivo:
            json.dump(rel, arquivo, indent=2)
    print("saida:", saida)
    print("projeto .revo:", os.path.join(saida, nome + ".revo"))
    return 2 if any("erro" in r for r in rel.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
