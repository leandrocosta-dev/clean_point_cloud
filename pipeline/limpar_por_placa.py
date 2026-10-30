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
from scipy import ndimage

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "revo_frames"))
import revo_projeto as RP
from revo_projeto import RevoScan, gravar_revo, plano_ransac


def baixa_prioridade():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _mascara(P, seg):
    c = seg["placa"]
    h = P @ c["n"] + c["d"]
    idx = np.floor((P - c["lo"]) / c["voxel"]).astype(np.int64)
    dentro = np.all((idx >= 0) & (idx < np.array(c["manter"].shape)), axis=1)
    ok = np.zeros(len(P), bool)
    ok[dentro] = c["manter"][tuple(idx[dentro].T)]
    return ok & (h > c["altura"])


def amostra_mundo(scan, passo_quadro, passo_pixel):
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    return np.vstack([scan.pontos(k, passo=passo_pixel)[0].astype(np.float32) for k in ks[::passo_quadro]])


def construir_regiao(A, a, log):
    n, d, frac = plano_ransac(A[:: max(1, len(A) // 150000)].astype(float), thr=2.0, it=2000)
    n = n / np.linalg.norm(n)
    h = A @ n + d
    log(f"   placa: {frac * 100:.0f}% da amostra no plano (normal {np.round(n, 3).tolist()}); "
        f"{(h > a.altura_min).mean() * 100:.1f}% dos pontos acima de {a.altura_min:g} mm")
    if frac < 0.15:
        raise RuntimeError("nao achei uma placa dominante (menos de 15% dos pontos no plano)")
    acima = A[h > a.altura_min].astype(float)
    v = a.voxel
    g = np.floor(acima / v).astype(np.int64)
    lo = g.min(0) - 3
    g -= lo
    dims = g.max(0) + 4
    cont = np.bincount(np.ravel_multi_index(g.T, dims), minlength=int(np.prod(dims))).reshape(dims)
    vol = cont > 0
    denso = cont >= a.min_pts_voxel                 # pontes de ruido (poucos pontos por voxel) nao ligam componentes
    lab, nl = ndimage.label(denso, structure=np.ones((3, 3, 3)))
    tam = np.bincount(lab[denso], minlength=nl + 1)
    tam[0] = 0
    ordem = np.argsort(-tam)[:a.componentes]
    log("   componentes acima da placa (voxels densos): " + ", ".join(f"{int(tam[i])}{'*' if i in ordem else ''}" for i in np.argsort(-tam)[:6]) +
        f"  (* = mantidos; voxel {v:g} mm, >= {a.min_pts_voxel} pontos)")
    nucleo = np.isin(lab, ordem)
    # voxels esparsos encostados no nucleo sao borda da propria peca; esparsos soltos ou densos de outro componente nao entram
    manter = nucleo | (ndimage.binary_dilation(nucleo, structure=np.ones((3, 3, 3))) & vol & ~denso)
    if a.margem_vox > 0:      # (iterations=0 no scipy repetiria ate encher o volume)
        manter = ndimage.binary_dilation(manter, structure=np.ones((3, 3, 3)), iterations=a.margem_vox)
    return {"n": n, "d": float(d), "lo": lo * v, "voxel": v, "manter": manter, "altura": a.altura_min}


def diagnostico(A, seg, caminho):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = seg["placa"]
    ok = _mascara(A.astype(float), seg)
    n = c["n"]
    u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    P = A.astype(float)
    h = P @ n + c["d"]
    fig, ax = plt.subplots(1, 3, figsize=(20, 7))
    for a_, (x, y, t) in zip(ax, [(P @ u, P @ v, "topo"), (P @ u, h, "lateral"), (P @ v, h, "frontal")]):
        a_.scatter(x[~ok], y[~ok], s=0.15, c="#c33", label="removido")
        a_.scatter(x[ok], y[ok], s=0.15, c="#222", label="mantido")
        a_.set_aspect("equal"); a_.set_title(t)
    ax[0].legend(markerscale=15)
    plt.tight_layout(); plt.savefig(caminho, dpi=60); plt.close()


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
            seg = {"placa": construir_regiao(A, a, lambda m: log.append(m))}
            if diag_dir:
                diagnostico(A[:: max(1, len(A) // 400000)], seg, os.path.join(diag_dir, f"{s}_antes_depois.png"))
            RP.mascara_manter = _mascara          # limpar_scan e o filtro do raw_preview usam esta funcao
            r = RP.limpar_scan(scan, seg, os.path.join(saida, "data", s), log=lambda *x: log.append(" ".join(map(str, x))),
                               pixels=not a.sem_pixels, ilhas=False, min_mantido=0.0, sem_ply_antigos=True,
                               zerar_vazios=True, min_pix_quadro=a.min_pix_quadro)
        finally:
            scan.c.close()
    log.append(f"   concluido em {time.time() - t0:.0f}s")
    return r, log


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True, help="pasta nova do projeto limpo (nao pode ter conteudo)")
    ap.add_argument("--scans", nargs="+", help="guids (padrao: todos do .revo)")
    ap.add_argument("--altura-min", type=float, default=16.0, help="mm acima do plano da placa; abaixo disso e placa, parafuso ou suporte baixo")
    ap.add_argument("--voxel", type=float, default=6.0, help="mm da grade usada para achar os componentes conectados")
    ap.add_argument("--componentes", type=int, default=1, help="quantos componentes (do maior para o menor) sao a peca")
    ap.add_argument("--min-pts-voxel", type=int, default=3, help="pontos minimos da amostra para um voxel contar na ligacao dos componentes")
    ap.add_argument("--margem-vox", type=int, default=0, help="voxels de folga ao redor da peca (margem = voxel x este valor)")
    ap.add_argument("--min-pix-quadro", type=int, default=200, help="quadro com menos pixels de peca perde a profundidade")
    ap.add_argument("--amostra-quadros", type=int, default=2, help="usa 1 a cada N quadros para achar a peca")
    ap.add_argument("--amostra-passo", type=int, default=8, help="usa 1 a cada N pixels por quadro para achar a peca")
    ap.add_argument("--sem-pixels", action="store_true", help="nao filtra pixels voadores nem manchas 2D")
    ap.add_argument("--diagnostico", help="pasta onde gravar a imagem antes/depois (fora da entrega)")
    a = ap.parse_args()
    baixa_prioridade()
    proj = os.path.abspath(a.projeto)
    revo = next(f for f in os.listdir(proj) if f.endswith(".revo"))
    nome = os.path.splitext(revo)[0] + "_limpo"
    saida = os.path.abspath(a.saida)
    if os.path.exists(saida) and (not os.path.isdir(saida) or os.listdir(saida)):
        sys.exit(f"destino ja existe: {saida}")
    nos = json.load(open(os.path.join(proj, revo), encoding="utf-8"))["nodes"]
    todos = [n["guid"] for n in nos]
    scans = a.scans or [n["guid"] for n in nos if n.get("type") != 5]      # tipo 5 = pasta de marcadores, nao e scan
    os.makedirs(os.path.join(saida, "data"), exist_ok=True)
    if a.diagnostico:
        os.makedirs(a.diagnostico, exist_ok=True)
    rel = {}
    for s in scans:
        try:
            r, log = processar(proj, s, saida, a, a.diagnostico)
            rel[s] = r
        except Exception as e:                     # scan que nao deu para limpar: copia intacto e avisa
            log = [f"== {s}", f"   ERRO: {type(e).__name__}: {e}; copiado sem alteracao"]
            shutil.rmtree(os.path.join(saida, "data", s), ignore_errors=True)
            shutil.copytree(os.path.join(proj, "data", s), os.path.join(saida, "data", s))
            rel[s] = {"erro": str(e)}
        print("\n".join(log), flush=True)
    for s in todos:
        if s not in scans:
            shutil.copytree(os.path.join(proj, "data", s), os.path.join(saida, "data", s))
    gravar_revo(os.path.join(proj, revo), saida, nome, todos)
    if a.diagnostico:
        json.dump(rel, open(os.path.join(a.diagnostico, "relatorio_limpeza.json"), "w", encoding="utf-8"), indent=2)
    print("saida:", saida)
    print("projeto .revo:", os.path.join(saida, nome + ".revo"))


if __name__ == "__main__":
    main()
