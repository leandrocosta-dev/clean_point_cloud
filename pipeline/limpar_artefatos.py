"""Tira artefatos soltos de uma nuvem JA FUNDIDA (ex.: nuvem_nitida.ply do fundir_nitido): pontos voando,
franja rala nas bordas e grupinhos desconectados, sem comer a borda real da superficie.

    python pipeline/limpar_artefatos.py nuvem.ply nuvem_limpa.ply [--dist-nucleo 0.25 --fora-plano 0.03]

Nucleo = pontos com >= --viz-nucleo vizinhos em --raio mm. Ponto ralo so sai se estiver a > --dist-nucleo mm do
nucleo ou a > --fora-plano mm do plano local do nucleo; depois saem componentes com < --comp-min pontos.
Grava tambem <saida>_removidos.ply para conferir. Remover so pelo numero de vizinhos, repetido, come a borda real
(Projeto 8: 37% dos removidos estavam no plano da superficie).
Contagens, distancias, planos locais e componentes na GPU (gpu.py); na CPU ficam so a leitura e a gravacao dos PLY (Open3D).
Componentes = grafo exato de pares a <= --raio, sem truncar a quantidade de vizinhos, com uniao de celulas na GPU.
"""

import argparse
import json

import gpu
import numpy as np
import open3d as o3d
import pca3_gpu
from conectividade_gpu import componentes_por_raio as _componentes_por_raio


def componentes_por_raio(X, raio, k=64):
    """Grafo exato a <= raio; k e mantido so por compatibilidade e nao limita arestas."""
    return _componentes_por_raio(X, raio)


def limpar(X, raio=0.2, viz_nucleo=15, dist_nucleo=0.15, comp_min=500, fora_plano=0.03):
    """Mascara bool numpy (True = manter) e dicionario de motivos. X (n, 3) numpy ou cupy."""
    m = gpu.xp()
    X = gpu.para_gpu(X, m.float64)
    manter = m.ones(len(X), bool)
    motivos = {}
    # 1-2) nucleo denso (>= --viz-nucleo vizinhos); ponto ralo so sai se nao continua a superficie do nucleo:
    #      longe dele (> --dist-nucleo) ou fora do plano local do nucleo (> --fora-plano). Borda real fica.
    n = gpu.contar_no_raio(X, X, raio) - 1
    nucleo = n >= viz_nucleo
    idx_n = m.flatnonzero(nucleo)
    ralo = m.flatnonzero(~nucleo)
    Xn = X[idx_n]
    d, _ = gpu.vizinho_mais_proximo(Xn, X[ralo])
    # plano local do nucleo em r = 0,3 mm em volta do ponto ralo; precisa >= 6 pontos
    off = m.full(len(ralo), m.inf)
    gr = gpu.Grade(Xn, 0.3)
    for a, b in gpu.blocos(len(ralo), bytes_por_item=1500):
        S = gr.somas(X[ralo[a:b]], None, gpu=True).astype(m.float64)
        s0 = S[:, 0]
        media = S[:, 1:4] / m.maximum(s0, 1e-30)[:, None]  # (centroide - ponto), relativo ao ponto
        xx, yy, zz, xy, xz, yz = (S[:, c] for c in range(4, 10))
        M = m.empty((b - a, 3, 3), m.float64)
        M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = xx, yy, zz
        M[:, 0, 1] = M[:, 1, 0] = xy
        M[:, 0, 2] = M[:, 2, 0] = xz
        M[:, 1, 2] = M[:, 2, 1] = yz
        M -= s0[:, None, None] * (media[:, :, None] * media[:, None, :])
        _, nrm = pca3_gpu.menor_autopar(M)
        o = m.abs((-media * nrm).sum(1))  # |(x - centroide) . normal|
        off[a:b] = m.where(s0 >= 6, o, m.inf)
    ruim = (d > dist_nucleo) | (off > fora_plano)
    manter[ralo[ruim]] = False
    motivos.update(
        nucleo=int(nucleo.sum()),
        ralos=int(len(ralo)),
        ralos_longe=int((d > dist_nucleo).sum()),
        ralos_fora_plano=int(((off > fora_plano) & (d <= dist_nucleo)).sum()),
    )
    # 3) componentes conectados pequenos
    idx = m.flatnonzero(manter)
    rot, sz = componentes_por_raio(X[idx], raio)
    ruim = sz[rot] < comp_min
    manter[idx[ruim]] = False
    motivos["componentes_pequenos"] = int(ruim.sum())
    motivos["componentes_antes"] = int(len(sz))
    motivos.update(entrada=len(X), saida=int(manter.sum()), removidos=int((~manter).sum()))
    return gpu.para_cpu(manter), motivos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entrada")
    ap.add_argument("saida")
    ap.add_argument("--raio", type=float, default=0.2)
    ap.add_argument("--viz-nucleo", type=int, default=15)
    ap.add_argument("--dist-nucleo", type=float, default=0.15)
    ap.add_argument("--comp-min", type=int, default=500)
    ap.add_argument(
        "--fora-plano",
        type=float,
        default=0.03,
        help="mm do plano local do nucleo (r 0,3); ponto ralo acima disso = voando",
    )
    a = ap.parse_args()
    gpu.xp()
    p = o3d.io.read_point_cloud(a.entrada)
    X = np.asarray(p.points)
    manter, motivos = limpar(X, a.raio, a.viz_nucleo, a.dist_nucleo, a.comp_min, a.fora_plano)
    q = p.select_by_index(np.flatnonzero(manter))
    o3d.io.write_point_cloud(a.saida, q)
    o3d.io.write_point_cloud(a.saida.replace(".ply", "_removidos.ply"), p.select_by_index(np.flatnonzero(~manter)))
    print(json.dumps(motivos, indent=1))


if __name__ == "__main__":
    main()
