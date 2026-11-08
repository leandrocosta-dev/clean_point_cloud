"""Completa uma malha de scans (construcao por cobertura) com pedacos de uma malha Poisson SO onde a primeira nao cobre.

    python pipeline/completar_com_poisson.py base.ply poisson.ply --saida out.ply [--tol-lacuna 2.0] [--abertura 3.0] [--min-tri 500]

Usar quando falta a malha original (fuse_mesh) de algum scan: os triangulos da Poisson a > --tol-lacuna mm da base (interior de lacuna: >= 60% da vizinhanca
de --abertura mm tambem e lacuna) entram, por componente >= --min-tri. Nada e promediado; os pedacos Poisson sao mais rugosos que a malha de scan (documentar).
Mesmo referencial (sem realinhamento).
"""

import argparse

import gpu
import numpy as np
import open3d as o3d
from metricas import componentes_grafo

# GPU: vizinhos, contagens e componentes. CPU so malha do Open3D (leitura/gravacao, amostragem, remove_*_by_mask, soma de malhas).

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("base")
ap.add_argument("poisson")
ap.add_argument("--saida", required=True)
ap.add_argument("--tol-lacuna", type=float, default=2.0)
ap.add_argument("--abertura", type=float, default=3.0)
ap.add_argument("--min-tri", type=int, default=500)
a = ap.parse_args()
B = o3d.io.read_triangle_mesh(a.base)
S = o3d.io.read_triangle_mesh(a.poisson)
Bs = np.asarray(B.sample_points_uniformly(2_500_000).points)
Sv = np.asarray(S.vertices)
St = np.asarray(S.triangles)
mg = gpu.xp()
cen = gpu.para_gpu(Sv)[gpu.para_gpu(St)].mean(1)
_, jb = gpu.vizinho_mais_proximo(Bs, cen, dist_max=a.tol_lacuna)
lac = jb < 0
if bool(lac.any()):
    # fracao de lacuna na vizinhanca de --abertura = (centroides de lacuna no raio) / (todos no raio), contagem na GPU
    cl = cen[lac]
    frac = gpu.contar_no_raio(cl, cl, a.abertura) / gpu.contar_no_raio(cen, cl, a.abertura)
    idx = gpu.para_cpu(mg.flatnonzero(lac)[frac >= 0.6])
else:
    idx = np.zeros(0, np.int64)
if len(idx):
    _, inv, _ = gpu.agrupar(gpu.para_gpu(St[idx].ravel()))
    inv = inv.reshape(-1, 3)
    _, lab = componentes_grafo(
        int(inv.max()) + 1, mg.concatenate([inv[:, 0], inv[:, 1]]), mg.concatenate([inv[:, 1], inv[:, 2]])
    )
    lab = lab[gpu.para_cpu(inv[:, 0])]
    tam = np.bincount(lab)  # triangulos que compartilham vertice = mesmo componente
    keep = idx[np.isin(lab, np.flatnonzero(tam >= a.min_tri))]
else:
    keep = idx
m = np.ones(len(St), bool)
m[keep] = False
sub = o3d.geometry.TriangleMesh(S)
sub.remove_triangles_by_mask(m)
sub.remove_unreferenced_vertices()
U = B + sub
print(
    "triangulos da base",
    len(B.triangles),
    "| pedacos Poisson acrescentados",
    len(sub.triangles),
    "| total",
    len(U.triangles),
)
o3d.io.write_triangle_mesh(a.saida, U)
