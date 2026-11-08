"""Garante que a interna fique DENTRO da externa pela espessura medida: vertices perto da pele com s > -t sao empurrados a s = -t (s = distancia
com sinal ate a casca externa ao longo da normal externa; vertices mais fundos, nervuras e fixacoes, ficam como estao). Triangulos com centroide
muito por fora (s > --corte) sao removidos antes. A deformacao e continua (min(s,-t)): nao cria degrau.

    python pipeline/encaixar_espessura.py interna.ply externa.ply nuvem_externa.ply --espessura 2.8 --saida out.ply [--corte 3.5] [--perto 6]

LIMITE: onde o scan interno estava desalinhado ate ~3 mm, a pele passa a ter espessura NOMINAL (a medida de 2,8 mm), nao medida ponto a ponto.
"""

import argparse

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna")
ap.add_argument("externa")
ap.add_argument("nuvem")
ap.add_argument("--saida", required=True)
ap.add_argument("--espessura", type=float, default=2.8)
ap.add_argument("--corte", type=float, default=3.5)
ap.add_argument("--perto", type=float, default=6.0)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna)
X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem)
Pn = np.asarray(E.points)
Nn = np.asarray(E.normals)
xp = gpu.xp()
P, _ = gg.amostrar_malha(np.asarray(X.vertices), np.asarray(X.triangles), 2_000_000, saida_gpu=True)
_, j = gpu.vizinho_mais_proximo(Pn, P, dist_max=np.inf)
N = gpu.para_gpu(Nn, xp.float64)[j]  # normal da nuvem mais proxima (GPU)


def sinal(Q):
    d, k = gpu.vizinho_mais_proximo(P, Q, dist_max=np.inf)
    return d, ((Q - P[k]) * N[k]).sum(1), N[k]


V = gpu.para_gpu(np.asarray(I.vertices), xp.float64)
T = np.asarray(I.triangles)
_, sc, _ = sinal(gg.centros_triangulos(V, T, saida_gpu=True))
I.remove_triangles_by_mask(gpu.para_cpu(sc > a.corte))
I.remove_unreferenced_vertices()
V = gpu.para_gpu(np.asarray(I.vertices), xp.float64)
d, s, n = sinal(V)
m = (d < a.perto) & (s > -a.espessura)
V[m] -= (s[m] + a.espessura)[:, None] * n[m]
I.vertices = o3d.utility.Vector3dVector(gpu.para_cpu(V))
print(
    "vertices empurrados para a espessura nominal: %.1f%%" % (float(m.mean()) * 100), "| triangulos:", len(I.triangles)
)
o3d.io.write_triangle_mesh(a.saida, I)
