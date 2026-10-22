"""Garante que a interna fique DENTRO da externa pela espessura medida: vertices perto da pele com s > -t sao empurrados a s = -t (s = distancia
com sinal ate a casca externa ao longo da normal externa; vertices mais fundos, nervuras e fixacoes, ficam como estao). Triangulos com centroide
muito por fora (s > --corte) sao removidos antes. A deformacao e continua (min(s,-t)): nao cria degrau.

    python pipeline/encaixar_espessura.py interna.ply externa.ply nuvem_externa.ply --espessura 2.8 --saida out.ply [--corte 3.5] [--perto 6]

LIMITE: onde o scan interno estava desalinhado ate ~3 mm, a pele passa a ter espessura NOMINAL (a medida de 2,8 mm), nao medida ponto a ponto.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna"); ap.add_argument("externa"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--espessura", type=float, default=2.8); ap.add_argument("--corte", type=float, default=3.5); ap.add_argument("--perto", type=float, default=6.0)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna); X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem); Pn = np.asarray(E.points); Nn = np.asarray(E.normals)
P = np.asarray(X.sample_points_uniformly(2_000_000).points); _, j = cKDTree(Pn).query(P, workers=8); N = Nn[j]; t = cKDTree(P)
def sinal(Q):
    d, k = t.query(Q, workers=8); return d, np.einsum("ij,ij->i", Q - P[k], N[k]), N[k]
V = np.asarray(I.vertices); T = np.asarray(I.triangles)
_, sc, _ = sinal(V[T].mean(1)); I.remove_triangles_by_mask(sc > a.corte); I.remove_unreferenced_vertices()
V = np.asarray(I.vertices).copy(); d, s, n = sinal(V)
m = (d < a.perto) & (s > -a.espessura)
V[m] -= ((s[m] + a.espessura))[:, None] * n[m]
I.vertices = o3d.utility.Vector3dVector(V)
print("vertices empurrados para a espessura nominal: %.1f%%" % (m.mean() * 100), "| triangulos:", len(I.triangles))
o3d.io.write_triangle_mesh(a.saida, I)
