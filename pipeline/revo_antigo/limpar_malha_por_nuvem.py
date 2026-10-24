"""Limpa a malha do Revo de um scan com a nuvem ja limpa dos quadros: so ficam triangulos apoiados na nuvem limpa.

Uso: python pipeline/revo_antigo/limpar_malha_por_nuvem.py malha.ply nuvem_limpa.npy --saida out.ply [--tol 1.0 --min-frac 0.2]
Triangulo fica se >= 2 dos 3 vertices estao a <= --tol mm de um ponto limpo; depois so componentes >= --min-frac do maior.
(a malha e a nuvem estao no mesmo referencial: poses do proprio scan.)
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(); ap.add_argument("malha"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--min-frac", type=float, default=0.2); a = ap.parse_args()
m = o3d.io.read_triangle_mesh(a.malha); V = np.asarray(m.vertices); F = np.asarray(m.triangles)
d, _ = cKDTree(np.load(a.nuvem)).query(V, workers=4); ok = (d <= a.tol)[F].sum(1) >= 2
m.remove_triangles_by_mask(~ok); m.remove_unreferenced_vertices()
cl, n, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
m.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(n < a.min_frac * n.max()))); m.remove_unreferenced_vertices()
print(f"triangulos {len(F)} -> {len(m.triangles)} ({100*len(m.triangles)/len(F):.1f}%), componentes finais {int((n >= a.min_frac*n.max()).sum())}")
o3d.io.write_triangle_mesh(a.saida, m, write_vertex_normals=False)
