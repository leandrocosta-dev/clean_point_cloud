"""Malha final = malha do Revo de um scan (base, intocada) + pedacos de A SO onde a base nao cobre e ha dado medido.

Uso: python pipeline/revo_antigo/completar_base_revo.py base.ply unida.ply --saida out.ply [--lacuna 1.2 --dado-max 0.4 --min-tri 300]
Poisson (prof. 8) da nuvem unida; descarta o que esta a > --dado-max mm da nuvem (isso tira todo fechamento de aberturas
reais, pois peca oca) e fica so com triangulos a > --lacuna mm da base, por componente >= --min-tri. Nada fecha furo.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--lacuna", type=float, default=1.2); ap.add_argument("--dado-max", type=float, default=0.4)
ap.add_argument("--min-tri", type=int, default=300); ap.add_argument("--prof", type=int, default=8)
a = ap.parse_args()
base = o3d.io.read_triangle_mesh(a.base); pc = o3d.io.read_point_cloud(a.nuvem); P = np.asarray(pc.points)
M, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=a.prof, scale=1.05)
V = np.asarray(M.vertices); dd, _ = cKDTree(P).query(V)
M.remove_vertices_by_mask((dd > a.dado_max) | (np.asarray(dens) < np.quantile(np.asarray(dens), 0.01)))
M.remove_unreferenced_vertices(); V = np.asarray(M.vertices); F = np.asarray(M.triangles)
tb = cKDTree(np.asarray(base.sample_points_uniformly(2_000_000).points))
cen = V[F].mean(1); d, _ = tb.query(cen); M.remove_triangles_by_mask(d <= a.lacuna); M.remove_unreferenced_vertices()
cl, n, _ = M.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
M.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(n < a.min_tri))); M.remove_unreferenced_vertices()
print("pedacos acrescentados:", len(M.triangles), "triangulos em", int((n >= a.min_tri).sum()), "componentes; base", len(base.triangles))
out = base + M
o3d.io.write_triangle_mesh(a.saida, out, write_vertex_normals=False)
