"""Malha por Poisson sobre a nuvem completa, aparada para casca aberta.

    python pipeline/malha_poisson.py <nuvem.ply (com normais)> --saida <malha.ply> [--profundidade 11] [--dist-max 1.5]

Poisson fecha a superficie; o pára-choque e uma casca ABERTA, entao se descartam (a) vertices de baixa densidade
(quantil `--quantil`) e (b) tudo a mais de `--dist-max` mm do ponto mais proximo da nuvem.
"""
import argparse, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade
ap = argparse.ArgumentParser(); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--profundidade", type=int, default=11); ap.add_argument("--quantil", type=float, default=0.02)
ap.add_argument("--dist-max", type=float, default=1.5); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args(); baixa_prioridade(); t0 = time.time()
pc = o3d.io.read_point_cloud(a.nuvem); P = np.asarray(pc.points)
print(f"{len(P):,} pontos; Poisson profundidade {a.profundidade}", flush=True)
M, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=a.profundidade, width=0, scale=1.05, linear_fit=False, n_threads=a.threads)
dens = np.asarray(dens); V = np.asarray(M.vertices)
print(f"Poisson: {len(V):,} vertices, {len(M.triangles):,} triangulos ({time.time()-t0:.0f}s)", flush=True)
d, _ = cKDTree(P).query(V, workers=4)
rem = (dens < np.quantile(dens, a.quantil)) | (d > a.dist_max)
M.remove_vertices_by_mask(rem)
M.remove_degenerate_triangles(); M.remove_unreferenced_vertices()
tri_cl, n_tri, area = M.cluster_connected_triangles()
tri_cl = np.asarray(tri_cl); n_tri = np.asarray(n_tri)
peq = np.isin(tri_cl, np.flatnonzero(n_tri < 2000))
M.remove_triangles_by_mask(peq); M.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(a.saida, M, write_vertex_normals=False)
print(f"malha final: {len(M.vertices):,} vertices, {len(M.triangles):,} triangulos | {time.time()-t0:.0f}s")
