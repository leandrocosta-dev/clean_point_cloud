"""Completa uma malha de scans (construcao por cobertura) com pedacos de uma malha Poisson SO onde a primeira nao cobre.

    python pipeline/completar_com_poisson.py base.ply poisson.ply --saida out.ply [--tol-lacuna 2.0] [--abertura 3.0] [--min-tri 500]

Usar quando falta a malha original (fuse_mesh) de algum scan: os triangulos da Poisson a > --tol-lacuna mm da base (interior de lacuna: >= 60% da vizinhanca
de --abertura mm tambem e lacuna) entram, por componente >= --min-tri. Nada e promediado; os pedacos Poisson sao mais rugosos que a malha de scan (documentar).
Mesmo referencial (sem realinhamento).
"""
import argparse
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("base"); ap.add_argument("poisson"); ap.add_argument("--saida", required=True)
ap.add_argument("--tol-lacuna", type=float, default=2.0); ap.add_argument("--abertura", type=float, default=3.0); ap.add_argument("--min-tri", type=int, default=500)
a = ap.parse_args()
B = o3d.io.read_triangle_mesh(a.base); S = o3d.io.read_triangle_mesh(a.poisson)
tB = cKDTree(np.asarray(B.sample_points_uniformly(2_500_000).points))
Sv = np.asarray(S.vertices); St = np.asarray(S.triangles); cen = Sv[St].mean(1)
d, _ = tB.query(cen, workers=8); lac = d > a.tol_lacuna
tc = cKDTree(cen); viz = tc.query_ball_point(cen[lac], a.abertura)
frac = np.array([lac[v].mean() for v in viz]); idx = np.flatnonzero(lac)[frac >= 0.6]
vv = St[idx].ravel(); tid = np.repeat(np.arange(len(idx)), 3); _, inv = np.unique(vv, return_inverse=True)
G = sparse.coo_matrix((np.ones(len(inv)), (tid, inv)), shape=(len(idx), inv.max() + 1)).tocsr()
nc, lab = connected_components(G @ G.T, directed=False); tam = np.bincount(lab)
keep = idx[np.isin(lab, np.flatnonzero(tam >= a.min_tri))]
m = np.ones(len(St), bool); m[keep] = False; sub = o3d.geometry.TriangleMesh(S); sub.remove_triangles_by_mask(m); sub.remove_unreferenced_vertices()
U = B + sub; print("triangulos da base", len(B.triangles), "| pedacos Poisson acrescentados", len(sub.triangles), "| total", len(U.triangles))
o3d.io.write_triangle_mesh(a.saida, U)
