"""Limpa a malha interna usando a externa aprovada como referencia e (opcional) junta as duas num arquivo.

    python pipeline/limpar_interna_com_externa.py interna.ply externa.ply nuvem_externa.ply --saida out.ply [--fora 1.5] [--longe 45] [--min-faces 1500] [--juntar uniao.ply]

Remove triangulos da interna cujo centroide esta FORA da casca externa (lado da normal externa, > --fora mm) ou a mais de --longe mm dela
(mesa/cenario/ruido), e componentes < --min-faces. Nao funde as superficies: a uniao e so a concatenacao (2 componentes), pois a parede tem ~2 mm
e uma fusao viraria pele dupla.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna"); ap.add_argument("externa"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--fora", type=float, default=1.5); ap.add_argument("--longe", type=float, default=45.0)
ap.add_argument("--min-faces", type=int, default=1500); ap.add_argument("--juntar")
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna); X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem); E.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
# normais da externa orientadas para fora (a nuvem foi orientada para fora); amostra da malha externa + normais da nuvem vizinha
XS = X.sample_points_uniformly(2_000_000); P = np.asarray(XS.points)
tN = cKDTree(np.asarray(E.points)); _, j = tN.query(P, workers=8); N = np.asarray(E.normals)[j]
tX = cKDTree(P)
Ti = np.asarray(I.triangles); Vi = np.asarray(I.vertices); C = Vi[Ti].mean(1)
d, k = tX.query(C, workers=8); s = np.einsum("ij,ij->i", C - P[k], N[k])
ruim = (s > a.fora) | (d > a.longe)
print("triangulos internos", len(Ti), "removidos (fora da casca ou longe):", int(ruim.sum()), "= %.1f%%" % (ruim.mean() * 100))
I.remove_triangles_by_mask(ruim); I.remove_unreferenced_vertices()
cl, n, _ = I.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
I.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(n < a.min_faces))); I.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(a.saida, I); print("interna limpa:", len(I.triangles), "tri ->", a.saida)
if a.juntar:
    U = X + I; o3d.io.write_triangle_mesh(a.juntar, U); print("uniao (2 cascas separadas):", len(U.triangles), "tri ->", a.juntar)
