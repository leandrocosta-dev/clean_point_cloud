"""Metricas de malha iguais as do painel Information do MeshInspector: triangulos, vertices, componentes, furos (todos os contornos abertos).

    python pipeline/metrica_malha.py malha1.ply [malha2.ply ...]
Calibrado com o MeshInspector 2.5.9 em parachoque_malha_1M_v8.ply (538 furos, 384 componentes).
"""
import sys
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components

def metricas(M):
    V = np.asarray(M.vertices); Tr = np.asarray(M.triangles)
    e = np.vstack([Tr[:, [0, 1]], Tr[:, [1, 2]], Tr[:, [2, 0]]]); e.sort(1)
    k = e[:, 0].astype(np.int64) * len(V) + e[:, 1]; u, cn = np.unique(k, return_counts=True)
    bd = u[cn == 1]; b0 = (bd // len(V)).astype(int); b1 = (bd % len(V)).astype(int)
    G = sparse.coo_matrix((np.ones(len(b0)), (b0, b1)), shape=(len(V),) * 2)
    _, lab = connected_components(G, directed=False)
    vs = np.unique(np.concatenate([b0, b1])); nloops = len(np.unique(lab[vs]))
    _, ncomp, _ = M.cluster_connected_triangles()
    return dict(triangulos=len(Tr), vertices=len(V), componentes=len(ncomp), furos=nloops, borda_mm=float(np.linalg.norm(V[b0] - V[b1], axis=1).sum()))

if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, metricas(o3d.io.read_triangle_mesh(f)))
