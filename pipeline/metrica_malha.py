"""Metricas de malha iguais as do painel Information do MeshInspector: triangulos, vertices, componentes, furos (todos os contornos abertos).

    python pipeline/metrica_malha.py malha1.ply [malha2.ply ...]
Calibrado com o MeshInspector 2.5.9 em parachoque_malha_1M_v8.ply (538 furos, 384 componentes).
"""

import sys

import gpu
import numpy as np
import open3d as o3d
from metricas import componentes_grafo

# GPU: arestas de borda, contornos (componentes) e comprimento da borda. CPU: leitura da malha e o cluster_connected_triangles do Open3D.


def metricas(M):
    m = gpu.xp()
    V = gpu.para_gpu(np.asarray(M.vertices))
    Tr = gpu.para_gpu(np.asarray(M.triangles)).astype(m.int64)
    e = m.sort(m.concatenate([Tr[:, [0, 1]], Tr[:, [1, 2]], Tr[:, [2, 0]]]), axis=1)
    u, _, cn = gpu.agrupar(e[:, 0] * len(V) + e[:, 1])
    bd = u[cn == 1]
    b0 = bd // len(V)
    b1 = bd % len(V)
    _, lab = componentes_grafo(len(V), b0, b1)
    vs = m.unique(m.concatenate([b0, b1]))
    nloops = len(np.unique(lab[gpu.para_cpu(vs)]))
    _, ncomp, _ = M.cluster_connected_triangles()
    return dict(
        triangulos=len(Tr),
        vertices=len(V),
        componentes=len(ncomp),
        furos=nloops,
        borda_mm=float(m.linalg.norm(V[b0] - V[b1], axis=1).sum()),
    )


if __name__ == "__main__":
    for f in sys.argv[1:]:
        print(f, metricas(o3d.io.read_triangle_mesh(f)))
