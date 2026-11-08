"""Remove os remendos (faces novas apos as N0 originais) cuja maioria das faces discorda das normais da nuvem externa: viram verso.

python pipeline/remover_remendos_discordantes.py malha.ply nuvem_externa.ply --n0 1327032 --saida out.ply
"""

import argparse

import gpu
import numpy as np
import open3d as o3d
from metricas import componentes_grafo

# GPU: adjacencia por aresta, componentes, normais das faces e vizinho mais proximo na nuvem. CPU: leitura/gravacao e a malha do Open3D.

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("malha")
ap.add_argument("nuvem")
ap.add_argument("--n0", type=int, required=True)
ap.add_argument("--saida", required=True)
a = ap.parse_args()
m = gpu.xp()
o = o3d.io.read_triangle_mesh(a.malha)
V = gpu.para_gpu(np.asarray(o.vertices))
T = np.asarray(o.triangles)
nu = o3d.io.read_point_cloud(a.nuvem)
NN = gpu.para_gpu(np.asarray(nu.normals))
Tn = gpu.para_gpu(T[a.n0 :]).astype(m.int64)
E = m.sort(m.concatenate([Tn[:, [0, 1]], Tn[:, [1, 2]], Tn[:, [2, 0]]]), axis=1)
fid = m.tile(m.arange(len(Tn)), 3)
key = E[:, 0] * len(V) + E[:, 1]
od = m.argsort(key)
k = key[od]
f = fid[od]
s = k[1:] == k[:-1]
x, y = f[:-1][s], f[1:][s]  # faces que compartilham aresta
n, lab = componentes_grafo(len(Tn), x, y)
lab = gpu.para_gpu(lab)
A, B, C = V[Tn[:, 0]], V[Tn[:, 1]], V[Tn[:, 2]]
nf = m.cross(B - A, C - A)
nf /= m.linalg.norm(nf, axis=1, keepdims=True) + 1e-12
cen = (A + B + C) / 3
_, j = gpu.vizinho_mais_proximo(gpu.para_gpu(np.asarray(nu.points)), cen)
dot = (nf * NN[j]).sum(1)
discorda = m.bincount(lab, weights=(dot < 0).astype(m.float64), minlength=n) / m.maximum(
    m.bincount(lab, minlength=n), 1
)
rm = gpu.para_cpu((discorda > 0.5)[lab])
print("remendos", n, "faces removidas", int(rm.sum()), "de", len(Tn))
o2 = o3d.geometry.TriangleMesh(o.vertices, o3d.utility.Vector3iVector(np.concatenate([T[: a.n0], T[a.n0 :][~rm]])))
o2.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(a.saida, o2)
