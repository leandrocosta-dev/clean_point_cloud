"""Desloca a malha interna para DENTRO pela espessura da parede, ao longo da normal externa local.

    python pipeline/deslocar_interna.py interna.ply nuvem_externa.ply --espessura 2.8 --saida out.ply

Por que: o ICP dos scans internos contra a nuvem externa (a unica referencia comum) os puxa ate a pele externa (a espessura e degenerada com um
movimento rigido: o erro e igual para qualquer t). A espessura vem de medida INDEPENDENTE de registro: dentro de um scan unico (20260921_185120) os
pares de folhas frente-a-frente (normais antiparalelas) estao a 2,8 mm (quartis 2,6-3,1; estavel entre as metades da peca).
Normal externa local = media das normais da nuvem externa num raio de 6 mm do ponto mais proximo.
"""

import argparse

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna")
ap.add_argument("nuvem")
ap.add_argument("--espessura", type=float, default=2.8)
ap.add_argument("--saida", required=True)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna)
E = o3d.io.read_point_cloud(a.nuvem)
P = np.asarray(E.points)
N = np.asarray(E.normals)
xp = gpu.xp()
V = gpu.para_gpu(np.asarray(I.vertices), xp.float64)
_, k = gpu.vizinho_mais_proximo(P, V, dist_max=np.inf)
# media das normais a <= 6 mm do ponto mais proximo (GPU; ate os 64 vizinhos mais proximos dentro do raio)
n = gg.media_no_raio(P, N, gpu.para_gpu(P, xp.float64)[k], 6.0, k=64)
n = gpu.para_gpu(np.nan_to_num(n))
n /= xp.linalg.norm(n, axis=1, keepdims=True) + 1e-12
I.vertices = o3d.utility.Vector3dVector(gpu.para_cpu(V - a.espessura * n))
o3d.io.write_triangle_mesh(a.saida, I)
print("interna deslocada", a.espessura, "mm para dentro ->", a.saida)
