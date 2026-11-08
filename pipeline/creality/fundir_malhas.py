"""Funde A + B (ambas do Creality Scan, B ja alinhada em A). Uso: fundir_malhas.py <A.ply> <B_alinhada.ply> <saida.ply> [--corte 1.5]

- Plano da mesa (RANSAC no scan A), gato em s<0. A malha do software em A traz um pedaco da mesa colado no plano: apaga os
  pontos de A com s > -8 mm cuja projecao no plano cai fora da pegada do gato (casco convexo dos pontos de B com s > -8,
  alargado 2 mm). A base plana do gato (oval com furo) vem de B, nao e mesa.
- De B entra so o que A (ja sem mesa) nao cobre (dist > corte). Poisson depth 10, corte a 0,3 mm.
"""

import sys
import time

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d
from scipy.spatial import (
    ConvexHull,
)  # casco 2-D (Qhull, CPU: so os vertices do contorno); o teste de pertinencia roda na GPU

xp = gpu.xp()
a, b, out = sys.argv[1:4]
corte = float(sys.argv[sys.argv.index("--corte") + 1]) if "--corte" in sys.argv else 1.5
MA = o3d.io.read_triangle_mesh(a)
MB = o3d.io.read_triangle_mesh(b)
PA, NA = gg.amostrar_malha(
    np.asarray(MA.vertices), np.asarray(MA.triangles), 1500000, saida_gpu=True
)  # normais do triangulo
PB, NB = gg.amostrar_malha(np.asarray(MB.vertices), np.asarray(MB.triangles), 1000000, saida_gpu=True)
n = np.array([0.935, -0.064, 0.349])
d0 = -124.841 / np.linalg.norm(n)
n /= np.linalg.norm(n)
u = np.cross(n, [0, 0, 1.0])
u /= np.linalg.norm(u)
v = np.cross(n, u)
ng, ug, vg = gpu.para_gpu(n), gpu.para_gpu(u), gpu.para_gpu(v)
sA = PA @ ng + d0
sB = PB @ ng + d0
q = PB[sB > -8]
Q = gpu.para_cpu(xp.stack([q @ ug, q @ vg], 1))
h = ConvexHull(Q)
H = Q[h.vertices]
c = H.mean(0)
H = H + 2.0 * (H - c) / np.linalg.norm(H - c, axis=1, keepdims=True)
eq = gpu.para_gpu(
    ConvexHull(H).equations
)  # (a, b, c): dentro do casco convexo de H se a x + b y + c <= 0 em todas (= Delaunay.find_simplex >= 0)
dentro = ((xp.stack([PA @ ug, PA @ vg], 1) @ eq[:, :2].T + eq[:, 2]) <= 1e-9).all(1)
mesa = (sA > -8) & ~dentro
print("A: pontos de mesa removidos", int(mesa.sum()), "de", len(PA))
PA, NA = PA[~mesa], NA[~mesa]
dB = gg.distancia_ate(PA, PB, saida_gpu=True)
novo = dB > corte
print("B novo", int(novo.sum()), "de", len(PB))
PF = xp.concatenate([PA, PB[novo]])
NF = xp.concatenate([NA, NB[novo]])
F = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(PF)))
F.normals = o3d.utility.Vector3dVector(gpu.para_cpu(NF))
t = time.time()
m, den = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(F, depth=10, width=0, scale=1.05, linear_fit=True)
den = np.asarray(den)
print("poisson", len(m.vertices), round(time.time() - t))  # Poisson: Open3D (CPU)
V = np.asarray(m.vertices)
dist = gg.distancia_ate(PF, V)
keep = (dist < 0.3) & (den > gg.quantil(den, 0.005))
m.remove_vertices_by_mask(~keep)
m.remove_degenerate_triangles()
m.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(out, m)
print(m)
