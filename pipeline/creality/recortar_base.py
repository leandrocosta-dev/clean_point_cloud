"""Recorta a mesa que sobrou na base da malha fundida e nivela. Uso: recortar_base.py <entrada.ply> <saida.ply>

Plano da mesa (RANSAC no scan A): n, d0; o gato esta em s<0. Pegada = contorno das paredes (vertices com -8<s<=-4),
rasterizado a 0,1 mm, dilatado 0,4 mm e preenchido. Triangulos com algum vertice em s>-4 fora da pegada sao apagados
(faixas de mesa). Depois: maior componente, furos fechados (MeshLib), malha girada para z=normal da mesa e apoiada em z=0.
"""

import sys

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d
from meshlib import mrmeshpy as mm

xp = gpu.xp()
ndi = gg.ndi()
ent, sai = sys.argv[1:3]
n = np.array([0.935, -0.064, 0.349])
d0 = -124.841 / np.linalg.norm(n)
n /= np.linalg.norm(n)
u = np.cross(n, [0, 0, 1.0])
u /= np.linalg.norm(u)
v = np.cross(n, u)
M = o3d.io.read_triangle_mesh(ent)
V = gpu.para_gpu(np.asarray(M.vertices), np.float64)
T = gpu.para_gpu(np.asarray(M.triangles))
s = V @ gpu.para_gpu(n) + d0
U = V @ gpu.para_gpu(u)
W = V @ gpu.para_gpu(v)
ring = (s > -8) & (s <= -4)
res = 0.1
u0, v0 = float(U[ring].min()) - 3, float(W[ring].min()) - 3
nx = int((float(U[ring].max()) + 3 - u0) / res) + 1
ny = int((float(W[ring].max()) + 3 - v0) / res) + 1
g = xp.zeros((nx, ny), bool)
g[((U[ring] - u0) / res).astype(int), ((W[ring] - v0) / res).astype(int)] = True
g = gg.dilatar(g, int(0.4 / res) + 1)
g = ndi.binary_fill_holes(g)  # pegada: raster, dilatacao e preenchimento na GPU
ix = xp.clip(((U - u0) / res).astype(int), 0, nx - 1)
iy = xp.clip(((W - v0) / res).astype(int), 0, ny - 1)
dentro = g[ix, iy] & (U >= u0) & (W >= v0)
ruim = (s > -4) & ~dentro
print("vertices de mesa:", int(ruim.sum()))
M.remove_triangles_by_mask(gpu.para_cpu(ruim[T].any(1)))
M.remove_unreferenced_vertices()
tc, nt = gg.componentes_triangulos(np.asarray(M.triangles), len(M.vertices))
M.remove_triangles_by_mask(tc != 0)
M.remove_unreferenced_vertices()  # rotulo 0 = maior componente
# nivelar: normal da mesa vira -z... gato em s<0 => z = -s; rotacao propria (det=+1) levando -n para +z
z = -n
x = u
y = np.cross(z, x)
R = np.vstack([x, y, z])
Vg = gpu.para_gpu(np.asarray(M.vertices), np.float64) @ gpu.para_gpu(R.T)
Vg[:, 2] -= Vg[:, 2].min()
V = gpu.para_cpu(Vg)
M.vertices = o3d.utility.Vector3dVector(V)
o3d.io.write_triangle_mesh(sai, M)
m = mm.loadMesh(sai)
print("antes do fechamento: furos", len(m.topology.findHoleRepresentiveEdges()))
mm.fixMultipleEdges(m)
p = mm.FillHoleNicelyParams() if hasattr(mm, "FillHoleNicelyParams") else None
for e in m.topology.findHoleRepresentiveEdges():
    mm.fillHole(m, e) if p is None else mm.fillHoleNicely(m, e, p)
print(
    "depois: furos",
    len(m.topology.findHoleRepresentiveEdges()),
    "comp",
    len(mm.MeshComponents.getAllComponents(m)),
    "faces",
    m.topology.numValidFaces(),
    "vol",
    round(m.volume()),
)
mm.saveMesh(m, sai)
