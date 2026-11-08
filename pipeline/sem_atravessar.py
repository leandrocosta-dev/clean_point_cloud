"""Remove da interna tudo que atravessa (ou encosta em) a externa; confirma por teste de colisao triangulo-triangulo do MeshLib.

    python pipeline/sem_atravessar.py interna.ply externa.ply nuvem_externa.ply --saida out.ply [--folga 1.0] [--aneis 2] [--min-faces 1500]

1) tira triangulos da interna com centroide a s > -folga mm da casca externa (por fora ou raso demais) em qualquer distancia;
2) repete: findCollidingTriangleBitsets(externa, interna) -> remove os da interna (+ --aneis aneis de vizinhos) ate zero colisoes;
3) tira componentes pequenos. Saida: interna sem colisao com a externa. Nao altera a externa.
"""

import argparse

import gpu
import meshlib.mrmeshpy as mr
import numpy as np
import open3d as o3d

# GPU: normais herdadas e sinal s dos centroides (vizinhos na GPU). CPU: malha do Open3D e MeshLib (colisao triangulo-triangulo,
# expand/deleteFaces), sem equivalente na GPU.

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna")
ap.add_argument("externa")
ap.add_argument("nuvem")
ap.add_argument("--saida", required=True)
ap.add_argument("--folga", type=float, default=1.0)
ap.add_argument("--aneis", type=int, default=2)
ap.add_argument("--min-faces", type=int, default=1500)
a = ap.parse_args()
Xo = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem)
Pn = np.asarray(E.points)
Nn = np.asarray(E.normals)
P = gpu.para_gpu(np.asarray(Xo.sample_points_uniformly(2_000_000).points))
_, j = gpu.vizinho_mais_proximo(Pn, P)
N = gpu.para_gpu(Nn)[j]
I = o3d.io.read_triangle_mesh(a.interna)
V = np.asarray(I.vertices)
T = np.asarray(I.triangles)
C = gpu.para_gpu(V)[gpu.para_gpu(T)].mean(1)
_, k = gpu.vizinho_mais_proximo(P, C)
s = ((C - P[k]) * N[k]).sum(1)
I.remove_triangles_by_mask(gpu.para_cpu(s > -a.folga))
I.remove_unreferenced_vertices()
tmp = a.saida + ".tmp.ply"
o3d.io.write_triangle_mesh(tmp, I)
X = mr.loadMesh(a.externa)
M = mr.loadMesh(tmp)
for it in range(8):
    bx, bi = mr.findCollidingTriangleBitsets(mr.MeshPart(X), mr.MeshPart(M))
    n = bi.count()
    print("iteracao", it, "triangulos da interna em colisao:", n, flush=True)
    if n == 0:
        break
    mr.expand(M.topology, bi, a.aneis)
    M.topology.deleteFaces(bi)
    M.invalidateCaches()
mr.saveMesh(M, tmp)
r = o3d.io.read_triangle_mesh(tmp)
r.remove_unreferenced_vertices()
cl, nn, _ = r.cluster_connected_triangles()
cl = np.asarray(cl)
nn = np.asarray(nn)
r.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(nn < a.min_faces)))
r.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(a.saida, r)
print("interna sem colisao:", len(r.triangles), "tri ->", a.saida)
