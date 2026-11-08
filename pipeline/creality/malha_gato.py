import os

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

from pipeline.creality.limpa_nuvem import R, carregar, maior_componente, objeto

out = r"C:\Users\pc\Desktop\Gatinho_ENTREGA"
p = carregar("pc_before.ply")
n, d, s = objeto(p)
F = o3d.io.read_triangle_mesh(R + r"\result\Fused.ply")
lado = np.sign(np.median(gpu.para_cpu(gpu.para_gpu(np.asarray(F.vertices), np.float64) @ gpu.para_gpu(n) + d)))
m = (s * lado) > 0.6
o = p.select_by_index(np.where(m)[0])
O = maior_componente(o)
O = O.select_by_index(
    np.where(gpu.remover_soltos(np.asarray(O.points), k=29, desvios=3.0))[0]
)  # estatistico 30 vizinhos (com o proprio), 3 desvios
P = np.asarray(O.points)
N = np.asarray(O.normals)
cen = P.mean(0)
print(
    "pontos",
    len(P),
    "frac normais p/ fora",
    float((gpu.para_cpu((gpu.para_gpu(P - cen) * gpu.para_gpu(N)).sum(1)) > 0).mean()),
)
o3d.io.write_point_cloud(os.path.join(out, "gatinho_nuvem_original_limpa.ply"), O)
import time

t = time.time()
mesh, den = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    O, depth=10, width=0, scale=1.1, linear_fit=True
)  # Poisson: Open3D (CPU, sem equivalente)
den = np.asarray(den)
V = np.asarray(mesh.vertices)
print("poisson", len(V), len(mesh.triangles), round(time.time() - t))
dist = gg.distancia_ate(P, V)
print("dist ao ponto: p50 p95 p99", np.percentile(dist, [50, 95, 99]).round(3))
keep = (dist < 0.35) & (den > gg.quantil(den, 0.01))
mesh.remove_vertices_by_mask(~keep)
mesh.remove_degenerate_triangles()
mesh.remove_unreferenced_vertices()
tc, nt = gg.componentes_triangulos(np.asarray(mesh.triangles), len(mesh.vertices))
print("componentes", sorted(nt)[-5:], len(nt))
mesh.remove_triangles_by_mask(nt[tc] < 0.02 * nt.max())
mesh.remove_unreferenced_vertices()
mesh.compute_vertex_normals()
o3d.io.write_triangle_mesh(os.path.join(out, "gatinho_malha_alta_resolucao.ply"), mesh)
print(mesh, "watertight", mesh.is_watertight(), mesh.get_axis_aligned_bounding_box().get_extent())
