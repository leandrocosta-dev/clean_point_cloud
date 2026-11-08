"""Alinha a malha ORIGINAL (Revo) de um scan ao referencial da nuvem final e mede a cobertura dela sobre a pele externa.

    python pipeline/base_por_cobertura.py <projeto_original> <scan> <segmentos_indice_pasta> <direto.json> <T_seg_revo.npy> <nuvem_externa.ply> --saida <pasta>

A malha do scan (fuse_mesh.ply, lisa) esta no referencial do proprio scan; usa-se a pose do segmento mais pesado + T_seg_revo como
chute e refina-se com ICP ponto-plano robusto contra a nuvem externa. Fica so o que concorda com a nuvem (tira mesa/cenario).
Saida: <scan>_base.ply (malha alinhada e aparada) e cobertura.json.
"""

import argparse
import json
import os

import gpu
import numpy as np
import open3d as o3d
from metricas import icp_robusto, normais_da_nuvem

# GPU: vizinhos, ICP e normais (gpu.py). CPU so malha do Open3D (leitura/gravacao, transform, amostragem, remove_*_by_mask).

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("projeto")
ap.add_argument("scan")
ap.add_argument("pasta_seg")
ap.add_argument("direto")
ap.add_argument("tseg")
ap.add_argument("externa")
ap.add_argument("--saida", required=True)
ap.add_argument("--tol", type=float, default=1.2)
a = ap.parse_args()
os.makedirs(a.saida, exist_ok=True)
ind = json.load(open(os.path.join(a.pasta_seg, "segmentos_indice.json")))[a.scan]
pj = json.load(open(a.direto))["segmentos"][a.scan]
Tr = np.load(a.tseg)
seg = max(ind, key=lambda s: len(np.load(s["arquivo"])))
T0 = Tr @ np.array(next(x for x in pj if x["seg"] == seg["seg"])["T"])
M = o3d.io.read_triangle_mesh(os.path.join(a.projeto, "data", a.scan, "fuse_mesh.ply"))
M.transform(T0)
E = o3d.io.read_point_cloud(a.externa)
normais_da_nuvem(E, 4, 30)
Ep = np.asarray(E.points)
S = np.asarray(M.sample_points_uniformly(1_500_000).points)
T, r = icp_robusto(S, Ep, np.asarray(E.normals), tols=(8, 4, 2, 1), iteracoes=50)  # ICP ponto-plano robusto (GPU)
M.transform(T)
_, j = gpu.vizinho_mais_proximo(Ep, gpu.para_gpu(np.asarray(M.vertices)), dist_max=a.tol)
dentro = j >= 0
manter = gpu.para_cpu(dentro[gpu.para_gpu(np.asarray(M.triangles))].all(1))
B = o3d.geometry.TriangleMesh(M)
B.remove_triangles_by_mask(~manter)
B.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(os.path.join(a.saida, a.scan + "_base.ply"), B, write_vertex_normals=False)
d2, _ = gpu.vizinho_mais_proximo(
    gpu.para_gpu(np.asarray(B.sample_points_uniformly(3_000_000).points)), gpu.para_gpu(Ep), dist_max=2.0
)  # inf alem de 2 mm
cob = {f"cobertura_{t}mm": float((d2 < t).mean()) for t in (0.5, 1.0, 2.0)}
print(
    a.scan,
    "fitness ICP %.3f rmse %.3f" % (r["fitness"], r["rmse"]),
    "| vertices mantidos %.1f%%" % (float(dentro.mean()) * 100),
    "| cobertura da pele externa:",
    cob,
)
json.dump(dict(scan=a.scan, T=(T @ T0).tolist(), **cob), open(os.path.join(a.saida, a.scan + "_cobertura.json"), "w"))
