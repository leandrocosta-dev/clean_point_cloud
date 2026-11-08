"""Alinha a malha B na malha A (ambas exportadas do Creality Scan, mm). Uso: alinhar_malhas.py <A.ply> <B.ply> <saida_dir>

Amostragem, decimacao, normais, ICP ponto-plano e distancias na GPU; FPFH + RANSAC por features em Open3D (CPU)."""

import os
import sys

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

reg = o3d.pipelines.registration
A_path, B_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
MA = o3d.io.read_triangle_mesh(A_path)
MB = o3d.io.read_triangle_mesh(B_path)
A, NA = gg.amostrar_malha(np.asarray(MA.vertices), np.asarray(MA.triangles), 800000, saida_gpu=True)
B, NB = gg.amostrar_malha(np.asarray(MB.vertices), np.asarray(MB.triangles), 600000, saida_gpu=True)
vox = 1.2


def prep(P, N0):
    C, Nm = gg.reduzir_voxel_normais(P, N0, vox, saida_gpu=True)
    N = gg.alinhar_sinal(gg.normais_hibrido(C, vox * 3, 40, saida_gpu=True), Nm, saida_gpu=True)
    q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(C)))
    q.normals = o3d.utility.Vector3dVector(gpu.para_cpu(N))
    return q, reg.compute_fpfh_feature(q, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 7, max_nn=150))


qa, fa = prep(A, NA)
qb, fb = prep(B, NB)
res = []
for seed in range(8):
    o3d.utility.random.seed(seed)
    r = reg.registration_ransac_based_on_feature_matching(
        qb,
        qa,
        fb,
        fa,
        True,
        vox * 2,
        reg.TransformationEstimationPointToPoint(False),
        3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(vox * 2)],
        reg.RANSACConvergenceCriteria(3000000, 0.999),
    )
    T, ic = gg.icp_encadeado(
        B, A, NA, r.transformation, (2.0, 1.0, 0.6, 0.4, 0.25), iteracoes=60
    )  # ICP ponto-plano na GPU
    print(seed, "ransac", round(r.fitness, 3), "icp", round(ic["fitness"], 3), "rmse", round(ic["rmse"], 3), flush=True)
    res.append((ic["fitness"], T))
res.sort(key=lambda x: -x[0])
T = res[0][1]
np.save(os.path.join(out, "T_B_para_A.npy"), T)
Bt = gg.transformar(B, T, saida_gpu=True)
d = gg.distancia_ate(A, Bt)
print("B->A pcts 25/50/75/90", np.percentile(d, [25, 50, 75, 90]).round(3), "frac<0.3", np.mean(d < 0.3).round(3))
V = gg.transformar(np.asarray(MB.vertices), T)
MB.vertices = o3d.utility.Vector3dVector(V)
MB.vertex_normals = o3d.utility.Vector3dVector(gg.normais_vertices(V, np.asarray(MB.triangles)))
o3d.io.write_triangle_mesh(os.path.join(out, "B_alinhada.ply"), MB)
