"""Alinha o scan B (outro lado) na nuvem limpa do scan A (Creality Otter). Uso: alinhar_lados.py <A_limpa.ply> <B_pc_before.ply> <saida_dir>

Decimacao, normais, DBSCAN, filtro estatistico, ICP ponto-plano e sobreposicao na GPU; FPFH + RANSAC por features em Open3D (CPU)."""

import os
import sys

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

A_path, B_path, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
A = o3d.io.read_point_cloud(A_path)
B = o3d.io.read_point_cloud(B_path)
B.points = o3d.utility.Vector3dVector(gpu.para_cpu(gpu.para_gpu(np.asarray(B.points), np.float64) * 1000.0))  # m -> mm
PB = np.asarray(B.points)
sp = float(
    np.median(gpu.para_cpu(gpu.knn(PB, PB, 2)[0][:, 1]))
)  # espacamento mediano: distancia ao vizinho mais proximo (GPU)
B = B.select_by_index(np.where(gg.dbscan_maior(PB, max(1.0, sp * 4), 10))[0])
B = B.select_by_index(np.where(gpu.remover_soltos(np.asarray(B.points), k=29, desvios=3.0))[0])
print("B principal", len(B.points), "A", len(A.points))
o3d.io.write_point_cloud(os.path.join(out, "B_principal.ply"), B)
vox = 1.5


def prep(p):
    """(P, N) cupy decimados (GPU) com normais PCA de 40 vizinhos a <= 2,5 voxels, no sinal das normais do arquivo; + FPFH (CPU)."""
    P = np.asarray(p.points)
    N0 = np.asarray(p.normals) if p.has_normals() else None
    if N0 is None:
        C, _ = gpu.reduzir_voxel(
            P,
            vox,
        )
        N0 = C - C.mean(0)  # sem normais: sinal "para fora"
    else:
        C, N0 = gg.reduzir_voxel_normais(P, N0, vox, saida_gpu=True)
    N = gg.alinhar_sinal(gg.normais_hibrido(C, vox * 2.5, 40, saida_gpu=True), N0, saida_gpu=True)
    q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(C)))
    q.normals = o3d.utility.Vector3dVector(gpu.para_cpu(N))
    f = o3d.pipelines.registration.compute_fpfh_feature(
        q, o3d.geometry.KDTreeSearchParamHybrid(radius=vox * 6, max_nn=120)
    )
    return q, f, C, N


qa, fa, CA, NA = prep(A)
qb, fb, CB, NB = prep(B)
best = None
for seed in range(8):
    o3d.utility.random.seed(seed)
    r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        qb,
        qa,
        fb,
        fa,
        True,
        vox * 2.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vox * 2.0),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),
    )
    T, info = gpu.icp_ponto_plano(
        CB, CA, NA, T0=r.transformation, dist_max=vox * 1.5, iteracoes=30, perda="l2"
    )  # ICP ponto-plano na GPU
    print(seed, "ransac", round(r.fitness, 3), "icp fit", round(info["fitness"], 3), "rmse", round(info["rmse"], 3))
    if best is None or info["fitness"] > best[0]:
        best = (info["fitness"], seed, T)
np.save(os.path.join(out, "T_ransac_best.npy"), best[2])
print("melhor", best[1], best[0])
