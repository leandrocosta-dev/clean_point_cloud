"""Alinhamento global de duas nuvens (FPFH + RANSAC, varias sementes) + ICP em tolerancias decrescentes.

Uso: python pipeline/revo_antigo/alinhar_global.py alvo.npy movel.npy --saida T.npy [--voxel 3 --sementes 8]
Imprime, por semente, a sobreposicao (% do movel a < 1 mm do alvo e inverso). Grava a melhor T (movel -> alvo).
"""

import argparse

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d


def pc(arq, v):
    """(P, N) cupy: nuvem decimada em voxel v (GPU) com normais PCA (30 vizinhos a <= 4 voxels), no sinal "para fora"
    (do centroide); o .npy nao traz normais."""
    P = np.load(arq).astype(np.float64)
    C, _ = gpu.reduzir_voxel(P, v)
    N = gg.alinhar_sinal(gg.normais_hibrido(C, v * 4, 30, saida_gpu=True), C - C.mean(0), saida_gpu=True)
    return C, N


def _o3d(P, N):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(P)))
    p.normals = o3d.utility.Vector3dVector(gpu.para_cpu(N))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("alvo")
    ap.add_argument("movel")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=3.0)
    ap.add_argument("--sementes", type=int, default=8)
    a = ap.parse_args()
    A, B = _o3d(*pc(a.alvo, a.voxel)), _o3d(*pc(a.movel, a.voxel))
    A1, NA1 = pc(a.alvo, 1.0)
    B1, _ = pc(a.movel, 1.0)
    # FPFH e RANSAC por features: Open3D na CPU (sem equivalente na GPU); o resto (ICP, sobreposicao) na GPU
    fp = lambda x: o3d.pipelines.registration.compute_fpfh_feature(
        x, o3d.geometry.KDTreeSearchParamHybrid(radius=a.voxel * 5, max_nn=100)
    )
    fA, fB = fp(A), fp(B)
    res = []
    for sd in range(a.sementes):
        o3d.utility.random.seed(sd)
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            B,
            A,
            fB,
            fA,
            True,
            a.voxel * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(a.voxel * 1.5),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(2_000_000, 0.999),
        )
        T, _ = gg.icp_encadeado(B1, A1, NA1, r.transformation, (3.0, 1.5, 0.8), iteracoes=50)
        Bt = gg.transformar(B1, T, saida_gpu=True)
        f1 = gg.fracao_ate(A1, Bt, 1.0)
        f2 = gg.fracao_ate(Bt, A1, 1.0)
        print(
            f"semente {sd}: ransac fit {r.fitness:.2f} | movel->alvo<1mm {100 * f1:.1f}% alvo->movel {100 * f2:.1f}%",
            flush=True,
        )
        res.append((f1 + f2, T))
    res.sort(key=lambda x: -x[0])
    np.save(a.saida, res[0][1])
    print("melhor soma", res[0][0])


if __name__ == "__main__":
    main()
