"""Alinhamento global de duas nuvens (FPFH + RANSAC, varias sementes) + ICP em tolerancias decrescentes.

Uso: python pipeline/revo_antigo/alinhar_global.py alvo.npy movel.npy --saida T.npy [--voxel 3 --sementes 8]
Imprime, por semente, a sobreposicao (% do movel a < 1 mm do alvo e inverso). Grava a melhor T (movel -> alvo).
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

def pc(arq, v):
    P = np.load(arq).astype(np.float64); p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(v)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v * 4, max_nn=30)); return p

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("alvo"); ap.add_argument("movel"); ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=3.0); ap.add_argument("--sementes", type=int, default=8); a = ap.parse_args()
    A, B = pc(a.alvo, a.voxel), pc(a.movel, a.voxel)
    A1, B1 = pc(a.alvo, 1.0), pc(a.movel, 1.0)
    fp = lambda x: o3d.pipelines.registration.compute_fpfh_feature(x, o3d.geometry.KDTreeSearchParamHybrid(radius=a.voxel * 5, max_nn=100))
    fA, fB = fp(A), fp(B); tA = cKDTree(np.asarray(A1.points)); res = []
    for sd in range(a.sementes):
        o3d.utility.random.seed(sd)
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(B, A, fB, fA, True, a.voxel * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(a.voxel * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(2_000_000, 0.999))
        T = r.transformation
        for tol in (3.0, 1.5, 0.8):
            T = o3d.pipelines.registration.registration_icp(B1, A1, tol, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)).transformation
        Bt = np.asarray(B1.points) @ T[:3, :3].T + T[:3, 3]
        f1 = float((tA.query(Bt)[0] < 1.0).mean()); f2 = float((cKDTree(Bt).query(np.asarray(A1.points))[0] < 1.0).mean())
        print(f"semente {sd}: ransac fit {r.fitness:.2f} | movel->alvo<1mm {100*f1:.1f}% alvo->movel {100*f2:.1f}%", flush=True); res.append((f1 + f2, T))
    res.sort(key=lambda x: -x[0]); np.save(a.saida, res[0][1]); print("melhor soma", res[0][0])

if __name__ == "__main__":
    main()
