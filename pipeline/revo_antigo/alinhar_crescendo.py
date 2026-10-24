"""Alinha N malhas parciais da mesma peca por crescimento: a de maior area e a referencia; cada passo registra a melhor das restantes
(FPFH+RANSAC com varias sementes + ICP ponto-a-plano 3 -> 1,5 -> 0,6 mm) contra a UNIAO ja montada.

Uso: python pipeline/revo_antigo/alinhar_crescendo.py m1.ply m2.ply ... --saida poses.json [--aceita 0.6]
Sobreposicao = fracao dos pontos do movel a < 1 mm da uniao (e o inverso, medido so na regiao de sobreposicao do movel).
Aceita o candidato se o movel->uniao >= --aceita E o resultado repete em >= 2 sementes (mesma T a < 0,5 mm / 0,5 grau).
"""
import argparse
import json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def amostra(m, n=250000):
    p = m.sample_points_uniformly(n); p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=30)); return p


def reg(mov, ref, sementes=6, voxel=3.0):
    fp = lambda x: o3d.pipelines.registration.compute_fpfh_feature(x, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    M, R = mov.voxel_down_sample(voxel), ref.voxel_down_sample(voxel)
    M.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30)); R.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    fM, fR = fp(M), fp(R)
    M1, R1 = mov.voxel_down_sample(1.0), ref.voxel_down_sample(1.0); tR = cKDTree(np.asarray(R1.points)); out = []
    for sd in range(sementes):
        o3d.utility.random.seed(sd)
        r = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(M, R, fM, fR, True, voxel * 1.5,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1_000_000, 0.999))
        T = r.transformation
        for tol in (3.0, 1.5, 0.6):
            T = o3d.pipelines.registration.registration_icp(M1, R1, tol, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)).transformation
        Mt = np.asarray(M1.points) @ T[:3, :3].T + T[:3, 3]; f = float((tR.query(Mt)[0] < 1.0).mean()); out.append((f, T))
    out.sort(key=lambda x: -x[0]); return out


def igual(T1, T2):
    d = np.linalg.inv(T1) @ T2; ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(d[:3, 3]) < 0.5 and ang < 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("malhas", nargs="+"); ap.add_argument("--saida", required=True); ap.add_argument("--aceita", type=float, default=0.6)
    a = ap.parse_args()
    ms = [o3d.io.read_triangle_mesh(m) for m in a.malhas]; pcs = [amostra(m) for m in ms]
    areas = [m.get_surface_area() for m in ms]; base = int(np.argmax(areas)); print("base", a.malhas[base], "area", round(areas[base]))
    poses = {base: np.eye(4)}; uniao = o3d.geometry.PointCloud(pcs[base]); pend = [i for i in range(len(ms)) if i != base]; falhas = {}
    while pend:
        melhor = None
        for i in pend:
            out = reg(pcs[i], uniao); f, T = out[0]; rep = sum(1 for g, U in out if igual(T, U)); print(f"  {a.malhas[i].split('/')[-1]}: melhor {100*f:.1f}% (repete em {rep}/{len(out)} sementes)", flush=True)
            if f >= a.aceita and rep >= 2 and (melhor is None or f > melhor[0]): melhor = (f, i, T)
        if melhor is None:
            print("nenhum candidato restante atinge o criterio:", [a.malhas[i] for i in pend]); break
        f, i, T = melhor; poses[i] = T; p = o3d.geometry.PointCloud(pcs[i]); p.transform(T); uniao += p; pend.remove(i); print(f"-> aceito {a.malhas[i]} ({100*f:.1f}%)", flush=True)
    json.dump({a.malhas[i]: poses[i].tolist() for i in poses}, open(a.saida, "w"))
    print("alinhados:", len(poses), "de", len(ms))


if __name__ == "__main__":
    main()
