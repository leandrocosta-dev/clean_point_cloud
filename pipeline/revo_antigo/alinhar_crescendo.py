"""Alinha N malhas parciais da mesma peca por crescimento: a de maior area e a referencia; cada passo registra a melhor das restantes
(FPFH+RANSAC com varias sementes + ICP ponto-a-plano 3 -> 1,5 -> 0,6 mm) contra a UNIAO ja montada.

Uso: python pipeline/revo_antigo/alinhar_crescendo.py m1.ply m2.ply ... --saida poses.json [--aceita 0.6]
Sobreposicao = fracao dos pontos do movel a < 1 mm da uniao (e o inverso, medido so na regiao de sobreposicao do movel).
Aceita o candidato se o movel->uniao >= --aceita E o resultado repete em >= 2 sementes (mesma T a < 0,5 mm / 0,5 grau).
"""

import argparse
import json

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

reg_o3d = o3d.pipelines.registration


def amostra(m, n=250000):
    """(P, N) numpy: n pontos uniformes na malha (GPU); normais = PCA de 30 vizinhos a <= 3 mm, no sinal da malha."""
    V, F = np.asarray(m.vertices), np.asarray(m.triangles)
    P, Nt = gg.amostrar_malha(V, F, n, saida_gpu=True)
    N = gg.alinhar_sinal(gg.normais_hibrido(P, 3.0, 30, saida_gpu=True), Nt, saida_gpu=True)
    return gpu.para_cpu(P), gpu.para_cpu(N)


def _o3d(P, N):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    return pc


def _voxel(P, N, v, raio_normais=None):
    """Decima (GPU); com `raio_normais` re-estima as normais (PCA, 30 vizinhos), no sinal das normais medias."""
    C, Nm = gg.reduzir_voxel_normais(P, N, v, saida_gpu=True)
    if raio_normais:
        Nm = gg.alinhar_sinal(gg.normais_hibrido(C, raio_normais, 30, saida_gpu=True), Nm, saida_gpu=True)
    return C, Nm


def reg(mov, ref, sementes=6, voxel=3.0):
    """mov, ref = (P, N) numpy. FPFH + RANSAC por features (Open3D, CPU: sem equivalente na GPU); decimacao, normais, ICP
    ponto-plano (3 -> 1,5 -> 0,6 mm) e fracoes de sobreposicao na GPU."""
    fp = lambda x: reg_o3d.compute_fpfh_feature(x, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    M = _o3d(*(gpu.para_cpu(x) for x in _voxel(*mov, voxel, voxel * 4)))
    R = _o3d(*(gpu.para_cpu(x) for x in _voxel(*ref, voxel, voxel * 4)))
    fM, fR = fp(M), fp(R)
    M1, _ = _voxel(*mov, 1.0)
    R1, NR1 = _voxel(*ref, 1.0)
    out = []
    for sd in range(sementes):
        o3d.utility.random.seed(sd)
        r = reg_o3d.registration_ransac_based_on_feature_matching(
            M,
            R,
            fM,
            fR,
            True,
            voxel * 1.5,
            reg_o3d.TransformationEstimationPointToPoint(False),
            3,
            [
                reg_o3d.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                reg_o3d.CorrespondenceCheckerBasedOnDistance(voxel * 1.5),
            ],
            reg_o3d.RANSACConvergenceCriteria(1_000_000, 0.999),
        )
        T, _ = gg.icp_encadeado(M1, R1, NR1, r.transformation, (3.0, 1.5, 0.6), iteracoes=50)
        f = gg.fracao_ate(R1, gg.transformar(M1, T, saida_gpu=True), 1.0)
        out.append((f, T))
    out.sort(key=lambda x: -x[0])
    return out


def igual(T1, T2):
    d = np.linalg.inv(T1) @ T2
    ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(d[:3, 3]) < 0.5 and ang < 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("malhas", nargs="+")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--aceita", type=float, default=0.6)
    a = ap.parse_args()
    ms = [o3d.io.read_triangle_mesh(m) for m in a.malhas]
    pcs = [amostra(m) for m in ms]
    areas = [gg.area_malha(np.asarray(m.vertices), np.asarray(m.triangles)) for m in ms]
    base = int(np.argmax(areas))
    print("base", a.malhas[base], "area", round(areas[base]))
    poses = {base: np.eye(4)}
    uP, uN = [pcs[base][0]], [pcs[base][1]]
    pend = [i for i in range(len(ms)) if i != base]
    while pend:
        melhor = None
        uniao = (np.concatenate(uP), np.concatenate(uN))
        for i in pend:
            out = reg(pcs[i], uniao)
            f, T = out[0]
            rep = sum(1 for g, U in out if igual(T, U))
            print(
                f"  {a.malhas[i].split('/')[-1]}: melhor {100 * f:.1f}% (repete em {rep}/{len(out)} sementes)",
                flush=True,
            )
            if f >= a.aceita and rep >= 2 and (melhor is None or f > melhor[0]):
                melhor = (f, i, T)
        if melhor is None:
            print("nenhum candidato restante atinge o criterio:", [a.malhas[i] for i in pend])
            break
        f, i, T = melhor
        poses[i] = T
        uP.append(gpu.para_cpu(gg.transformar(pcs[i][0], T)))
        uN.append(pcs[i][1] @ T[:3, :3].T)
        pend.remove(i)
        print(f"-> aceito {a.malhas[i]} ({100 * f:.1f}%)", flush=True)
    json.dump({a.malhas[i]: poses[i].tolist() for i in poses}, open(a.saida, "w"))
    print("alinhados:", len(poses), "de", len(ms))


if __name__ == "__main__":
    main()
