"""Alinha o scan B ao scan A (peca prismatica/cilindrica vista em duas poses) por candidatos.

Uso: python pipeline/revo_antigo/unir_dois.py nA.ply nB.ply --saida T.json
Candidatos: eixo longo (PCA) de B alinhado ao de A, com/sem inversao, rotacao em torno do eixo de 0 a 355 graus
(passo 5); cada um refinado por ICP ponto-a-plano (3 mm -> 1 mm -> 0,5 mm). Ranking por fracao de pontos de B a < 0,5 mm
de A + mediana. Imprime os melhores e a diferenca entre o 1o e o melhor candidato de OUTRA familia (inversao).
"""
import argparse
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

os.environ.setdefault("OMP_NUM_THREADS", "4")


def eixos(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return c, vt  # linhas = eixos (comprimento, e dois transversais)


def rot(eixo, ang):
    e = eixo / np.linalg.norm(eixo)
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def avaliar(A, tA, B, T, d=0.5):
    Bt = B @ T[:3, :3].T + T[:3, 3]
    dist, _ = tA.query(Bt)
    dist2, _ = cKDTree(Bt).query(A)
    return float((dist < d).mean()), float(np.median(dist)), float((dist2 < d).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--saida", required=True)
    ap.add_argument("--passo-ang", type=float, default=5)
    a = ap.parse_args()
    pa = o3d.io.read_point_cloud(a.a).voxel_down_sample(0.6)
    pb = o3d.io.read_point_cloud(a.b).voxel_down_sample(0.6)
    A, B = np.asarray(pa.points), np.asarray(pb.points)
    tA = cKDTree(A)
    cA, vA = eixos(A); cB, vB = eixos(B)
    res = []
    for inv in (0, 1):
        # R0 leva eixo longo de B ao de A (com ou sem inversao), preservando mao
        Rb = vB.copy()
        Ra = vA.copy()
        if inv:
            Rb = Rb * np.array([[-1], [-1], [1]])  # inverte longo e um transversal (rotacao propria)
        R0 = Ra.T @ Rb
        for ang in np.arange(0, 360, a.passo_ang):
            R = rot(vA[0], np.radians(ang)) @ R0
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cA - R @ cB
            pbt = o3d.geometry.PointCloud(pb); pbt.transform(T)
            for tol in (3.0, 1.0, 0.5):
                r = o3d.pipelines.registration.registration_icp(
                    pbt, pa, tol, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
                pbt.transform(r.transformation); T = r.transformation @ T
            fb, med, fa = avaliar(A, tA, B, T)
            res.append(dict(inv=inv, ang=float(ang), fb=fb, fa=fa, med=med, T=T.tolist()))
    res.sort(key=lambda r: -(r["fb"] + r["fa"]))
    for r in res[:8]:
        print(f"inv={r['inv']} ang={r['ang']:5.0f}  B->A<0,5mm {100*r['fb']:5.1f}%  A->B {100*r['fa']:5.1f}%  mediana {r['med']:.3f}")
    outra = [r for r in res if r["inv"] != res[0]["inv"]]
    print("melhor da outra inversao:", f"{100*outra[0]['fb']:.1f}% / {100*outra[0]['fa']:.1f}%")
    json.dump(res[:20], open(a.saida, "w"))


if __name__ == "__main__":
    main()
