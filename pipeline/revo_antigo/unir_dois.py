"""Alinha o scan B ao scan A (peca prismatica/cilindrica vista em duas poses) por candidatos.

Uso: python pipeline/revo_antigo/unir_dois.py nA.ply nB.ply --saida T.json
Candidatos: eixo longo (PCA) de B alinhado ao de A, com/sem inversao, rotacao em torno do eixo de 0 a 355 graus
(passo 5); cada um refinado por ICP ponto-a-plano (3 mm -> 1 mm -> 0,5 mm). Ranking por fracao de pontos de B a < 0,5 mm
de A + mediana. Imprime os melhores e a diferenca entre o 1o e o melhor candidato de OUTRA familia (inversao).
"""

import argparse
import json
import os

import geom_gpu as gg
import gpu
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "4")


def eixos(P):
    """Centro e eixos principais (linhas). Covariancia na GPU; SVD da matriz 3x3 na CPU (trivial)."""
    Pg = gpu.para_gpu(P, np.float64)
    c = Pg.mean(0)
    Y = Pg - c
    cov = gpu.para_cpu(Y.T @ Y)
    c = gpu.para_cpu(c)
    _, _, vt = np.linalg.svd(cov)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    return c, vt  # linhas = eixos (comprimento, e dois transversais)


def rot(eixo, ang):
    e = eixo / np.linalg.norm(eixo)
    K = np.array([[0, -e[2], e[1]], [e[2], 0, -e[0]], [-e[1], e[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def avaliar(A, B, T, d=0.5):
    """(fracao de B a < d de A, mediana da distancia B->A, fracao de A a < d de B); A e B cupy, tudo na GPU."""
    Bt = gg.transformar(B, T, saida_gpu=True)
    dist = gg.distancia_ate(A, Bt, saida_gpu=True)
    return float((dist < d).mean()), float(gpu.xp().median(dist)), gg.fracao_ate(Bt, A, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--passo-ang", type=float, default=5)
    a = ap.parse_args()
    A, NA = gg.ler_nuvem(a.a, voxel=0.6)
    B, _ = gg.ler_nuvem(a.b, voxel=0.6)
    A, NA, B = (gpu.para_gpu(x, np.float64) for x in (A, NA, B))
    cA, vA = eixos(A)
    cB, vB = eixos(B)
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
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cA - R @ cB
            T, _ = gg.icp_encadeado(B, A, NA, T, (3.0, 1.0, 0.5), iteracoes=40)  # ICP ponto-plano na GPU
            fb, med, fa = avaliar(A, B, T)
            res.append(dict(inv=inv, ang=float(ang), fb=fb, fa=fa, med=med, T=T.tolist()))
    res.sort(key=lambda r: -(r["fb"] + r["fa"]))
    for r in res[:8]:
        print(
            f"inv={r['inv']} ang={r['ang']:5.0f}  B->A<0,5mm {100 * r['fb']:5.1f}%  A->B {100 * r['fa']:5.1f}%  mediana {r['med']:.3f}"
        )
    outra = [r for r in res if r["inv"] != res[0]["inv"]]
    print("melhor da outra inversao:", f"{100 * outra[0]['fb']:.1f}% / {100 * outra[0]['fa']:.1f}%")
    json.dump(res[:20], open(a.saida, "w"))


if __name__ == "__main__":
    main()
