"""Otimizacao CONJUNTA das poses de todos os segmentos (ICP de varias vistas, ponto-a-plano).

    python pipeline/alinhar_direto.py <pasta_dos_segmentos> --poses <segmentos.json|poses_iniciais.json> --saida <json>

Em vez de transformar cada par de segmentos numa medida de pose relativa (o ICP ponto-a-plano desliza ao longo
de superficies lisas, entao so a direcao normal e confiavel), minimiza a SOMA dos residuos normais
n.(q - y) de todos os pares sobrepostos ao mesmo tempo (Gauss-Newton, Tukey), com uma restricao fraca
entre segmentos consecutivos do mesmo scan (deriva pequena) e o no de referencia fixo.
Parametrizacao local: T <- T exp(xi), xi = (omega, v).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade
import alinhar_segmentos as S


def exp_se3(x):
    w, v = x[:3], x[3:]
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-12:
        R = np.eye(3) + K
    else:
        R = np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = v
    return T


def log_so3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def carregar(nos):
    P, N, arv = [], [], []
    for no in nos:
        c = S._normais(np.load(no["arquivo"]))
        P.append(np.asarray(c.points, np.float64))
        N.append(np.asarray(c.normals, np.float64))
        arv.append(cKDTree(P[-1]))
    return P, N, arv


def otimizar_direto(nos, T, pares, odo_idx, ref, iteracoes=14, amostra=12000, sigma_odo=0.5, alavanca=200.0, tol_ini=6.0, tol_fim=1.2, log=print):
    n = len(nos)
    P, N, arv = carregar(nos)
    rng = np.random.default_rng(0)
    tol_sched = np.interp(np.arange(iteracoes), [0, iteracoes - 1], [tol_ini, tol_fim])
    for it in range(iteracoes):
        tol = float(tol_sched[it])
        H = np.zeros((6 * n, 6 * n))
        b = np.zeros(6 * n)
        soma_r, soma_n = 0.0, 0
        for (i, j) in pares:
            Rr = np.linalg.inv(T[i]) @ T[j]
            sel = rng.choice(len(P[j]), min(amostra, len(P[j])), replace=False)
            Sj = P[j][sel]
            q = Sj @ Rr[:3, :3].T + Rr[:3, 3]
            d, idx = S.vizinho(arv[i], q, len(P[i]))
            m = d < tol
            if m.sum() < 200:
                continue
            q, Sj, idx = q[m], Sj[m], idx[m]
            nrm = N[i][idx]
            r = np.einsum("ij,ij->i", q - P[i][idx], nrm)
            c = 0.5                                              # Tukey (mm)
            w = np.where(np.abs(r) < c, (1 - (r / c) ** 2) ** 2, 0.0)
            mloc = nrm @ Rr[:3, :3]                              # R_ij^T n  (linhas)
            Jj = np.hstack([np.cross(Sj, mloc), mloc])
            Ji = -np.hstack([np.cross(q, nrm), nrm])
            ci, cj = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
            Wi = Ji * w[:, None]
            Wj = Jj * w[:, None]
            H[ci, ci] += Wi.T @ Ji
            H[cj, cj] += Wj.T @ Jj
            H[ci, cj] += Wi.T @ Jj
            H[cj, ci] += Wj.T @ Ji
            b[ci] += Wi.T @ r
            b[cj] += Wj.T @ r
            soma_r += float(np.sum(w * r * r))
            soma_n += int(w.sum() > 0) * len(r)
        # odometria: relativo entre consecutivos ~ identidade
        wo = 1.0 / (sigma_odo ** 2)
        esc = np.array([alavanca] * 3 + [1.0] * 3)                # rotacao em mm de alavanca
        for (s, t) in odo_idx:
            D = np.linalg.inv(T[s]) @ T[t]
            e = np.r_[log_so3(D[:3, :3]), D[:3, 3]] * esc
            cs, ct = slice(6 * s, 6 * s + 6), slice(6 * t, 6 * t + 6)
            Wm = np.diag(esc * esc) * wo / np.maximum(esc * esc, 1)   # peso por componente em mm^-2
            Wm = np.diag(np.ones(6) * wo)
            Je = np.diag(esc)
            A = Je.T @ Wm @ Je
            H[ct, ct] += A
            H[cs, cs] += A
            H[cs, ct] -= A
            H[ct, cs] -= A
            g = Je.T @ Wm @ e
            b[ct] += g
            b[cs] -= g
        # no de referencia fixo
        rs = slice(6 * ref, 6 * ref + 6)
        H[rs, :] = 0
        H[:, rs] = 0
        H[rs, rs] = np.eye(6)
        b[rs] = 0
        lam = 1e-3 * np.maximum(np.diag(H), 1e-9)
        delta = np.linalg.solve(H + np.diag(lam), -b)
        passo = 0.0
        for k in range(n):
            if k == ref:
                continue
            T[k] = T[k] @ exp_se3(delta[6 * k:6 * k + 6])
            passo = max(passo, float(np.linalg.norm(delta[6 * k + 3:6 * k + 6])))
        rms = float(np.sqrt(soma_r / max(soma_n, 1)))
        log(f"  iteracao {it+1}/{iteracoes}: tol {tol:.1f} mm | residuo normal RMS (robusto) {rms*1000:.0f} um | maior passo {passo:.3f} mm", flush=True)
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta", help="pasta com segmentos_indice.json e segmentos/ (de alinhar_segmentos.py)")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--referencia")
    ap.add_argument("--iteracoes", type=int, default=14)
    ap.add_argument("--sigma-odo", type=float, default=0.5)
    ap.add_argument("--k-por-no", type=int, default=5)
    ap.add_argument("--tol-ini", type=float, default=6.0)
    ap.add_argument("--tol-fim", type=float, default=1.2)
    ap.add_argument("--amostra", type=int, default=12000)
    a = ap.parse_args()
    baixa_prioridade()
    os.environ["OMP_NUM_THREADS"] = "4"
    ind = json.load(open(os.path.join(a.pasta, "segmentos_indice.json")))
    pj = json.load(open(a.poses))
    if "segmentos" in pj:                                         # saida de alinhar_segmentos.py
        T = {(g, s["seg"]): np.array(s["T"]) for g, v in pj["segmentos"].items() for s in v}
        ref_scan = pj["referencia"]
    else:                                                         # poses iniciais por scan
        T = {(g, s["seg"]): np.array(pj["poses_iniciais"][g]) for g, v in ind.items() if g in pj["poses_iniciais"] for s in v}
        ref_scan = pj["referencia"]
    nos = [s for g in sorted(ind) if (g, 0) in T for s in ind[g]]
    Tn = [T[(s["scan"], s["seg"])].copy() for s in nos]
    ref = next(k for k, s in enumerate(nos) if s["scan"] == (a.referencia or ref_scan) and s["seg"] == 0)
    odo_idx = [(k, k + 1) for k in range(len(nos) - 1) if nos[k]["scan"] == nos[k + 1]["scan"]]
    t0 = time.time()
    pares = S.candidatos(nos, Tn, k_por_no=a.k_por_no, sobreposicao_min=0.15, tol=8.0)
    print(f"{len(nos)} segmentos, {len(pares)} pares sobrepostos, {len(odo_idx)} elos de odometria ({time.time()-t0:.0f}s)", flush=True)
    Tn = otimizar_direto(nos, Tn, pares, odo_idx, ref, iteracoes=a.iteracoes, sigma_odo=a.sigma_odo,
                       tol_ini=a.tol_ini, tol_fim=a.tol_fim, amostra=a.amostra)
    saida = {}
    for k, s in enumerate(nos):
        saida.setdefault(s["scan"], []).append(dict(k0=s["k0"], k1=s["k1"], seg=s["seg"], T=Tn[k].tolist()))
    json.dump(dict(referencia=ref_scan, segmentos=saida), open(a.saida, "w"), indent=1)
    print("gravado", a.saida, f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
