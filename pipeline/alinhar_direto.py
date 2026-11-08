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


def tukey_c(tol, modo="auto"):
    """Corte do Tukey (mm). 'auto' = max(0,5; tol/4): com busca a 6 mm, c fixo de 0,5 mm zerava o peso
    de pares ainda longe (inicializacao imperfeita nao refinava). Numero = valor fixo (0.5 = regra antiga)."""
    if modo == "auto":
        return max(0.5, tol / 4.0)
    return float(modo)


def conectividade(n, pares, odo_idx, ref):
    """Nos alcancaveis a partir da referencia pelos pares + odometria. Devolve lista dos NAO alcancaveis."""
    viz = [[] for _ in range(n)]
    for i, j in list(pares) + list(odo_idx):
        viz[i].append(j)
        viz[j].append(i)
    visto, pilha = {ref}, [ref]
    while pilha:
        a = pilha.pop()
        for b in viz[a]:
            if b not in visto:
                visto.add(b)
                pilha.append(b)
    return [k for k in range(n) if k not in visto]


def _termos_pares(P, N, arv, T, pares, amostras, tol, c, jacobiano=True):
    """Custo robusto de Tukey sobre uma amostra FIXA de cada par (pontos sem correspondencia a < tol contam
    c^2/6, entao o custo e comparavel entre poses). Opcionalmente acumula H e b (Gauss-Newton)."""
    n = len(T)
    H = np.zeros((6 * n, 6 * n)) if jacobiano else None
    b = np.zeros(6 * n) if jacobiano else None
    custo, soma_wr2, soma_w, usados = 0.0, 0.0, 0.0, 0
    rho_max = c * c / 6.0
    for (i, j), sel in zip(pares, amostras):
        Rr = np.linalg.inv(T[i]) @ T[j]
        Sj = P[j][sel]
        q = Sj @ Rr[:3, :3].T + Rr[:3, 3]
        d, idx = S.vizinho(arv[i], q, len(P[i]))
        m = d < tol
        custo += rho_max * float((~m).sum())
        if m.sum() < 200:
            custo += rho_max * float(m.sum())
            continue
        q, Sj, idx = q[m], Sj[m], idx[m]
        nrm = N[i][idx]
        r = np.einsum("ij,ij->i", q - P[i][idx], nrm)
        dentro = np.abs(r) < c
        u = (r / c) ** 2
        custo += float(np.sum(np.where(dentro, rho_max * (1 - (1 - u) ** 3), rho_max)))
        w = np.where(dentro, (1 - u) ** 2, 0.0)
        soma_wr2 += float(np.sum(w * r * r))
        soma_w += float(w.sum())
        usados += 1
        if not jacobiano:
            continue
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
    return dict(custo=custo, H=H, b=b, rms=float(np.sqrt(soma_wr2 / soma_w)) if soma_w > 0 else float("nan"),
                n_efetivo=soma_w, pares_usados=usados)


def _termos_odometria(T, odo_idx, sigma_odo, alavanca, H=None, b=None):
    """Restricao fraca entre segmentos consecutivos (relativo ~ identidade). Devolve o custo quadratico."""
    wo = 1.0 / (sigma_odo ** 2)
    esc = np.array([alavanca] * 3 + [1.0] * 3)                # rotacao em mm de alavanca
    Je = np.diag(esc)
    A = Je.T @ (np.eye(6) * wo) @ Je
    custo = 0.0
    for (s, t) in odo_idx:
        D = np.linalg.inv(T[s]) @ T[t]
        e = np.r_[log_so3(D[:3, :3]), D[:3, 3]] * esc
        custo += 0.5 * wo * float(e @ e)
        if H is None:
            continue
        cs, ct = slice(6 * s, 6 * s + 6), slice(6 * t, 6 * t + 6)
        H[ct, ct] += A
        H[cs, cs] += A
        H[cs, ct] -= A
        H[ct, cs] -= A
        g = Je.T @ (np.eye(6) * wo) @ e
        b[ct] += g
        b[cs] -= g
    return custo


def otimizar_direto(nos, T, pares, odo_idx, ref, iteracoes=14, amostra=12000, sigma_odo=0.5, alavanca=200.0,
                    tol_ini=6.0, tol_fim=1.2, log=print, tukey="auto", tentativas=5):
    """Gauss-Newton amortecido (Levenberg-Marquardt) com ACEITACAO: um passo so e aplicado se o custo robusto
    (pares + odometria, mesma amostra) cair; senao o amortecimento aumenta e o passo e refeito (ate
    `tentativas`). Passo sem melhora e rejeitado (pose anterior mantida). Devolve (T, historico)."""
    n = len(nos)
    P, N, arv = carregar(nos)
    rng = np.random.default_rng(0)
    tol_sched = np.interp(np.arange(iteracoes), [0, iteracoes - 1], [tol_ini, tol_fim])
    hist = []
    mu = 1e-3
    for it in range(iteracoes):
        tol = float(tol_sched[it])
        c = tukey_c(tol, tukey)
        amostras = [rng.choice(len(P[j]), min(amostra, len(P[j])), replace=False) for (_, j) in pares]
        t0 = _termos_pares(P, N, arv, T, pares, amostras, tol, c)
        H, b = t0["H"], t0["b"]
        custo0 = t0["custo"] + _termos_odometria(T, odo_idx, sigma_odo, alavanca, H, b)
        rs = slice(6 * ref, 6 * ref + 6)
        H[rs, :] = 0
        H[:, rs] = 0
        H[rs, rs] = np.eye(6)
        b[rs] = 0
        aceito, custo1, passo = False, custo0, 0.0
        for _ in range(tentativas):
            lam = mu * np.maximum(np.diag(H), 1e-9)
            delta = np.linalg.solve(H + np.diag(lam), -b)
            Tn = [T[k] if k == ref else T[k] @ exp_se3(delta[6 * k:6 * k + 6]) for k in range(n)]
            t1 = _termos_pares(P, N, arv, Tn, pares, amostras, tol, c, jacobiano=False)
            custo1 = t1["custo"] + _termos_odometria(Tn, odo_idx, sigma_odo, alavanca)
            if custo1 <= custo0:
                aceito = True
                passo = max((float(np.linalg.norm(delta[6 * k + 3:6 * k + 6])) for k in range(n) if k != ref), default=0.0)
                T = Tn
                mu = max(mu / 3.0, 1e-6)
                break
            mu = min(mu * 10.0, 1e4)     # teto: rejeicoes seguidas nao podem congelar as iteracoes seguintes
        hist.append(dict(iteracao=it + 1, tol_mm=tol, tukey_mm=c, custo_antes=custo0, custo_depois=custo1 if aceito else custo0,
                         aceito=aceito, rms_ponderado_um=t0["rms"] * 1000, n_efetivo=t0["n_efetivo"],
                         pares_usados=t0["pares_usados"], maior_passo_mm=passo))
        log(f"  iteracao {it+1}/{iteracoes}: tol {tol:.1f} mm, Tukey {c:.2f} mm | RMS ponderado {t0['rms']*1000:.0f} um "
            f"(n efetivo {t0['n_efetivo']:.0f}) | custo {custo0:.1f} -> {custo1 if aceito else custo0:.1f} "
            f"{'ACEITO' if aceito else 'REJEITADO (pose mantida)'} | maior passo {passo:.3f} mm", flush=True)
    return T, hist


def condicionamento(nos, T, pares, ref, amostra=4000, tol=1.2):
    """Razao entre o menor e o maior autovalor do bloco 6x6 de cada no (rotacao em mm de alavanca 100):
    perto de 0 = direcao pouco observada (ex.: deslizar ao longo de superficie lisa)."""
    P, N, arv = carregar(nos)
    rng = np.random.default_rng(1)
    amostras = [rng.choice(len(P[j]), min(amostra, len(P[j])), replace=False) for (_, j) in pares]
    t = _termos_pares(P, N, arv, T, pares, amostras, tol, tukey_c(tol))
    D = np.diag([1 / 100.0] * 3 + [1.0] * 3)
    out = {}
    for k in range(len(nos)):
        if k == ref:
            continue
        B = D @ t["H"][6 * k:6 * k + 6, 6 * k:6 * k + 6] @ D
        ev = np.linalg.eigvalsh(B)
        out[k] = float(ev[0] / ev[-1]) if ev[-1] > 0 else 0.0
    return out


def vies_reservado(nos, Ts, pares, n_max=60, seed=7):
    """Vies normal com sinal por celula de 30 mm (mediana dos pares) em pares sorteados; mesma amostra
    para todas as poses de `Ts`. Devolve lista de medianas (um) por conjunto de poses."""
    rng = np.random.default_rng(seed)
    sel = [pares[k] for k in rng.choice(len(pares), min(n_max, len(pares)), replace=False)] if pares else []
    cache = {}

    def nuv(k):
        if k not in cache:
            cache[k] = S._normais(np.load(nos[k]["arquivo"]))
        return cache[k]
    res = []
    for T in Ts:
        v = []
        for (i, j) in sel:
            r = S.residuo_plano(nuv(i), nuv(j), np.linalg.inv(T[i]) @ T[j])
            if r[1] is not None:
                v.append(r[1] * 1000)
        res.append(float(np.median(v)) if v else None)
    return res, len(sel)


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
    ap.add_argument("--tukey", default="auto", help="corte robusto em mm, ou auto = max(0,5; tol/4)")
    ap.add_argument("--sem-reversao", action="store_true",
                    help="grava o resultado mesmo se o vies por celula (pares reservados) piorar")
    a = ap.parse_args()
    baixa_prioridade()
    from recursos import orcamento_threads
    orcamento_threads(1, teto=4)
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
    soltos = conectividade(len(nos), pares, odo_idx, ref)
    if soltos:
        print(f"AVISO: {len(soltos)} segmento(s) sem ligacao com a referencia (ficam com a pose de entrada): "
              + ", ".join(f"{nos[k]['scan']}#{nos[k]['seg']}" for k in soltos[:12]), flush=True)
    # pares reservados para a avaliacao: metade sorteada fica FORA da otimizacao
    rng = np.random.default_rng(11)
    ordem = rng.permutation(len(pares))
    reserva = [pares[k] for k in ordem[: len(pares) // 5]] if len(pares) >= 20 else list(pares)
    uso = [pares[k] for k in ordem[len(pares) // 5:]] if len(pares) >= 20 else list(pares)
    if conectividade(len(nos), uso, odo_idx, ref) != soltos:     # a reserva nao pode desligar o grafo
        uso, reserva = list(pares), list(pares)
    T_ini = [x.copy() for x in Tn]
    Tn, hist = otimizar_direto(nos, Tn, uso, odo_idx, ref, iteracoes=a.iteracoes, sigma_odo=a.sigma_odo,
                               tol_ini=a.tol_ini, tol_fim=a.tol_fim, amostra=a.amostra, tukey=a.tukey)
    (v_antes, v_depois), n_res = vies_reservado(nos, [T_ini, Tn], reserva)
    print(f"vies por celula de 30 mm em {n_res} pares reservados: {v_antes} -> {v_depois} um (mediana)", flush=True)
    revertido = False
    if v_antes is not None and v_depois is not None and v_depois > v_antes and not a.sem_reversao:
        print("PIOROU nos pares reservados: poses de entrada mantidas (use --sem-reversao para gravar mesmo assim)", flush=True)
        Tn, revertido = T_ini, True
    cond = condicionamento(nos, Tn, pares, ref) if pares else {}
    fracos = sorted(cond, key=cond.get)[:5]
    if fracos:
        print("direcoes menos observadas (razao autovalores; ~0 = desliza): "
              + ", ".join(f"{nos[k]['scan'][-6:]}#{nos[k]['seg']} {cond[k]:.1e}" for k in fracos), flush=True)
    saida = {}
    for k, s in enumerate(nos):
        saida.setdefault(s["scan"], []).append(dict(k0=s["k0"], k1=s["k1"], seg=s["seg"], T=Tn[k].tolist()))
    json.dump(dict(referencia=ref_scan, segmentos=saida,
                   avaliacao=dict(vies_reservado_antes_um=v_antes, vies_reservado_depois_um=v_depois, pares_reservados=n_res,
                                  revertido=revertido, segmentos_sem_ligacao=[f"{nos[k]['scan']}#{nos[k]['seg']}" for k in soltos],
                                  condicionamento={f"{nos[k]['scan']}#{nos[k]['seg']}": v for k, v in cond.items()},
                                  historico=hist)),
              open(a.saida, "w"), indent=1)
    print("gravado", a.saida, f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
