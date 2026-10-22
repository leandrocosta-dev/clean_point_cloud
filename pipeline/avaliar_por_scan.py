"""Viés normal por scan: para cada scan, ate 8 pares (segmento de outro scan com maior sobreposicao).

    python pipeline/avaliar_por_scan.py <pasta_segmentos> <poses.json> [saida.json]
Um scan cujo viés mediano e muito maior que o dos outros (ou sem pares) nao encaixa na peca.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alinhar_segmentos as S
from alinhar_varios import baixa_prioridade

baixa_prioridade()
os.environ["OMP_NUM_THREADS"] = "4"
pasta, arq = sys.argv[1], sys.argv[2]
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
pj = json.load(open(arq))
T = {(g, s["seg"]): np.array(s["T"]) for g, v in pj["segmentos"].items() for s in v}
nos = [s for g in sorted(ind) if (g, 0) in T for s in ind[g]]
Tn = [T[(s["scan"], s["seg"])] for s in nos]
pares = S.candidatos(nos, Tn, k_por_no=8, sobreposicao_min=0.15, tol=8.0)
cache = {}
def nuvem(n):
    if n not in cache: cache[n] = S._normais(np.load(nos[n]["arquivo"]))
    return cache[n]
por = {}
for (i, j) in pares:
    si, sj = nos[i]["scan"], nos[j]["scan"]
    if si == sj: continue
    r = S.residuo_plano(nuvem(i), nuvem(j), np.linalg.inv(Tn[i]) @ Tn[j])
    if r[1] is None: continue
    for g, o in ((si, sj), (sj, si)):
        por.setdefault(g, []).append((r[1] * 1000, r[0], o))
saida = {}
print(f"{'scan':16s} {'pares':>5s} {'vies med (um)':>14s} {'p90':>6s}  piores parceiros")
for g in sorted(ind):
    v = por.get(g, [])
    if not v:
        print(f"{g:16s} {0:5d}    SEM PARES"); saida[g] = None; continue
    b = np.array([x[0] for x in v]); piores = sorted(v, reverse=True)[:2]
    saida[g] = dict(pares=len(v), mediana_um=float(np.median(b)), p90_um=float(np.percentile(b, 90)))
    print(f"{g:16s} {len(v):5d} {np.median(b):14.0f} {np.percentile(b,90):6.0f}  " + ", ".join(f"{x[2][-6:]}:{x[0]:.0f}" for x in piores), flush=True)
if len(sys.argv) > 3: json.dump(saida, open(sys.argv[3], "w"), indent=1)
