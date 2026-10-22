"""Mede o alinhamento de poses por segmento: vies normal por celula de 30 mm em pares de segmentos sobrepostos.

    python pipeline/avaliar_segmentos.py <pasta_segmentos> <poses.json|segmentos.json> [outro.json ...]
Compara varios arquivos de poses na MESMA amostra de pares (os candidatos sao calculados com o 1o arquivo).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alinhar_segmentos as S
from alinhar_varios import baixa_prioridade

baixa_prioridade()
os.environ["OMP_NUM_THREADS"] = "4"
pasta, arquivos = sys.argv[1], sys.argv[2:]
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))


def poses(arq):
    pj = json.load(open(arq))
    if "segmentos" in pj:
        return {(g, s["seg"]): np.array(s["T"]) for g, v in pj["segmentos"].items() for s in v}
    return {(g, s["seg"]): np.array(pj["poses_iniciais"][g]) for g, v in ind.items() if g in pj["poses_iniciais"] for s in v}


P = [poses(a) for a in arquivos]
nos = [s for g in sorted(ind) if (g, 0) in P[0] for s in ind[g]]
Ts = [[p[(s["scan"], s["seg"])] for s in nos] for p in P]
pares = S.candidatos(nos, Ts[0], k_por_no=4, sobreposicao_min=0.15, tol=8.0)
rng = np.random.default_rng(1)
pares = [pares[k] for k in rng.choice(len(pares), min(120, len(pares)), replace=False)]
cache = {}
def nuvem(n):
    if n not in cache: cache[n] = S._normais(np.load(nos[n]["arquivo"]))
    return cache[n]
res = [[] for _ in arquivos]
for (i, j) in pares:
    for a, T in enumerate(Ts):
        r = S.residuo_plano(nuvem(i), nuvem(j), np.linalg.inv(T[i]) @ T[j])
        if r[1] is not None:
            res[a].append((r[1] * 1000, r[0]))
for a, arq in enumerate(arquivos):
    v = np.array([x[0] for x in res[a]])
    print(f"{os.path.basename(os.path.dirname(arq)) + '/' + os.path.basename(arq):42s} {len(v):3d} pares | vies normal/celula: mediana {np.median(v):4.0f} um, p75 {np.percentile(v,75):4.0f}, p90 {np.percentile(v,90):4.0f}, max {v.max():4.0f}", flush=True)
