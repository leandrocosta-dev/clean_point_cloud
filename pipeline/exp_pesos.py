"""Reotimiza o grafo de segmentos com varios pesos de odometria e mede o vies normal por celula nos pares."""
import sys, os, json, pickle, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alinhar_segmentos as S
from alinhar_varios import baixa_prioridade

work = sys.argv[1]
baixa_prioridade()
os.environ["OMP_NUM_THREADS"] = "4"
ind = json.load(open(os.path.join(work, "segmentos_indice.json")))
ini = json.load(open(sys.argv[2]))
nos = [s for g in sorted(ini["poses_iniciais"]) if g in ind for s in ind[g]]
T0 = [np.array(ini["poses_iniciais"][s["scan"]]) for s in nos]
ref = next(n for n, s in enumerate(nos) if s["scan"] == ini["referencia"] and s["seg"] == 0)
lacos = pickle.load(open(os.path.join(work, "lacos_rodada1.pkl"), "rb"))
cache = {}


def nuvem(n):
    if n not in cache:
        cache[n] = S._normais(np.load(nos[n]["arquivo"]))
    return cache[n]


def vies(T, amostra=80):
    rng = np.random.default_rng(0)
    escolha = rng.choice(len(lacos), min(amostra, len(lacos)), replace=False)
    v = []
    for k in escolha:
        i, j = lacos[k][0], lacos[k][1]
        r = S.residuo_plano(nuvem(i), nuvem(j), np.linalg.inv(T[i]) @ T[j])
        if r[1] is not None:
            v.append(r[1] * 1000)
    return float(np.median(v)), float(np.percentile(v, 90))


m, p9 = vies(T0)
print(f"poses iniciais (rigido por scan): vies normal por celula {m:.0f} um (p90 {p9:.0f})", flush=True)
for peso in [float(x) for x in sys.argv[3:]]:
    odo = S.arestas_odometria(nos, peso)
    t = time.time()
    T, _ = S.otimizar(nos, [x.copy() for x in T0], odo, lacos, ref)
    m, p9 = vies(T)
    print(f"peso odometria {peso:g}: vies normal por celula {m:.0f} um (p90 {p9:.0f}) [{time.time()-t:.0f}s]", flush=True)
    json.dump([x.tolist() for x in T], open(os.path.join(work, f"T_peso_{peso:g}.json"), "w"))
