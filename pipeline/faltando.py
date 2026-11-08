"""O que dos segmentos (limpos) ainda NAO esta na nuvem fundida, e quais segmentos nao usados o trariam.

python pipeline/faltando.py <fundida.ply> <pasta_segmentos> <segmentos.json> <selecao.json> --saida <pasta>
"""

import json
import os
import sys

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import gpu
import matplotlib.pyplot as plt
from consulta3d import _celulas_int
from metricas import normais_hibridas
from recursos import baixa_prioridade

# GPU: reducao por voxel, normais, ICP, vizinhos, celulas e projecoes. CPU: leitura de .ply/.npy/.json, os conjuntos de celulas
# da gulosa (logica de conjuntos de inteiros) e o desenho matplotlib.
baixa_prioridade()
m = gpu.xp()
ply, pasta, segjson, selecao = sys.argv[1:5]
saida = sys.argv[sys.argv.index("--saida") + 1]
os.makedirs(saida, exist_ok=True)
A = gpu.reduzir_voxel(gpu.para_gpu(np.asarray(o3d.io.read_point_cloud(ply).points)), 0.6)[0]
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
pj = json.load(open(segjson))["segmentos"]
esc = json.load(open(selecao))["por_scan"]
segs = []
for g in sorted(ind):
    for s in ind[g]:
        T4 = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
        C = gpu.para_gpu(np.load(s["arquivo"]), m.float64)
        segs.append((g, s["seg"], C @ gpu.para_gpu(T4[:3, :3]).T + gpu.para_gpu(T4[:3, 3]), s["seg"] in esc.get(g, [])))
U = m.concatenate([c for *_, c, u in segs if u])
ref = A[::6]
nref = gpu.para_gpu(normais_hibridas(ref, 4, 30))
src = gpu.reduzir_voxel(U, 1.5)[0]
Tr, _ = gpu.icp_ponto_plano(
    src, ref, nref, T0=np.eye(4), dist_max=4.0, iteracoes=30, perda="l2"
)  # ponto-plano sem perda robusta (padrao do Open3D)
Trg = gpu.para_gpu(Tr[:3, :3])
Ttg = gpu.para_gpu(Tr[:3, 3])
linhas = []
falt = []
for g, s, C, usado in segs:
    C = C @ Trg.T + Ttg
    _, j = gpu.vizinho_mais_proximo(A, C, dist_max=2.0)  # sem vizinho a <= 2 mm = fora da fundida
    mk = j < 0
    linhas.append((g, s, usado, len(C), int(mk.sum())))
    if bool(mk.any()):
        falt.append((g, s, usado, C[mk]))
tot = sum(l[3] for l in linhas)
tot_f = sum(l[4] for l in linhas)
print(f"centroides de todos os segmentos: {tot:,} | a mais de 2 mm da fundida: {tot_f:,} ({tot_f / tot * 100:.1f}%)")
u_f = sum(l[4] for l in linhas if l[2])
n_f = sum(l[4] for l in linhas if not l[2])
print(
    f"  dos segmentos USADOS: {u_f:,} pontos fora da fundida (descartados pelo Revo: poucos quadros/peso) | dos NAO usados: {n_f:,}"
)
# gulosa: que segmentos nao usados cobrem mais superficie que a fundida nao tem (em celulas de 3 mm)
cel = 3.0
chave = lambda P: _celulas_int(P, cel)[0]
cand = {}
for g, s_, usado, X in falt:
    if not usado:
        cand[(g, s_)] = set(map(tuple, chave(X)))
faltam = set().union(*cand.values()) if cand else set()
cobertas, ordem = set(), []
total_celulas = len(faltam)
print(f"superficie de segmentos nao usados que a fundida nao tem: {total_celulas:,} celulas de 3 mm")
while cand:
    g_, ganho = max(((k, len(v - cobertas)) for k, v in cand.items()), key=lambda x: x[1])
    if ganho < max(40, 0.004 * total_celulas):
        break
    cobertas |= cand.pop(g_)
    ordem.append((g_, ganho))
print("segmentos a acrescentar (ganho em celulas de 3 mm, acumulado %): ")
acc = 0
for (g, s), gn in ordem[:20]:
    acc += gn
    print(f"  +{g[-6:]}#{s}: {gn:6,} ({acc / total_celulas * 100:4.1f}% do que falta)")
json.dump(
    dict(
        acrescentar=[dict(scan=g, seg=s, ganho=gn) for (g, s), gn in ordem],
        total_celulas_faltando=total_celulas,
        centroides_fora=tot_f,
    ),
    open(os.path.join(saida, "faltando.json"), "w"),
    indent=1,
)
# imagem: fundida em cinza, o que falta em vermelho
Q0 = A.mean(0)
w, V = np.linalg.eigh(gpu.para_cpu(m.cov((A - Q0).T)))
R = gpu.para_gpu(V[:, ::-1])
X = m.concatenate([x for g, s, u, x in falt]) if falt else m.zeros((0, 3))
rng = np.random.default_rng(0)
As = A[gpu.para_gpu(rng.choice(len(A), min(600000, len(A)), replace=False))]
Xs = X[gpu.para_gpu(rng.choice(len(X), min(250000, len(X)), replace=False))] if len(X) else X
PA, PX = gpu.para_cpu((As - Q0) @ R), (gpu.para_cpu((Xs - Q0) @ R) if len(Xs) else None)
fig, ax = plt.subplots(1, 3, figsize=(24, 8), dpi=70)
for a, (u, v) in zip(ax, ((0, 1), (0, 2), (1, 2))):
    a.scatter(PA[:, u], PA[:, v], s=0.15, c="#bbbbbb", linewidths=0)
    if PX is not None:
        a.scatter(PX[:, u], PX[:, v], s=0.4, c="#d62728", linewidths=0)
    a.set_aspect("equal")
    a.axis("off")
ax[0].set_title("cinza: fundida | vermelho: superficie dos segmentos (limpos) que a fundida nao tem")
fig.tight_layout()
fig.savefig(os.path.join(saida, "o_que_falta.png"))
