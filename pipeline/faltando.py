"""O que dos segmentos (limpos) ainda NAO esta na nuvem fundida, e quais segmentos nao usados o trariam.

    python pipeline/faltando.py <fundida.ply> <pasta_segmentos> <segmentos.json> <selecao.json> --saida <pasta>
"""
import json, os, sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade
baixa_prioridade()
ply, pasta, segjson, selecao = sys.argv[1:5]
saida = sys.argv[sys.argv.index("--saida") + 1]
os.makedirs(saida, exist_ok=True)
F = o3d.io.read_point_cloud(ply).voxel_down_sample(0.6)
A = np.asarray(F.points)
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json"))); pj = json.load(open(segjson))["segmentos"]; esc = json.load(open(selecao))["por_scan"]
segs = []
for g in sorted(ind):
    for s in ind[g]:
        T4 = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"]); C = np.load(s["arquivo"]).astype(float)
        segs.append((g, s["seg"], C @ T4[:3, :3].T + T4[:3, 3], s["seg"] in esc.get(g, [])))
U = np.vstack([c for *_, c, u in segs if u]); 
ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A[::6])); ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(U)).voxel_down_sample(1.5)
Tr = o3d.pipelines.registration.registration_icp(src, ref, 4.0, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane()).transformation
arv = cKDTree(A)
linhas = []; falt = []
for g, s, C, usado in segs:
    C = C @ Tr[:3, :3].T + Tr[:3, 3]
    d, _ = arv.query(C, workers=4)
    m = d > 2.0
    linhas.append((g, s, usado, len(C), int(m.sum())))
    if m.any(): falt.append((g, s, usado, C[m]))
tot = sum(l[3] for l in linhas); tot_f = sum(l[4] for l in linhas)
print(f"centroides de todos os segmentos: {tot:,} | a mais de 2 mm da fundida: {tot_f:,} ({tot_f/tot*100:.1f}%)")
u_f = sum(l[4] for l in linhas if l[2]); n_f = sum(l[4] for l in linhas if not l[2])
print(f"  dos segmentos USADOS: {u_f:,} pontos fora da fundida (descartados pelo Revo: poucos quadros/peso) | dos NAO usados: {n_f:,}")
# gulosa: que segmentos nao usados cobrem mais superficie que a fundida nao tem (em celulas de 3 mm)
cel = 3.0
chave = lambda P: np.unique(np.floor(P / cel).astype(np.int64), axis=0)
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
    if ganho < max(40, 0.004 * total_celulas): break
    cobertas |= cand.pop(g_); ordem.append((g_, ganho))
print(f"segmentos a acrescentar (ganho em celulas de 3 mm, acumulado %): ")
acc = 0
for (g, s), gn in ordem[:20]:
    acc += gn; print(f"  +{g[-6:]}#{s}: {gn:6,} ({acc/total_celulas*100:4.1f}% do que falta)")
json.dump(dict(acrescentar=[dict(scan=g, seg=s, ganho=gn) for (g, s), gn in ordem], total_celulas_faltando=total_celulas, centroides_fora=tot_f), open(os.path.join(saida, "faltando.json"), "w"), indent=1)
# imagem: fundida em cinza, o que falta em vermelho
Q0 = A.mean(0); w, V = np.linalg.eigh(np.cov((A - Q0).T)); R = V[:, ::-1]
X = np.vstack([x for g, s, u, x in falt]) if falt else np.zeros((0, 3))
rng = np.random.default_rng(0); As = A[rng.choice(len(A), min(600000, len(A)), replace=False)]
Xs = X[rng.choice(len(X), min(250000, len(X)), replace=False)] if len(X) else X
fig, ax = plt.subplots(1, 3, figsize=(24, 8), dpi=70)
for a, (u, v) in zip(ax, ((0, 1), (0, 2), (1, 2))):
    a.scatter(((As - Q0) @ R)[:, u], ((As - Q0) @ R)[:, v], s=0.15, c="#bbbbbb", linewidths=0)
    if len(Xs): a.scatter(((Xs - Q0) @ R)[:, u], ((Xs - Q0) @ R)[:, v], s=0.4, c="#d62728", linewidths=0)
    a.set_aspect("equal"); a.axis("off")
ax[0].set_title("cinza: fundida | vermelho: superficie dos segmentos (limpos) que a fundida nao tem")
fig.tight_layout(); fig.savefig(os.path.join(saida, "o_que_falta.png"))
