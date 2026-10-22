"""Mapa de confianca: em quantos scans INDEPENDENTES cada ponto da nuvem final se apoia.

    python pipeline/confianca.py <nuvem_completa.ply> <fonte.npy> <pasta_segmentos> <segmentos.json> --saida <pasta> [--tol 1.0]

Para cada ponto da nuvem (todos os acrescentados + amostra do Revo) conta os scans que tem pontos a menos de --tol mm
(centroides de 1 mm dos segmentos limpos, no referencial do Revo por ICP rigido leve). Classes: 0 = sem apoio,
1 = um scan, 2 = dois, 3+ = tres ou mais. Gera estatisticas, imagem e uma nuvem colorida:
  cinza = Revo | laranja = acrescentado e confirmado (>= 2 scans) | vermelho = acrescentado com 1 scan so.
"""
import argparse, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade

ap = argparse.ArgumentParser(); ap.add_argument("nuvem"); ap.add_argument("fonte"); ap.add_argument("pasta"); ap.add_argument("segmentos")
ap.add_argument("--saida", required=True); ap.add_argument("--tol", type=float, default=1.0); ap.add_argument("--amostra-revo", type=int, default=3_000_000)
a = ap.parse_args(); baixa_prioridade(); os.makedirs(a.saida, exist_ok=True); t0 = time.time()
U = o3d.io.read_point_cloud(a.nuvem); P = np.asarray(U.points); N = np.asarray(U.normals); f = np.load(a.fonte)
ind = json.load(open(os.path.join(a.pasta, "segmentos_indice.json"))); pj = json.load(open(a.segmentos))["segmentos"]
por = {}
for g in sorted(ind):
    L = []
    for s in ind[g]:
        T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"]); C = np.load(s["arquivo"]).astype(float); L.append(C @ T[:3, :3].T + T[:3, 3])
    por[g] = np.vstack(L)
# referencial: segmentos -> Revo
rr = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[f == 0][::8])).voxel_down_sample(1.0); rr.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
ss = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([x[::10] for x in por.values()]))).voxel_down_sample(1.5)
T = o3d.pipelines.registration.registration_icp(ss, rr, 4.0, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(0.5))).transformation
arv = {g: cKDTree(X @ T[:3, :3].T + T[:3, 3]) for g, X in por.items()}
print(f"{len(arv)} scans indexados ({time.time()-t0:.0f}s)", flush=True)
rng = np.random.default_rng(0)
idx_rev = np.flatnonzero(f == 0); idx_rev = rng.choice(idx_rev, min(a.amostra_revo, len(idx_rev)), replace=False)
idx_add = np.flatnonzero(f == 1)
idx = np.concatenate([idx_rev, idx_add])
cont = np.zeros(len(idx), np.int16)
for g, t in arv.items():
    for i0 in range(0, len(idx), 2_000_000):
        d, _ = t.query(P[idx[i0:i0 + 2_000_000]], distance_upper_bound=a.tol, workers=4)
        cont[i0:i0 + 2_000_000] += np.isfinite(d)
n_rev = len(idx_rev); c_rev, c_add = cont[:n_rev], cont[n_rev:]
def lin(nome, c):
    print(f"  {nome:34s} 0: {np.mean(c==0)*100:5.1f}% | 1: {np.mean(c==1)*100:5.1f}% | 2: {np.mean(c==2)*100:5.1f}% | 3+: {np.mean(c>=3)*100:5.1f}%  (>= 2 scans: {np.mean(c>=2)*100:5.1f}%)")
print(f"apoio por scans independentes a < {a.tol:g} mm:")
lin("pontos do Revo (amostra)", c_rev); lin("pontos acrescentados (todos)", c_add)
tot_conf = (np.sum(c_rev >= 2) / len(c_rev)) * (f == 0).sum() + np.sum(c_add >= 2)
print(f"superficie final confirmada por >= 2 scans: {tot_conf/len(P)*100:.1f}% dos pontos")
# nuvem colorida: acrescentados confirmados x nao confirmados
cor = np.vstack([np.tile([0.78, 0.78, 0.78], (len(P), 1))])
conf_add = c_add >= 2
cor[idx_add[conf_add]] = [1.0, 0.45, 0.0]; cor[idx_add[~conf_add]] = [0.85, 0.05, 0.05]
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.normals = o3d.utility.Vector3dVector(N); pc.colors = o3d.utility.Vector3dVector(cor)
o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_completa_confianca.ply"), pc, write_ascii=False)
mask_remover = np.zeros(len(P), bool); mask_remover[idx_add[~conf_add]] = True
np.save(os.path.join(a.saida, "nao_confirmados.npy"), mask_remover)
json.dump(dict(tol_mm=a.tol, revo=dict(zero=float(np.mean(c_rev == 0)), um=float(np.mean(c_rev == 1)), dois_mais=float(np.mean(c_rev >= 2))),
               acrescentados=dict(n=int(len(c_add)), confirmados=int(conf_add.sum()), nao_confirmados=int((~conf_add).sum())),
               confirmada_total_pct=float(tot_conf / len(P) * 100)), open(os.path.join(a.saida, "confianca.json"), "w"), indent=1)
# imagem: so os acrescentados por classe sobre a peca em cinza claro
c0 = P.mean(0); w, V = np.linalg.eigh(np.cov((P - c0).T)); R_ = V[:, ::-1]
sel = rng.choice(len(P), 1_200_000, replace=False); Q = (P - c0) @ R_
fig, ax = plt.subplots(1, 3, figsize=(24, 8), dpi=65)
for a_, (u, v) in zip(ax, ((0, 1), (0, 2), (1, 2))):
    a_.scatter(Q[sel, u], Q[sel, v], s=0.2, c="#d0d0d0", linewidths=0)
    ia = idx_add[conf_add][::3]; a_.scatter(Q[ia, u], Q[ia, v], s=0.4, c="#ff7300", linewidths=0)
    ib = idx_add[~conf_add][::1]; a_.scatter(Q[ib, u], Q[ib, v], s=0.6, c="#d90d0d", linewidths=0); a_.set_aspect("equal"); a_.axis("off")
ax[0].set_title("cinza: Revo | laranja: acrescentado confirmado por >=2 scans | vermelho: acrescentado com 1 scan so")
fig.tight_layout(); fig.savefig(os.path.join(a.saida, "mapa_de_confianca.png"))
print(f"pronto ({time.time()-t0:.0f}s)")
