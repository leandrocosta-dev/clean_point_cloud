"""Avalia a nuvem fundida pelo Revo contra os segmentos alinhados que a originaram.

    python pipeline/avaliar_fusao.py <fused.ply> <pasta_segmentos> <segmentos.json> <selecao.json>

1) ICP rigido leve da fundida sobre os segmentos (o Revo pode recentrar); 2) vies normal por celula de 30 mm de
cada scan contra a fundida (o Revo reotimizou/deslocou poses? scan que nao encaixa na fundida); 3) espessura local
da fundida (camada dupla = desalinhamento ou fantasma).
"""
import json, os, sys, time
import numpy as np
import open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alinhar_segmentos as S
from alinhar_varios import baixa_prioridade
from scipy.spatial import cKDTree

baixa_prioridade(); os.environ["OMP_NUM_THREADS"] = "6"
ply, pasta, segjson, selecao = sys.argv[1:5]
reg = o3d.pipelines.registration
t0 = time.time()
F = o3d.io.read_point_cloud(ply)
P = np.asarray(F.points)
print(f"fundida: {len(P):,} pontos | extensao {np.round(np.ptp(P,0),0)} mm | {time.time()-t0:.0f}s", flush=True)
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
pj = json.load(open(segjson))["segmentos"]
esc = json.load(open(selecao))["por_scan"]
partes = {}
for g, segs in esc.items():
    X = []
    for s in ind[g]:
        if s["seg"] in segs:
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            C = np.load(s["arquivo"]).astype(np.float64); X.append(C @ T[:3, :3].T + T[:3, 3])
    partes[g] = np.vstack(X)
U = np.vstack(list(partes.values()))
# referencia: fundida a 0.8 mm (centroides) com normais
rd = F.voxel_down_sample(0.8); rd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.4, max_nn=40))
ud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(U)).voxel_down_sample(1.5)
ic = reg.registration_icp(ud, rd, 6.0, np.eye(4), reg.TransformationEstimationPointToPlane(reg.TukeyLoss(0.6)), reg.ICPConvergenceCriteria(max_iteration=40))
Tg = ic.transformation
ang = np.degrees(np.arccos(np.clip((np.trace(Tg[:3, :3]) - 1) / 2, -1, 1)))
print(f"ICP rigido segmentos->fundida: deslocamento {np.round(Tg[:3,3],2)} mm, {ang:.3f} graus (referencial do Revo)", flush=True)
todos = []
print(f"{'scan':16s} {'cobertura':>9s} {'vies med (um)':>14s} {'p90':>6s}")
for g, X in sorted(partes.items()):
    pb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)).voxel_down_sample(0.8)
    r = S.residuo_plano(rd, pb, Tg, tol=2.0)
    if r[1] is None:
        print(f"{g:16s} sem encaixe"); continue
    todos.append(r[1] * 1000)
    print(f"{g:16s} {r[0]*100:8.0f}% {r[1]*1000:14.0f} {r[2]*1000:6.0f}", flush=True)
print(f"todos os scans contra a fundida: vies mediano {np.median(todos):.0f} um (p90 entre scans {np.percentile(todos,90):.0f})")
# espessura local
rng = np.random.default_rng(0); q = P[rng.choice(len(P), 400000, replace=False)]
arv = cKDTree(rd.points if False else np.asarray(F.voxel_down_sample(0.4).points))
d, idx = arv.query(q, k=30, workers=4)
N = np.asarray(F.voxel_down_sample(0.4).points)[idx] ; N = N - N.mean(1, keepdims=True)
C = np.einsum("nki,nkj->nij", N, N) / 30
sig = np.sqrt(np.maximum(np.linalg.eigvalsh(C)[:, 0], 0))
print(f"espessura local (sigma do plano em 30 vizinhos, ~1,2 mm de raio): mediana {np.median(sig)*1000:.0f} um, p90 {np.percentile(sig,90)*1000:.0f}, p99 {np.percentile(sig,99)*1000:.0f}")
# mapa espacial da espessura
c = q.mean(0); w, V = np.linalg.eigh(np.cov((q - c).T)); Q = (q - c) @ V[:, ::-1]
print("espessura mediana (um) por faixa ao longo do comprimento:", [int(np.median(sig[(Q[:,0]>=a)&(Q[:,0]<b)])*1000) for a, b in zip(np.percentile(Q[:,0], np.linspace(0,100,9))[:-1], np.percentile(Q[:,0], np.linspace(0,100,9))[1:])])
