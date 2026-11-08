"""Avalia a nuvem fundida pelo Revo contra os segmentos alinhados que a originaram.

    python pipeline/avaliar_fusao.py <fused.ply> <pasta_segmentos> <segmentos.json> <selecao.json>

1) ICP rigido leve da fundida sobre os segmentos (o Revo pode recentrar); 2) vies normal por celula de 30 mm de
cada scan contra a fundida (o Revo reotimizou/deslocou poses? scan que nao encaixa na fundida); 3) espessura local
da fundida (camada dupla = desalinhamento ou fantasma).
"""

import json
import os
import sys
import time

import alinhar_segmentos as S
import gpu
import numpy as np
import open3d as o3d
from metricas import normais_hibridas
from recursos import baixa_prioridade

# GPU: reducao por voxel, normais, ICP robusto, espessura local (kNN + autovalores) e estatisticas. CPU: leitura de .ply/.npy/.json
# e a montagem das PointCloud do Open3D que alinhar_segmentos.residuo_plano aceita.
baixa_prioridade()
os.environ["OMP_NUM_THREADS"] = "6"
m = gpu.xp()
ply, pasta, segjson, selecao = sys.argv[1:5]
t0 = time.time()
F = o3d.io.read_point_cloud(ply)
P = gpu.para_gpu(np.asarray(F.points))
print(
    f"fundida: {len(P):,} pontos | extensao {np.round(gpu.para_cpu(P.max(0) - P.min(0)), 0)} mm | {time.time() - t0:.0f}s",
    flush=True,
)
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
pj = json.load(open(segjson))["segmentos"]
esc = json.load(open(selecao))["por_scan"]
partes = {}
for g, segs in esc.items():
    X = []
    for s in ind[g]:
        if s["seg"] in segs:
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            C = gpu.para_gpu(np.load(s["arquivo"]), m.float64)
            X.append(C @ gpu.para_gpu(T[:3, :3]).T + gpu.para_gpu(T[:3, 3]))
    partes[g] = m.concatenate(X)
U = m.concatenate(list(partes.values()))


def pcd(P, N=None):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(P)))
    if N is not None:
        pc.normals = o3d.utility.Vector3dVector(N)
    return pc


# referencia: fundida a 0.8 mm (centroides) com normais
rdP = gpu.reduzir_voxel(P, 0.8)[0]
rdN = normais_hibridas(rdP, 2.4, 40)
rd = pcd(rdP, rdN)
ud = gpu.reduzir_voxel(U, 1.5)[0]
Tg, ic = gpu.icp_ponto_plano(ud, rdP, rdN, T0=np.eye(4), dist_max=6.0, iteracoes=40, perda="tukey", c=0.6)
ang = np.degrees(np.arccos(np.clip((np.trace(Tg[:3, :3]) - 1) / 2, -1, 1)))
print(
    f"ICP rigido segmentos->fundida: deslocamento {np.round(Tg[:3, 3], 2)} mm, {ang:.3f} graus (referencial do Revo)",
    flush=True,
)
todos = []
print(f"{'scan':16s} {'cobertura':>9s} {'vies med (um)':>14s} {'p90':>6s}")
for g, X in sorted(partes.items()):
    pb = pcd(gpu.reduzir_voxel(X, 0.8)[0])
    r = S.residuo_plano(rd, pb, Tg, tol=2.0)
    if r[1] is None:
        print(f"{g:16s} sem encaixe")
        continue
    todos.append(r[1] * 1000)
    print(f"{g:16s} {r[0] * 100:8.0f}% {r[1] * 1000:14.0f} {r[2] * 1000:6.0f}", flush=True)
print(
    f"todos os scans contra a fundida: vies mediano {np.median(todos):.0f} um (p90 entre scans {np.percentile(todos, 90):.0f})"
)
# espessura local
rng = np.random.default_rng(0)
q = P[gpu.para_gpu(rng.choice(len(P), 400000, replace=False))]
Pd = gpu.reduzir_voxel(P, 0.4)[0]
_, idx = gpu.knn(Pd, q, 30)
N = Pd[idx]
N = N - N.mean(1, keepdims=True)
C = m.einsum("nki,nkj->nij", N, N) / 30
sig = m.sqrt(m.maximum(m.linalg.eigvalsh(C)[:, 0], 0))
print(
    f"espessura local (sigma do plano em 30 vizinhos, ~1,2 mm de raio): mediana {float(m.median(sig)) * 1000:.0f} um, p90 {float(m.percentile(sig, 90)) * 1000:.0f}, p99 {float(m.percentile(sig, 99)) * 1000:.0f}"
)
# mapa espacial da espessura
c = q.mean(0)
w, V = np.linalg.eigh(gpu.para_cpu(m.cov((q - c).T)))
Q = (q - c) @ gpu.para_gpu(V[:, ::-1])
borda = m.percentile(Q[:, 0], m.linspace(0, 100, 9))
print(
    "espessura mediana (um) por faixa ao longo do comprimento:",
    [int(float(m.median(sig[(Q[:, 0] >= lo) & (Q[:, 0] < hi)])) * 1000) for lo, hi in zip(borda[:-1], borda[1:])],
)
