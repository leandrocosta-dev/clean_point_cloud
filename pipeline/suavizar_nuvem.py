"""Suaviza uma nuvem ao longo da normal, so entre pontos da MESMA face (normais concordantes).

    python pipeline/suavizar_nuvem.py <nuvem.ply (com normais)> --saida <nova.ply> [--raio 0.8] [--passadas 2] [--cos-min 0.8]

Cada ponto vai para a media (ponderada por gaussiana) da altura dos vizinhos ate --raio mm cuja normal
tem cos > --cos-min com a dele, medida ao longo da normal do ponto. Desloca so ao longo da normal (nao encolhe
bordas no plano). A outra face de uma chapa fina nao entra (normal oposta). Reduz o ruido de camada entre scans.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--raio", type=float, default=0.8); ap.add_argument("--passadas", type=int, default=2)
ap.add_argument("--cos-min", type=float, default=0.8); ap.add_argument("--k", type=int, default=32)
a = ap.parse_args()
pc = o3d.io.read_point_cloud(a.nuvem)
P = np.asarray(pc.points).copy(); N = np.asarray(pc.normals)
sig = a.raio / 2
for it in range(a.passadas):
    t = cKDTree(P)
    novo = P.copy(); desl = []
    for s in range(0, len(P), 200000):
        sl = slice(s, s + 200000)
        d, i = t.query(P[sl], k=a.k, distance_upper_bound=a.raio, workers=4)
        ok = np.isfinite(d); i2 = np.where(ok, i, 0)
        cosn = np.einsum("nkc,nc->nk", N[i2], N[sl])
        ok &= cosn > a.cos_min
        w = np.exp(-(d ** 2) / (2 * sig ** 2)) * ok
        h = np.einsum("nkc,nc->nk", P[i2] - P[sl][:, None, :], N[sl])
        dh = (w * h).sum(1) / np.maximum(w.sum(1), 1e-9)
        dh[ok.sum(1) < 5] = 0
        novo[sl] = P[sl] + N[sl] * dh[:, None]
        desl.append(np.abs(dh))
    P = novo
    print(f"passada {it+1}: deslocamento mediano {np.median(np.concatenate(desl))*1000:.1f} um", flush=True)
pc.points = o3d.utility.Vector3dVector(P)
o3d.io.write_point_cloud(a.saida, pc)
print("gravado", a.saida)
