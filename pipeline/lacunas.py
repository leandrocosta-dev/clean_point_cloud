"""Relatorio de lacunas: contornos abertos da malha (buracos da casca), com tamanho e posicao, e imagem.

    python pipeline/lacunas.py <malha.ply> --saida <pasta> [--min-perimetro 40]
"""
import argparse, json, os
import numpy as np, open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ap = argparse.ArgumentParser(); ap.add_argument("malha"); ap.add_argument("--saida", required=True); ap.add_argument("--min-perimetro", type=float, default=40.0)
a = ap.parse_args(); os.makedirs(a.saida, exist_ok=True)
M = o3d.io.read_triangle_mesh(a.malha); V = np.asarray(M.vertices); T = np.asarray(M.triangles).astype(np.int64)
e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); e.sort(axis=1); key = e[:, 0] * len(V) + e[:, 1]
u, cnt = np.unique(key, return_counts=True); b = u[cnt == 1]; a0, a1 = b // len(V), b % len(V)
vs = np.unique(np.concatenate([a0, a1])); mp = np.full(len(V), -1); mp[vs] = np.arange(len(vs))
G = sparse.coo_matrix((np.ones(len(b)), (mp[a0], mp[a1])), shape=(len(vs), len(vs)))
n, lab = connected_components(G, directed=False)
L = np.linalg.norm(V[a0] - V[a1], axis=1); per = np.bincount(lab[mp[a0]], weights=L, minlength=n)
c0 = V.mean(0); w, Vv = np.linalg.eigh(np.cov((V[::20] - c0).T)); R = Vv[:, ::-1]
lacunas = []
for i in np.flatnonzero(per >= a.min_perimetro):
    P = V[vs[lab == i]]; c = P.mean(0)
    d_eq = per[i] / np.pi                      # diametro equivalente de um circulo com esse perimetro
    lacunas.append(dict(perimetro_mm=float(per[i]), diametro_equivalente_mm=float(d_eq), centro=np.round(c, 1).tolist(),
                        extensao_mm=np.round(np.ptp(P, 0), 1).tolist()))
lacunas.sort(key=lambda x: -x["perimetro_mm"])
print(f"malha {len(T):,} triangulos | contornos: {n:,} | com perimetro >= {a.min_perimetro:g} mm: {len(lacunas):,}")
for lo, hi in ((40, 100), (100, 250), (250, 600), (600, 1e9)):
    k = [x for x in lacunas if lo <= x["perimetro_mm"] < hi]; print(f"  perimetro {lo}-{hi if hi<1e8 else 'inf'} mm: {len(k)}")
print("maiores contornos (o primeiro e geralmente a borda externa da casca):")
for x in lacunas[:12]: print(f"  perimetro {x['perimetro_mm']:8.0f} mm | diam. eq. {x['diametro_equivalente_mm']:6.0f} mm | extensao {x['extensao_mm']} | centro {x['centro']}")
json.dump(lacunas, open(os.path.join(a.saida, "lacunas.json"), "w"), indent=1)
Q = (V[::6] - c0) @ R; fig, ax = plt.subplots(1, 3, figsize=(24, 8), dpi=65)
sel = np.isin(lab, [i for i in np.flatnonzero(per >= a.min_perimetro)]); vv = vs[np.flatnonzero(np.isin(lab, np.flatnonzero(per >= a.min_perimetro)))]; QB = (V[vv] - c0) @ R
for a_, (p, q) in zip(ax, ((0, 1), (0, 2), (1, 2))):
    a_.scatter(Q[:, p], Q[:, q], s=0.15, c="#cfcfcf", linewidths=0); a_.scatter(QB[:, p], QB[:, q], s=0.8, c="#d62728", linewidths=0); a_.set_aspect("equal"); a_.axis("off")
ax[0].set_title(f"contornos abertos da malha (vermelho): {len(lacunas)} com perimetro >= {a.min_perimetro:g} mm")
fig.tight_layout(); fig.savefig(os.path.join(a.saida, "lacunas.png"))
