"""Relatorio de lacunas: contornos abertos da malha (buracos da casca), com tamanho e posicao, e imagem.

python pipeline/lacunas.py <malha.ply> --saida <pasta> [--min-perimetro 40]
"""

import argparse
import json
import os

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import gpu
import matplotlib.pyplot as plt
from metricas import componentes_grafo

# GPU: arestas de borda, componentes (contornos), perimetros, centros, extensoes e PCA. CPU: leitura da malha, JSON e imagem.
ap = argparse.ArgumentParser()
ap.add_argument("malha")
ap.add_argument("--saida", required=True)
ap.add_argument("--min-perimetro", type=float, default=40.0)
a = ap.parse_args()
os.makedirs(a.saida, exist_ok=True)
M = o3d.io.read_triangle_mesh(a.malha)
m = gpu.xp()
V = gpu.para_gpu(np.asarray(M.vertices))
T = gpu.para_gpu(np.asarray(M.triangles)).astype(m.int64)
nV = len(V)
e = m.sort(m.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
key = e[:, 0] * nV + e[:, 1]
u, _, cnt = gpu.agrupar(key)
b = u[cnt == 1]
a0, a1 = b // nV, b % nV  # arestas de borda = usadas por 1 triangulo
vs = m.unique(m.concatenate([a0, a1]))
mp = m.full(nV, -1, dtype=m.int64)
mp[vs] = m.arange(len(vs))
n, lab = componentes_grafo(len(vs), mp[a0], mp[a1])
lab = gpu.para_gpu(lab)
L = m.linalg.norm(V[a0] - V[a1], axis=1)
per = m.bincount(lab[mp[a0]], weights=L, minlength=n)
c0 = V.mean(0)
w, Vv = np.linalg.eigh(gpu.para_cpu(m.cov((V[::20] - c0).T)))
R = gpu.para_gpu(Vv[:, ::-1])
grandes = m.flatnonzero(per >= a.min_perimetro)
# por contorno grande: centro (media), extensao (max - min) por eixo, via soma/min/max por grupo
Pb = V[vs]
cont = m.bincount(lab, minlength=n)
centro = gpu.somar_por_grupo(Pb, lab, n) / m.maximum(cont, 1)[:, None]
mn = m.full((n, 3), m.inf)
mx = m.full((n, 3), -m.inf)
if gpu.em_gpu():
    import cupyx

    cupyx.scatter_min(mn, lab, Pb)
    cupyx.scatter_max(mx, lab, Pb)
else:
    np.minimum.at(mn, lab, Pb)
    np.maximum.at(mx, lab, Pb)
per_c, centro_c, ext_c = (gpu.para_cpu(x[grandes]) for x in (per, centro, mx - mn))
lacunas = []
for k in range(len(per_c)):
    d_eq = per_c[k] / np.pi  # diametro equivalente de um circulo com esse perimetro
    lacunas.append(
        dict(
            perimetro_mm=float(per_c[k]),
            diametro_equivalente_mm=float(d_eq),
            centro=np.round(centro_c[k], 1).tolist(),
            extensao_mm=np.round(ext_c[k], 1).tolist(),
        )
    )
lacunas.sort(key=lambda x: -x["perimetro_mm"])
print(f"malha {len(T):,} triangulos | contornos: {n:,} | com perimetro >= {a.min_perimetro:g} mm: {len(lacunas):,}")
for lo, hi in ((40, 100), (100, 250), (250, 600), (600, 1e9)):
    k = [x for x in lacunas if lo <= x["perimetro_mm"] < hi]
    print(f"  perimetro {lo}-{hi if hi < 1e8 else 'inf'} mm: {len(k)}")
print("maiores contornos (o primeiro e geralmente a borda externa da casca):")
for x in lacunas[:12]:
    print(
        f"  perimetro {x['perimetro_mm']:8.0f} mm | diam. eq. {x['diametro_equivalente_mm']:6.0f} mm | extensao {x['extensao_mm']} | centro {x['centro']}"
    )
json.dump(lacunas, open(os.path.join(a.saida, "lacunas.json"), "w"), indent=1)
Q = gpu.para_cpu((V[::6] - c0) @ R)
fig, ax = plt.subplots(1, 3, figsize=(24, 8), dpi=65)
QB = gpu.para_cpu((Pb[m.isin(lab, grandes)] - c0) @ R)
for a_, (p, q) in zip(ax, ((0, 1), (0, 2), (1, 2))):
    a_.scatter(Q[:, p], Q[:, q], s=0.15, c="#cfcfcf", linewidths=0)
    a_.scatter(QB[:, p], QB[:, q], s=0.8, c="#d62728", linewidths=0)
    a_.set_aspect("equal")
    a_.axis("off")
ax[0].set_title(f"contornos abertos da malha (vermelho): {len(lacunas)} com perimetro >= {a.min_perimetro:g} mm")
fig.tight_layout()
fig.savefig(os.path.join(a.saida, "lacunas.png"))
