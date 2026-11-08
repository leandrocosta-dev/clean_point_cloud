"""Buracos da malha e quais segmentos NAO usados poderiam fecha-los.

    python pipeline/buracos.py <malha.ply> <pasta_segmentos> <segmentos.json> <selecao.json> [--saida buracos.json]
Contornos abertos (arestas de um so triangulo) agrupados em lacos; cada laco grande e testado contra os centroides
dos segmentos nao escolhidos (no referencial do Revo, via ICP leve). Lista o que da para fechar acrescentando
segmentos e o que so um novo scan resolveria.
"""

import json
import os
import sys

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d
from alinhar_varios import baixa_prioridade

baixa_prioridade()
malha, pasta, segjson, selecao = sys.argv[1:5]
saida = sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else None
M = o3d.io.read_triangle_mesh(malha)
V = np.asarray(M.vertices)
T = np.asarray(M.triangles).astype(np.int64)
xp = gpu.xp()
Vg = gpu.para_gpu(V, xp.float64)
Tg = gpu.para_gpu(T)
e = xp.sort(
    xp.concatenate([Tg[:, [0, 1]], Tg[:, [1, 2]], Tg[:, [2, 0]]]), axis=1
)  # arestas de contorno na GPU (aresta de 1 so triangulo)
u, _, cnt = gpu.agrupar(e[:, 0] * len(V) + e[:, 1])
b = u[cnt == 1]
a0, a1 = b // len(V), b % len(V)
print(f"malha: {len(V):,} vertices, {len(T):,} triangulos | arestas de contorno: {len(b):,}", flush=True)
vs = xp.unique(xp.concatenate([a0, a1]))
ia, ib = xp.searchsorted(vs, a0), xp.searchsorted(vs, a1)
lab = gg.rotulos_por_arestas(len(vs), ia, ib)
_, lab = xp.unique(lab, return_inverse=True)
lab = lab.reshape(-1)
n = int(lab.max()) + 1 if len(lab) else 0
per = gpu.para_cpu(xp.bincount(lab[ia], weights=xp.linalg.norm(Vg[a0] - Vg[a1], axis=1), minlength=n))
nv = gpu.para_cpu(xp.bincount(lab, minlength=n))
lab = gpu.para_cpu(lab)
vs = gpu.para_cpu(vs)
grandes = [i for i in np.argsort(-per) if per[i] >= 25.0][:60]
print(
    f"contornos: {n:,} | com perimetro >= 25 mm: {sum(per >= 25):,} | maiores: {np.round(per[grandes[:8]]).astype(int).tolist()} mm",
    flush=True,
)
# nuvens dos segmentos nao usados no referencial do Revo
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
pj = json.load(open(segjson))["segmentos"]
esc = json.load(open(selecao))["por_scan"]
usados, livres = [], []
for g in sorted(ind):
    for s in ind[g]:
        T4 = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
        C = np.load(s["arquivo"]).astype(float)
        C = C @ T4[:3, :3].T + T4[:3, 3]
        (usados if s["seg"] in esc.get(g, []) else livres).append((g, s["seg"], C))
U = np.vstack([c for *_, c in usados])
ref, _ = gpu.reduzir_voxel(V[::20], 1.5)
Nref = gg.normais_hibrido(ref, 4, 30)
bom = np.isfinite(Nref).all(1)
ref, Nref = ref[bom], Nref[bom]
src, _ = gpu.reduzir_voxel(U, 1.5)
Tr, _ = gpu.icp_ponto_plano(
    src, ref, Nref, T0=np.eye(4), dist_max=4.0, iteracoes=30, perda="l2"
)  # ICP ponto-plano na GPU
arv = [(g, s, gg.transformar(c, Tr, saida_gpu=True)) for g, s, c in livres]
res = []
for i in grandes:
    sel = np.flatnonzero(lab == i)
    idv = vs[sel]
    P = V[idv]
    c = P.mean(0)
    raio = max(6.0, float(np.percentile(np.linalg.norm(P - c, axis=1), 90)))
    cobre = []
    for g, s, t in arv:
        n_ = int(gpu.contar_no_raio(t, c[None, :], raio * 0.8)[0])
        if n_ >= 200:
            cobre.append((n_, g[-6:], s))
    cobre.sort(reverse=True)
    res.append(
        dict(
            perimetro_mm=float(per[i]),
            centro=c.round(1).tolist(),
            raio_mm=raio,
            preenchivel=bool(cobre),
            cobertura=[dict(pts=n_, scan=g, seg=s) for n_, g, s in cobre[:4]],
        )
    )
fech = [r for r in res if r["preenchivel"]]
print(
    f"dos {len(res)} maiores buracos: {len(fech)} tem pontos nos segmentos nao usados (da para fechar) | {len(res) - len(fech)} so com novo scan"
)
for r in res[:25]:
    print(
        f"  perimetro {r['perimetro_mm']:6.0f} mm centro {r['centro']} | "
        + (
            ", ".join(f"{c['scan']}#{c['seg']}({c['pts']})" for c in r["cobertura"])
            if r["preenchivel"]
            else "SEM DADOS: precisa de novo scan"
        )
    )
if saida:
    json.dump(dict(buracos=res), open(saida, "w"), indent=1)
