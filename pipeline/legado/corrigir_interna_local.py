"""Corrige a posicao LOCAL da interna para a espessura medida (ponte), preservando o relevo interno, em vez de cortar o que atravessa.

    python pipeline/corrigir_interna_local.py interna.ply externa.ply nuvem_externa.ply --espessura 2.0 --saida out.ply [--raio 25] [--perto 8]

Para cada vertice de pele interna (perto da externa) calcula delta = -(s + espessura), s = distancia com sinal ate a externa ao longo da normal externa.
O delta e SUAVIZADO espacialmente (mediana num raio de --raio mm): corrige o desalinhamento de grande escala (extremidades/flancos onde o scan interno
ficou ~2-5 mm fora do lugar) mas deixa as nervuras/fixacoes/relevo (alta frequencia) como medidos. Aplica-se a todos os vertices a < 30 mm da externa.
Premissa (do usuario e da ponte): espessura quase constante; valor 2,0 mm = mediana medida na ponte 20260921_190200. Rodar depois sem_atravessar.py.
"""

import argparse

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna")
ap.add_argument("externa")
ap.add_argument("nuvem")
ap.add_argument("--saida", required=True)
ap.add_argument("--espessura", type=float, default=2.0)
ap.add_argument("--raio", type=float, default=25.0)
ap.add_argument("--perto", type=float, default=8.0)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna)
X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem)
Pn = np.asarray(E.points)
Nn = np.asarray(E.normals)
xp = gpu.xp()
P, _ = gg.amostrar_malha(np.asarray(X.vertices), np.asarray(X.triangles), 2_000_000, saida_gpu=True)
_, j = gpu.vizinho_mais_proximo(Pn, P, dist_max=np.inf)
N = gpu.para_gpu(Nn, xp.float64)[j]  # normal da nuvem mais proxima (GPU)
V = gpu.para_gpu(np.asarray(I.vertices), xp.float64)
d, k = gpu.vizinho_mais_proximo(P, V, dist_max=np.inf)
n = N[k]
s = ((V - P[k]) * n).sum(1)
pele = (d < a.perto) & (xp.abs(s + a.espessura) < a.perto)
delta = -(s + a.espessura)
dp = gpu.para_cpu(delta[pele])
print(
    "vertices de pele: %.1f%% | delta bruto mediano %.2f p10 %.2f p90 %.2f"
    % (float(pele.mean()) * 100, np.median(dp), *np.percentile(dp, [10, 90]))
)
Vs = V[pele]
ds = delta[pele]
sub = gpu.para_gpu(np.random.default_rng(0).choice(len(Vs), min(len(Vs), 400_000), replace=False))
dsub = ds[sub]
quer = d < 30
dist, idx = gpu.knn(Vs[sub], V[quer], 48, raio_max=a.raio)  # 48 vizinhos a <= raio (GPU); idx = -1 onde faltam
vals = xp.where(idx >= 0, dsub[xp.maximum(idx, 0)], xp.nan)
dm = xp.nanmedian(vals, axis=1)
dm = xp.where(xp.isnan(dm), 0.0, dm)  # mediana dos vizinhos; sem vizinho: 0
dsm = xp.zeros(len(V))
dsm[quer] = dm
V = V + dsm[:, None] * n
I.vertices = o3d.utility.Vector3dVector(gpu.para_cpu(V))
d2, k2 = gpu.vizinho_mais_proximo(P, V, dist_max=np.inf)
s2 = ((V - P[k2]) * N[k2]).sum(1)
p2 = d2 < a.perto
dq = gpu.para_cpu(dsm[quer])
s2p = gpu.para_cpu(s2[p2])
print(
    "deslocamento suave aplicado: mediana %.2f mm, p10 %.2f, p90 %.2f" % (np.median(dq), *np.percentile(dq, [10, 90]))
)
print(
    "s depois (perto da pele): mediana %.2f p10 %.2f p25 %.2f p75 %.2f p90 %.2f"
    % (np.median(s2p), *np.percentile(s2p, [10, 25, 75, 90]))
)
o3d.io.write_triangle_mesh(a.saida, I)
