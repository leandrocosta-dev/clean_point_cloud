"""Corrige a posicao LOCAL da interna para a espessura medida (ponte), preservando o relevo interno, em vez de cortar o que atravessa.

    python pipeline/corrigir_interna_local.py interna.ply externa.ply nuvem_externa.ply --espessura 2.0 --saida out.ply [--raio 25] [--perto 8]

Para cada vertice de pele interna (perto da externa) calcula delta = -(s + espessura), s = distancia com sinal ate a externa ao longo da normal externa.
O delta e SUAVIZADO espacialmente (mediana num raio de --raio mm): corrige o desalinhamento de grande escala (extremidades/flancos onde o scan interno
ficou ~2-5 mm fora do lugar) mas deixa as nervuras/fixacoes/relevo (alta frequencia) como medidos. Aplica-se a todos os vertices a < 30 mm da externa.
Premissa (do usuario e da ponte): espessura quase constante; valor 2,0 mm = mediana medida na ponte 20260921_190200. Rodar depois sem_atravessar.py.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna"); ap.add_argument("externa"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--espessura", type=float, default=2.0); ap.add_argument("--raio", type=float, default=25.0); ap.add_argument("--perto", type=float, default=8.0)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna); X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem); Pn = np.asarray(E.points); Nn = np.asarray(E.normals)
P = np.asarray(X.sample_points_uniformly(2_000_000).points); _, j = cKDTree(Pn).query(P, workers=8); N = Nn[j]; t = cKDTree(P)
V = np.asarray(I.vertices).copy(); d, k = t.query(V, workers=8); n = N[k]; s = np.einsum("ij,ij->i", V - P[k], n)
pele = (d < a.perto) & (np.abs(s + a.espessura) < a.perto)
delta = -(s + a.espessura)
print("vertices de pele: %.1f%% | delta bruto mediano %.2f p10 %.2f p90 %.2f" % (pele.mean() * 100, np.median(delta[pele]), *np.percentile(delta[pele], [10, 90])))
Vs = V[pele]; ds = delta[pele]; sub = np.random.default_rng(0).choice(len(Vs), min(len(Vs), 400_000), replace=False); tr = cKDTree(Vs[sub]); dsub = ds[sub]
quer = d < 30
dist, idx = tr.query(V[quer], k=48, distance_upper_bound=a.raio, workers=8)
vals = np.where(np.isfinite(dist), dsub[np.minimum(idx, len(dsub) - 1)], np.nan)
dm = np.nanmedian(np.where(np.isfinite(vals), vals, np.nan), axis=1) if False else np.array([np.median(r[np.isfinite(r)]) if np.isfinite(r).any() else 0.0 for r in vals])
dsm = np.zeros(len(V)); dsm[quer] = dm
V += dsm[:, None] * n
I.vertices = o3d.utility.Vector3dVector(V)
d2, k2 = t.query(V, workers=8); s2 = np.einsum("ij,ij->i", V - P[k2], N[k2]); p2 = (d2 < a.perto)
print("deslocamento suave aplicado: mediana %.2f mm, p10 %.2f, p90 %.2f" % (np.median(dsm[quer]), *np.percentile(dsm[quer], [10, 90])))
print("s depois (perto da pele): mediana %.2f p10 %.2f p25 %.2f p75 %.2f p90 %.2f" % (np.median(s2[p2]), *np.percentile(s2[p2], [10, 25, 75, 90])))
o3d.io.write_triangle_mesh(a.saida, I)
