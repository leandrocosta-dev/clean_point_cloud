"""Tira artefatos soltos de uma nuvem JA FUNDIDA (ex.: nuvem_nitida.ply do fundir_nitido): pontos voando,
franja rala nas bordas e grupinhos desconectados, sem comer a borda real da superficie.

    python pipeline/limpar_artefatos.py nuvem.ply nuvem_limpa.ply [--dist-nucleo 0.25 --fora-plano 0.03]

Nucleo = pontos com >= --viz-nucleo vizinhos em --raio mm. Ponto ralo so sai se estiver a > --dist-nucleo mm do
nucleo ou a > --fora-plano mm do plano local do nucleo; depois saem componentes com < --comp-min pontos.
Grava tambem <saida>_removidos.ply para conferir. Remover so pelo numero de vizinhos, repetido, come a borda real
(Projeto 8: 37% dos removidos estavam no plano da superficie).
"""
import argparse, json
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("entrada"); ap.add_argument("saida")
ap.add_argument("--raio", type=float, default=0.2)
ap.add_argument("--viz-nucleo", type=int, default=15)
ap.add_argument("--dist-nucleo", type=float, default=0.15)
ap.add_argument("--comp-min", type=int, default=500)
ap.add_argument("--fora-plano", type=float, default=0.03, help="mm do plano local do nucleo (r 0,3); ponto ralo acima disso = voando")
a = ap.parse_args()
p = o3d.io.read_point_cloud(a.entrada); X = np.asarray(p.points); N = np.asarray(p.normals)
manter = np.ones(len(X), bool); motivos = {}
# 1-2) nucleo denso (>= --viz-nucleo vizinhos); ponto ralo so sai se nao continua a superficie do nucleo:
#      longe dele (> --dist-nucleo) ou fora do plano local do nucleo (> --fora-plano). Borda real fica.
t = cKDTree(X); n = np.array([len(v) for v in t.query_ball_point(X, a.raio, workers=8)]) - 1
nucleo = n >= a.viz_nucleo
idx_n = np.flatnonzero(nucleo); tn = cKDTree(X[idx_n])
ralo = np.flatnonzero(~nucleo)
d, j = tn.query(X[ralo], workers=8)
viz = tn.query_ball_point(X[ralo], 0.3, workers=8)
off = np.full(len(ralo), np.inf)
for k, v in enumerate(viz):
    if len(v) >= 6:
        Q = X[idx_n[v]]; c = Q.mean(0); w, V = np.linalg.eigh(np.cov((Q - c).T))
        off[k] = abs((X[ralo[k]] - c) @ V[:, 0])
ruim = (d > a.dist_nucleo) | (off > a.fora_plano)
manter[ralo[ruim]] = False
motivos.update(nucleo=int(nucleo.sum()), ralos=int(len(ralo)), ralos_longe=int((d > a.dist_nucleo).sum()),
               ralos_fora_plano=int(((off > a.fora_plano) & (d <= a.dist_nucleo)).sum()))
# 3) componentes conectados pequenos
idx = np.flatnonzero(manter); t = cKDTree(X[idx]); pr = t.query_pairs(a.raio, output_type="ndarray")
nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(idx), len(idx))), directed=False)
sz = np.bincount(lab); ruim = sz[lab] < a.comp_min
manter[idx[ruim]] = False; motivos["componentes_pequenos"] = int(ruim.sum()); motivos["componentes_antes"] = int(nc)
q = p.select_by_index(np.flatnonzero(manter))
o3d.io.write_point_cloud(a.saida, q)
o3d.io.write_point_cloud(a.saida.replace(".ply", "_removidos.ply"), p.select_by_index(np.flatnonzero(~manter)))
motivos.update(entrada=len(X), saida=int(manter.sum()), removidos=int((~manter).sum()))
print(json.dumps(motivos, indent=1))
