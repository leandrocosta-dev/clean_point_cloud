"""Buracos da malha e quais segmentos NAO usados poderiam fecha-los.

    python pipeline/buracos.py <malha.ply> <pasta_segmentos> <segmentos.json> <selecao.json> [--saida buracos.json]
Contornos abertos (arestas de um so triangulo) agrupados em lacos; cada laco grande e testado contra os centroides
dos segmentos nao escolhidos (no referencial do Revo, via ICP leve). Lista o que da para fechar acrescentando
segmentos e o que so um novo scan resolveria.
"""
import json, os, sys
import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade
baixa_prioridade()
malha, pasta, segjson, selecao = sys.argv[1:5]
saida = sys.argv[sys.argv.index("--saida") + 1] if "--saida" in sys.argv else None
M = o3d.io.read_triangle_mesh(malha)
V = np.asarray(M.vertices); T = np.asarray(M.triangles).astype(np.int64)
e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]); e.sort(axis=1)
u, inv, cnt = np.unique(e[:, 0] * len(V) + e[:, 1], return_inverse=True, return_counts=True)
b = u[cnt == 1]; a0, a1 = b // len(V), b % len(V)
print(f"malha: {len(V):,} vertices, {len(T):,} triangulos | arestas de contorno: {len(b):,}", flush=True)
vs = np.unique(np.concatenate([a0, a1])); mp = {v: i for i, v in enumerate(vs)}
G = sparse.coo_matrix((np.ones(len(b)), ([mp[x] for x in a0], [mp[x] for x in a1])), shape=(len(vs), len(vs)))
n, lab = connected_components(G, directed=False)
per = np.bincount(lab[[mp[x] for x in a0]], weights=np.linalg.norm(V[a0] - V[a1], axis=1), minlength=n)
nv = np.bincount(lab, minlength=n)
grandes = [i for i in np.argsort(-per) if per[i] >= 25.0][:60]
print(f"contornos: {n:,} | com perimetro >= 25 mm: {sum(per>=25):,} | maiores: {np.round(per[grandes[:8]]).astype(int).tolist()} mm", flush=True)
# nuvens dos segmentos nao usados no referencial do Revo
ind = json.load(open(os.path.join(pasta, "segmentos_indice.json"))); pj = json.load(open(segjson))["segmentos"]; esc = json.load(open(selecao))["por_scan"]
usados, livres = [], []
for g in sorted(ind):
    for s in ind[g]:
        T4 = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"]); C = np.load(s["arquivo"]).astype(float); C = C @ T4[:3, :3].T + T4[:3, 3]
        (usados if s["seg"] in esc.get(g, []) else livres).append((g, s["seg"], C))
U = np.vstack([c for *_, c in usados])
ref = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(V[::20])).voxel_down_sample(1.5)
ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(U)).voxel_down_sample(1.5)
ic = o3d.pipelines.registration.registration_icp(src, ref, 4.0, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane())
Tr = ic.transformation
arv = [(g, s, cKDTree(c @ Tr[:3, :3].T + Tr[:3, 3])) for g, s, c in livres]
res = []
for i in grandes:
    sel = np.flatnonzero(lab == i); idv = vs[sel]; P = V[idv]; c = P.mean(0)
    raio = max(6.0, float(np.percentile(np.linalg.norm(P - c, axis=1), 90)))
    cobre = []
    for g, s, t in arv:
        n_ = len(t.query_ball_point(c, raio * 0.8))
        if n_ >= 200: cobre.append((n_, g[-6:], s))
    cobre.sort(reverse=True)
    res.append(dict(perimetro_mm=float(per[i]), centro=c.round(1).tolist(), raio_mm=raio, preenchivel=bool(cobre), cobertura=[dict(pts=n_, scan=g, seg=s) for n_, g, s in cobre[:4]]))
fech = [r for r in res if r["preenchivel"]]
print(f"dos {len(res)} maiores buracos: {len(fech)} tem pontos nos segmentos nao usados (da para fechar) | {len(res)-len(fech)} so com novo scan")
for r in res[:25]:
    print(f"  perimetro {r['perimetro_mm']:6.0f} mm centro {r['centro']} | " + (", ".join(f"{c['scan']}#{c['seg']}({c['pts']})" for c in r["cobertura"]) if r["preenchivel"] else "SEM DADOS: precisa de novo scan"))
if saida: json.dump(dict(buracos=res), open(saida, "w"), indent=1)
