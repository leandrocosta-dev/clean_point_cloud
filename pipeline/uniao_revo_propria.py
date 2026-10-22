"""Une a nuvem fundida do Revo (superficie nitida) com a fusao propria refinada (completa, mais ruidosa).

    python pipeline/uniao_revo_propria.py <fundida_revo.ply> <nuvem_refinada.ply> --saida <pasta> [--dist 1.2] [--min-componente 3000]

Regra: onde o Revo tem superficie, fica a do Revo. Dos pontos da fusao propria, entram so os que ficam a mais de
`--dist` mm de qualquer ponto do Revo (regioes que o Revo descartou por baixa confianca), sem os pedacos pequenos
(componentes de voxels de 2 mm com menos de `--min-componente` pontos). O referencial e o do Revo (ICP rigido leve).
Cada ponto e colorido pela fonte: cinza = Revo, laranja = acrescentado.
"""
import argparse, json, os, sys, time
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade

ap = argparse.ArgumentParser(); ap.add_argument("revo"); ap.add_argument("propria"); ap.add_argument("--saida", required=True)
ap.add_argument("--dist", type=float, default=1.2); ap.add_argument("--min-componente", type=int, default=3000)
a = ap.parse_args(); baixa_prioridade(); os.makedirs(a.saida, exist_ok=True); t0 = time.time()
R = o3d.io.read_point_cloud(a.revo); PR = np.asarray(R.points); NR = np.asarray(R.normals)
S = o3d.io.read_point_cloud(a.propria); PS = np.asarray(S.points); NS = np.asarray(S.normals)
print(f"Revo {len(PR):,} pontos | propria refinada {len(PS):,}", flush=True)
# ICP rigido leve: propria -> Revo
rr = R.voxel_down_sample(1.0); rr.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3, max_nn=30))
ss = S.voxel_down_sample(1.5)
T = o3d.pipelines.registration.registration_icp(ss, rr, 4.0, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(o3d.pipelines.registration.TukeyLoss(0.5))).transformation
ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
print(f"referencial: propria -> Revo desloca {np.round(T[:3,3],2)} mm, {ang:.3f} graus", flush=True)
PS = PS @ T[:3, :3].T + T[:3, 3]; NS = NS @ T[:3, :3].T
d, _ = cKDTree(PR).query(PS, workers=4)
nova = d > a.dist
print(f"da propria, {int(nova.sum()):,} pontos ({nova.mean()*100:.0f}%) estao a > {a.dist} mm do Revo", flush=True)
X = PS[nova]; Nn = NS[nova]
# componentes conexas (voxels de 2 mm): tira pedacos pequenos
v = 2.0; k = np.floor((X - X.min(0)) / v).astype(int); G = np.zeros(k.max(0) + 3, bool); G[tuple((k + 1).T)] = True
lab, n = ndimage.label(G, structure=np.ones((3, 3, 3)))
tam = np.bincount(lab[tuple((k + 1).T)], minlength=n + 1)
bom = tam[lab[tuple((k + 1).T)]] >= a.min_componente
print(f"componentes: {n:,}; removidos {int((~bom).sum()):,} pontos de pedacos < {a.min_componente} pontos; ficam {int(bom.sum()):,}", flush=True)
X, Nn = X[bom], Nn[bom]
P = np.vstack([PR, X]); N = np.vstack([NR, Nn])
cor = np.vstack([np.tile([0.78, 0.78, 0.78], (len(PR), 1)), np.tile([1.0, 0.45, 0.0], (len(X), 1))])
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); pc.normals = o3d.utility.Vector3dVector(N); pc.colors = o3d.utility.Vector3dVector(cor)
o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_completa.ply"), pc, write_ascii=False)
np.save(os.path.join(a.saida, "fonte.npy"), np.r_[np.zeros(len(PR), np.uint8), np.ones(len(X), np.uint8)])
json.dump(dict(revo=int(len(PR)), acrescentados=int(len(X)), dist_mm=a.dist, ICP_mm=T[:3, 3].tolist(), ICP_graus=float(ang)), open(os.path.join(a.saida, "relatorio.json"), "w"))
print(f"nuvem completa: {len(P):,} pontos (Revo {len(PR):,} + acrescentados {len(X):,}) | {time.time()-t0:.0f}s")
