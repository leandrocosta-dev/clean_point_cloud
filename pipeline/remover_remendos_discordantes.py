"""Remove os remendos (faces novas apos as N0 originais) cuja maioria das faces discorda das normais da nuvem externa: viram verso.

    python pipeline/remover_remendos_discordantes.py malha.ply nuvem_externa.ply --n0 1327032 --saida out.ply
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("malha"); ap.add_argument("nuvem"); ap.add_argument("--n0", type=int, required=True); ap.add_argument("--saida", required=True)
a = ap.parse_args()
o = o3d.io.read_triangle_mesh(a.malha); V = np.asarray(o.vertices); T = np.asarray(o.triangles)
nu = o3d.io.read_point_cloud(a.nuvem); tr = o3d.geometry.KDTreeFlann(nu); NN = np.asarray(nu.normals)
Tn = T[a.n0:]
E = np.sort(np.concatenate([Tn[:, [0, 1]], Tn[:, [1, 2]], Tn[:, [2, 0]]]), axis=1); fid = np.tile(np.arange(len(Tn)), 3)
key = E[:, 0].astype(np.int64) * 10**7 + E[:, 1]; od = np.argsort(key); k = key[od]; f = fid[od]
s = k[1:] == k[:-1]; x, y = f[:-1][s], f[1:][s]
n, lab = connected_components(coo_matrix((np.ones(len(x)), (x, y)), shape=(len(Tn),) * 2), directed=False)
A, B, C = V[Tn[:, 0]], V[Tn[:, 1]], V[Tn[:, 2]]; nf = np.cross(B - A, C - A); nf /= np.linalg.norm(nf, axis=1, keepdims=True) + 1e-12
cen = (A + B + C) / 3
dot = np.array([np.dot(nf[i], NN[tr.search_knn_vector_3d(cen[i], 1)[1][0]]) for i in range(len(Tn))])
rm = np.array([(dot[lab == c] < 0).mean() > 0.5 for c in range(n)])[lab]
print("remendos", n, "faces removidas", int(rm.sum()), "de", len(Tn))
o2 = o3d.geometry.TriangleMesh(o.vertices, o3d.utility.Vector3iVector(np.concatenate([T[:a.n0], Tn[~rm]])))
o2.remove_unreferenced_vertices(); o3d.io.write_triangle_mesh(a.saida, o2)
