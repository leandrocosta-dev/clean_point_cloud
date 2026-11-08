"""Une as nuvens (dos .dph ja limpos) de A e B (B levado a A por T) numa nuvem com normais orientadas pela visada.

Uso: python pipeline/revo_antigo/nuvem_unida.py <pastaA> <pastaB> --T T.json --saida unida.ply [--voxel 0.3]
"""

import argparse
import json

import geom_gpu as gg
import numpy as np
import open3d as o3d

from pipeline.revo_antigo.nuvem_scan import nuvem_arrays

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--T", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--voxel", type=float, default=0.3)
ap.add_argument("--indice", type=int, default=0)
a = ap.parse_args()
T = np.array(json.load(open(a.T))[a.indice]["T"])
Pa, Na = nuvem_arrays(a.a, a.voxel)
Pb, Nb = nuvem_arrays(a.b, a.voxel)
Pb, Nb = gg.transformar(Pb, T), Nb @ T[:3, :3].T  # B levado a A (GPU)
P, N = gg.reduzir_voxel_normais(
    np.concatenate([Pa, Pb]), np.concatenate([Na, Nb]), a.voxel
)  # voxel + normais medias (GPU)
u = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
u.normals = o3d.utility.Vector3dVector(N)
o3d.io.write_point_cloud(a.saida, u)
print(a.saida, len(u.points))
