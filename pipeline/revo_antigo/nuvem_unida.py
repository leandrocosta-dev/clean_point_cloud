"""Une as nuvens (dos .dph ja limpos) de A e B (B levado a A por T) numa nuvem com normais orientadas pela visada.

Uso: python pipeline/revo_antigo/nuvem_unida.py <pastaA> <pastaB> --T T.json --saida unida.ply [--voxel 0.3]
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nuvem_scan import nuvem

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--T", required=True); ap.add_argument("--saida", required=True)
ap.add_argument("--voxel", type=float, default=0.3); ap.add_argument("--indice", type=int, default=0)
a = ap.parse_args()
T = np.array(json.load(open(a.T))[a.indice]["T"])
pa, pb = nuvem(a.a, a.voxel), nuvem(a.b, a.voxel)
pb.transform(T)
u = pa + pb
u = u.voxel_down_sample(a.voxel)
n = np.asarray(u.normals); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
u.normals = o3d.utility.Vector3dVector(n)
o3d.io.write_point_cloud(a.saida, u)
print(a.saida, len(u.points))
