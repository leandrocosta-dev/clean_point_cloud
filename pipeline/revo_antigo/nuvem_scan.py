"""Nuvem de um scan do formato antigo a partir dos .dph (limpos): pontos no referencial do scan.

Uso: python pipeline/revo_antigo/nuvem_scan.py <pasta_do_scan> --saida nuvem.ply [--voxel 0.4] [--passo 1]
Pose: global_register_pose.pose se existir (Revo), senao as .inf. Normais orientadas para a camera.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, pose, poses_registradas, profundidade, pontos


def nuvem(pasta, voxel=0.4, passo=1):
    K = intrinsecos(pasta)
    reg = poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    P, C = [], []
    for f in listar(pasta):
        d = profundidade(f)
        p, _, _ = pontos(d, K, passo=passo)
        if not len(p):
            continue
        T = reg[int(f[-8:-4])] if reg else pose(f)
        P.append(p @ T[:3, :3].T + T[:3, 3])
        C.append(np.tile(T[:3, 3], (len(p), 1)))   # centro da camera, para orientar normais
    P, C = np.concatenate(P), np.concatenate(C)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(C - P)  # provisorio: direcao da visada
    pc = pc.voxel_down_sample(voxel)
    vis = np.asarray(pc.normals); vis /= np.linalg.norm(vis, axis=1, keepdims=True) + 1e-12
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 6, max_nn=30))
    n = np.asarray(pc.normals); n[(n * vis).sum(1) < 0] *= -1
    pc.normals = o3d.utility.Vector3dVector(n)
    return pc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta"); ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.4); ap.add_argument("--passo", type=int, default=1)
    a = ap.parse_args()
    pc = nuvem(a.pasta, a.voxel, a.passo)
    o3d.io.write_point_cloud(a.saida, pc)
    print(a.saida, len(pc.points), np.ptp(np.asarray(pc.points), 0).round(1))
