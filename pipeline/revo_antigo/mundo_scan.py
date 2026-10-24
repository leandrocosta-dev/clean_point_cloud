"""Nuvem de mundo (todos os quadros, pose por quadro escolhida contra o fuse) + utilitarios de plano da mesa."""
import os
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, poses_registradas, pose, pose_do_quadro, profundidade, pontos


def poses_do_scan(pasta, usar_fuse=True):
    """{arquivo: (T, origem)} escolhendo registrada x .inf pelo fuse quando existe."""
    K = intrinsecos(pasta)
    reg = poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    t = None
    fz = os.path.join(pasta, "fuse.ply")
    if usar_fuse and reg and os.path.exists(fz):
        t = cKDTree(np.asarray(o3d.io.read_point_cloud(fz).points))
    return {f: pose_do_quadro(f, reg, K, t) for f in listar(pasta)}


def nuvem_mundo(pasta, poses, passo=2):
    K = intrinsecos(pasta); P, idx = [], []
    for k, f in enumerate(listar(pasta)):
        d = profundidade(f); p, v, u = pontos(d, K, passo=passo)
        if not len(p):
            continue
        T = poses[f][0]; P.append(p @ T[:3, :3].T + T[:3, 3]); idx.append(np.full(len(p), k))
    return np.concatenate(P), np.concatenate(idx)


def plano_ransac(P, dist=1.0, n=3000, semente=0):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    m, inl = pc.segment_plane(dist, 3, n)
    return np.array(m), np.array(inl)
