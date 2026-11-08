"""Nuvem de mundo (todos os quadros, pose por quadro escolhida contra o fuse) + utilitarios de plano da mesa."""

import os

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

from pipeline.revo_antigo.dph import (
    intrinsecos,
    listar,
    para_mundo,
    pontos_gpu,
    pose_do_quadro,
    poses_registradas,
    profundidade,
)


def poses_do_scan(pasta, usar_fuse=True):
    """{arquivo: (T, origem)} escolhendo registrada x .inf pelo fuse quando existe."""
    K = intrinsecos(pasta)
    reg = (
        poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    )
    t = None
    fz = os.path.join(pasta, "fuse.ply")
    if usar_fuse and reg and os.path.exists(fz):
        t = gpu.para_gpu(np.asarray(o3d.io.read_point_cloud(fz).points), np.float64)  # pontos do fuse na GPU
    return {f: pose_do_quadro(f, reg, K, t) for f in listar(pasta)}


def nuvem_mundo(pasta, poses, passo=2):
    """(pontos (n, 3), indice do quadro (n,)) numpy; retroprojecao e pose na GPU."""
    K = intrinsecos(pasta)
    P, idx = [], []
    for k, f in enumerate(listar(pasta)):
        d = profundidade(f)
        p, v, u = pontos_gpu(d, K, passo=passo)
        if not len(p):
            continue
        T = poses[f][0]
        P.append(para_mundo(p, T))
        idx.append(np.full(len(p), k))
    return gpu.para_cpu(gpu.xp().concatenate(P)), np.concatenate(idx)


def plano_ransac(P, dist=1.0, n=3000, semente=0):
    """Plano (a, b, c, d) de |normal| = 1 e indices dos inliers, RANSAC na GPU (substitui Open3D segment_plane)."""
    return gg.ransac_plano_open3d(P, limiar=dist, iteracoes=n, semente=semente)
