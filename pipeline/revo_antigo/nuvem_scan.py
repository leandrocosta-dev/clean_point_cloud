"""Nuvem de um scan do formato antigo a partir dos .dph (limpos): pontos no referencial do scan.

Uso: python pipeline/revo_antigo/nuvem_scan.py <pasta_do_scan> --saida nuvem.ply [--voxel 0.4] [--passo 1]
Pose: global_register_pose.pose se existir (Revo), senao as .inf. Normais orientadas para a camera.
"""

import argparse
import os

import geom_gpu as gg
import numpy as np
import open3d as o3d

from pipeline.revo_antigo.dph import intrinsecos, listar, para_mundo, pontos_gpu, pose, poses_registradas, profundidade


def nuvem_arrays(pasta, voxel=0.4, passo=1):
    """(P (n, 3), N (n, 3)) float64 numpy: pontos decimados por voxel e normais orientadas para a camera.
    Retroprojecao, decimacao (voxel, fluxo quadro a quadro) e normais (PCA, 30 vizinhos a <= 6 voxels) na GPU."""
    K = intrinsecos(pasta)
    reg = (
        poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    )
    acc = gg.AcumuladorVoxel(voxel)
    for f in listar(pasta):
        d = profundidade(f)
        p, _, _ = pontos_gpu(d, K, passo=passo)
        if not len(p):
            continue
        T = reg[int(f[-8:-4])] if reg else pose(f)
        W = para_mundo(p, T)
        acc.add(
            W, extra=gg.gpu.xp().broadcast_to(gg.gpu.para_gpu(T[:3, 3]), W.shape)
        )  # centro da camera, para orientar normais
    C, _, Cam = acc.resultado(saida_gpu=True)
    m = gg.gpu.xp()
    vis = Cam - C  # direcao media da visada em cada voxel
    vis /= m.linalg.norm(vis, axis=1, keepdims=True) + 1e-12
    n = gg.normais_hibrido(C, voxel * 6, 30, saida_gpu=True)
    n = m.where(m.isnan(n[:, :1]), vis, n)  # sem vizinhos suficientes: fica a visada (como o Open3D)
    n = m.where(((n * vis).sum(1) < 0)[:, None], -n, n)
    return gg.gpu.para_cpu(C), gg.gpu.para_cpu(n)


def nuvem(pasta, voxel=0.4, passo=1):
    """open3d PointCloud com normais (para escrita/Poisson); o calculo e `nuvem_arrays` (GPU)."""
    P, N = nuvem_arrays(pasta, voxel, passo)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    return pc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--passo", type=int, default=1)
    a = ap.parse_args()
    pc = nuvem(a.pasta, a.voxel, a.passo)
    o3d.io.write_point_cloud(a.saida, pc)
    print(a.saida, len(pc.points), np.ptp(np.asarray(pc.points), 0).round(1))
