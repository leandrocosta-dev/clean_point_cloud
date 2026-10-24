"""Malha de um scan (formato antigo) por fusao TSDF dos .dph (use os limpos): nao fecha o que nao foi medido.

Uso: python pipeline/revo_antigo/tsdf_scan.py <pasta_do_scan> --saida malha.ply [--voxel 0.4 --trunc 1.6 --max-dist 400]
Pose por quadro: registrada (global_register_pose) se existir senao .inf. Extrinseca Open3D = S . inv(T), S = diag(1,-1,-1)
(convencao de camera do Revo antigo: (x,-y,-z)). Mantem componentes >= 5% do maior.
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, pose, poses_registradas, profundidade


def malha_tsdf(pasta, voxel=0.4, trunc=1.6, max_dist=400.0, peso_min=2.0, poses=None):
    """VoxelBlockGrid (a API legada ScalableTSDFVolume nao gera nada neste Open3D 0.20; e o extract_triangle_mesh() da API de
    tensores descarta voxels com peso < 3 por padrao: com poucos quadros a malha sai vazia -> passar weight_threshold)."""
    import open3d.core as o3c
    K = intrinsecos(pasta); f, cx, cy, W0, esc = K
    reg = poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    dev = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
                                        voxel, 16, 80000, dev)
    S = np.diag([1.0, -1.0, -1.0, 1.0]); n = 0
    for fq in listar(pasta):
        d = profundidade(fq)
        if not d.any():
            continue
        k = d.shape[1] / W0
        intr = o3c.Tensor(np.array([[f * k, 0, cx * k], [0, f * k, cy * k], [0, 0, 1]], np.float64))
        T = poses[fq] if poses is not None else (reg[int(fq[-8:-4])] if reg else pose(fq))
        ext = o3c.Tensor(S @ np.linalg.inv(T))
        depth = o3d.t.geometry.Image(o3c.Tensor((d.astype(np.float32) * esc)[:, :, None]))
        color = o3d.t.geometry.Image(o3c.Tensor(np.zeros((d.shape[0], d.shape[1], 3), np.float32)))
        bc = vbg.compute_unique_block_coordinates(depth, intr, ext, 1.0, max_dist, trunc / voxel)
        vbg.integrate(bc, depth, color, intr, ext, 1.0, max_dist, trunc / voxel); n += 1
    m = vbg.extract_triangle_mesh(weight_threshold=peso_min).to_legacy()
    if not len(m.triangles):
        return m, n
    cl, nt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); nt = np.asarray(nt)
    m.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(nt < 0.05 * nt.max()))); m.remove_unreferenced_vertices()
    return m, n


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("pasta"); ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.4); ap.add_argument("--trunc", type=float, default=1.6); ap.add_argument("--max-dist", type=float, default=400.0); ap.add_argument("--peso-min", type=float, default=2.0)
    a = ap.parse_args()
    m, n = malha_tsdf(a.pasta, a.voxel, a.trunc, a.max_dist, a.peso_min)
    o3d.io.write_triangle_mesh(a.saida, m, write_vertex_normals=False)
    print(f"{a.saida}: {n} quadros, {len(m.triangles)} triangulos, area {m.get_surface_area():.0f} mm2, ext {np.ptp(np.asarray(m.vertices),0).round()}")


def malha_tsdf_varios(scans, voxel=0.3, trunc=1.2, max_dist=400.0, peso_min=3.0):
    """TSDF unica com os quadros de varios scans: scans = [(pasta, T_scan_para_referencia 4x4)]. So faz sentido se os scans estao
    alinhados a ~0,1-0,2 mm (verificar antes: mediana scan x scan); senao a media borra a parede."""
    import open3d.core as o3c
    dev = o3c.Device("CPU:0")
    vbg = o3d.t.geometry.VoxelBlockGrid(("tsdf", "weight", "color"), (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)), voxel, 16, 200000, dev)
    S = np.diag([1.0, -1.0, -1.0, 1.0]); n = 0
    for pasta, Ts in scans:
        K = intrinsecos(pasta); f, cx, cy, W0, esc = K
        reg = poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
        for fq in listar(pasta):
            d = profundidade(fq)
            if not d.any():
                continue
            k = d.shape[1] / W0
            intr = o3c.Tensor(np.array([[f * k, 0, cx * k], [0, f * k, cy * k], [0, 0, 1]], np.float64))
            T = Ts @ (reg[int(fq[-8:-4])] if reg else pose(fq)); ext = o3c.Tensor(S @ np.linalg.inv(T))
            depth = o3d.t.geometry.Image(o3c.Tensor((d.astype(np.float32) * esc)[:, :, None]))
            color = o3d.t.geometry.Image(o3c.Tensor(np.zeros((d.shape[0], d.shape[1], 3), np.float32)))
            bc = vbg.compute_unique_block_coordinates(depth, intr, ext, 1.0, max_dist, trunc / voxel)
            vbg.integrate(bc, depth, color, intr, ext, 1.0, max_dist, trunc / voxel); n += 1
    m = vbg.extract_triangle_mesh(weight_threshold=peso_min).to_legacy()
    cl, nt, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); nt = np.asarray(nt)
    m.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(nt < 0.02 * nt.max()))); m.remove_unreferenced_vertices()
    return m, n
