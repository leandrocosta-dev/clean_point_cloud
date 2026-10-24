"""Refina as poses por quadro de um scan (formato antigo) contra a malha TSDF do proprio scan e refaz a malha.

Uso: python pipeline/revo_antigo/refinar_poses.py <pasta_do_scan> --saida malha_refinada.ply [--rodadas 2 --voxel 0.4]
Rodada: malha TSDF -> amostra com normais -> ICP ponto-a-plano de cada quadro (pontos da imagem, passo 4, dist. max 1 mm, 25 it)
-> T = delta @ T -> nova TSDF. Imprime deslocamento medio/maximo dos quadros e a rugosidade nao e calculada aqui (use rugosidade_interna.py).
"""
import argparse
import os
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, pose, poses_registradas, profundidade, pontos
from tsdf_scan import malha_tsdf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pasta"); ap.add_argument("--saida", required=True)
    ap.add_argument("--rodadas", type=int, default=2); ap.add_argument("--voxel", type=float, default=0.4); ap.add_argument("--dist", type=float, default=1.0)
    a = ap.parse_args()
    K = intrinsecos(a.pasta); fs = listar(a.pasta)
    reg = poses_registradas(a.pasta) if os.path.exists(os.path.join(a.pasta, "cache", "global_register_pose.pose")) else None
    poses = {f: (reg[int(f[-8:-4])] if reg else pose(f)).copy() for f in fs}
    for r in range(a.rodadas):
        m, _ = malha_tsdf(a.pasta, a.voxel, poses=poses)
        alvo = m.sample_points_uniformly(max(300000, len(m.triangles) * 2)); alvo.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        desl = []
        for f in fs:
            d = profundidade(f)
            if not d.any():
                continue
            p, _, _ = pontos(d, K, passo=4); T = poses[f]
            src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p @ T[:3, :3].T + T[:3, 3]))
            res = o3d.pipelines.registration.registration_icp(src, alvo, a.dist, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
            if res.fitness < 0.3:
                continue
            poses[f] = res.transformation @ T; desl.append(np.linalg.norm(res.transformation[:3, 3]))
        print(f"rodada {r+1}: {len(desl)} quadros ajustados, deslocamento medio {np.mean(desl):.3f} mm, max {np.max(desl):.2f} mm", flush=True)
    m, n = malha_tsdf(a.pasta, a.voxel, poses=poses)
    o3d.io.write_triangle_mesh(a.saida, m, write_vertex_normals=False)
    np.save(a.saida.replace(".ply", "_poses.npy"), np.stack([poses[f] for f in fs]))
    print(f"{a.saida}: {n} quadros, {len(m.triangles)} triangulos, area {m.get_surface_area():.0f} mm2")


if __name__ == "__main__":
    main()
