"""Refina as poses por quadro de um scan (formato antigo) contra a malha TSDF do proprio scan e refaz a malha.

Uso: python pipeline/revo_antigo/refinar_poses.py <pasta_do_scan> --saida malha_refinada.ply [--rodadas 2 --voxel 0.4]
Rodada: malha TSDF -> amostra com normais -> ICP ponto-a-plano de cada quadro (pontos da imagem, passo 4, dist. max 1 mm, 25 it)
-> T = delta @ T -> nova TSDF. Imprime deslocamento medio/maximo dos quadros e a rugosidade nao e calculada aqui (use rugosidade_interna.py).
"""

import argparse
import os

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

from pipeline.revo_antigo.dph import intrinsecos, listar, para_mundo, pontos_gpu, pose, poses_registradas, profundidade
from pipeline.revo_antigo.tsdf_scan import malha_tsdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pasta")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--rodadas", type=int, default=2)
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--dist", type=float, default=1.0)
    a = ap.parse_args()
    K = intrinsecos(a.pasta)
    fs = listar(a.pasta)
    reg = (
        poses_registradas(a.pasta)
        if os.path.exists(os.path.join(a.pasta, "cache", "global_register_pose.pose"))
        else None
    )
    poses = {f: (reg[int(f[-8:-4])] if reg else pose(f)).copy() for f in fs}
    for r in range(a.rodadas):
        m, _ = malha_tsdf(a.pasta, a.voxel, poses=poses)
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        alvo, nt = gg.amostrar_malha(V, F, max(300000, len(F) * 2), saida_gpu=True)
        Na = gg.normais_hibrido(alvo, 2.0, 30, saida_gpu=True)
        bom = ~gpu.xp().isnan(Na[:, 0])  # normais PCA (GPU)
        alvo, Na = alvo[bom], Na[bom]
        desl = []
        for f in fs:
            d = profundidade(f)
            if not d.any():
                continue
            p, _, _ = pontos_gpu(d, K, passo=4)
            T = poses[f]
            Tr, info = gpu.icp_ponto_plano(
                para_mundo(p, T), alvo, Na, T0=np.eye(4), dist_max=a.dist, iteracoes=25, perda="l2"
            )  # ICP ponto-plano (GPU)
            if info["fitness"] < 0.3:
                continue
            poses[f] = Tr @ T
            desl.append(np.linalg.norm(Tr[:3, 3]))
        print(
            f"rodada {r + 1}: {len(desl)} quadros ajustados, deslocamento medio {np.mean(desl):.3f} mm, max {np.max(desl):.2f} mm",
            flush=True,
        )
    m, n = malha_tsdf(a.pasta, a.voxel, poses=poses)
    o3d.io.write_triangle_mesh(a.saida, m, write_vertex_normals=False)
    np.save(a.saida.replace(".ply", "_poses.npy"), np.stack([poses[f] for f in fs]))
    print(
        f"{a.saida}: {n} quadros, {len(m.triangles)} triangulos, area {gg.area_malha(np.asarray(m.vertices), np.asarray(m.triangles)):.0f} mm2"
    )


if __name__ == "__main__":
    main()
