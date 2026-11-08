"""Limpeza 3D dos quadros .dph de dois scans ja alinhados: tira pixels sem apoio na nuvem unida.

Uso: python pipeline/revo_antigo/limpar_3d.py <pastaA> <pastaB> --T T_B_para_A.json [--indice 0] [--raio 1.5 --min-viz 12 --dbscan 1.5]
Escreve nos .dph (rode sobre a copia do projeto). Une os pontos dos quadros (A no seu referencial, B levado a A),
voxel 0,3 mm, tira pontos com < min-viz vizinhos a raio e mantem so o maior componente (DBSCAN); pixel cujo ponto
nao tem vizinho mantido a < 0,6 mm vira 0.
"""

import argparse
import json
import os

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

from pipeline.revo_antigo.dph import intrinsecos, listar, para_mundo, pontos_gpu, pose, poses_registradas, profundidade


def quadros(pasta, T0):
    """Gera (arquivo, profundidade cupy, pontos no referencial T0 cupy, v, u cupy); retroprojecao e pose na GPU."""
    K = intrinsecos(pasta)
    reg = (
        poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    )
    for f in listar(pasta):
        d = gpu.para_gpu(profundidade(f))
        p, v, u = pontos_gpu(d, K)
        if not len(p):
            yield f, d, p, v, u
            continue
        T = T0 @ (reg[int(f[-8:-4])] if reg else pose(f))
        yield f, d, para_mundo(p, T), v, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--T", required=True)
    ap.add_argument("--indice", type=int, default=0)
    ap.add_argument("--raio", type=float, default=1.5)
    ap.add_argument("--min-viz", type=int, default=12)
    ap.add_argument("--dbscan", type=float, default=1.5)
    a = ap.parse_args()
    TB = np.array(json.load(open(a.T))[a.indice]["T"])
    pastas = [(a.a, np.eye(4)), (a.b, TB)]
    acc = gg.AcumuladorVoxel(0.3)  # voxel 0,3 mm em fluxo (nuvem unida nunca inteira na memoria)
    bruto = 0
    for pasta, T0 in pastas:
        for f, d, W, v, u in quadros(pasta, T0):
            if len(W):
                acc.add(W.astype(gpu.xp().float32))
                bruto += len(W)
    print("pontos brutos", bruto)
    C, _, _ = acc.resultado(saida_gpu=True)
    print("voxel 0,3:", len(C))
    C = C[
        gpu.contar_no_raio(C, C, a.raio) > a.min_viz
    ]  # remove_radius_outlier: mantem se contagem (com o proprio) > min-viz
    lab = gg.dbscan(C, a.dbscan, 10)
    cont = np.bincount(lab[lab >= 0])
    big = int(np.argmax(cont))
    print("componentes", len(cont), "maior", cont[big], "total", len(lab))
    Pk = C[gpu.para_gpu(lab == big)]
    for pasta, T0 in pastas:
        tot = mant = 0
        for f, d, W, v, u in quadros(pasta, T0):
            if not len(W):
                continue
            dist, j = gpu.vizinho_mais_proximo(Pk, W, dist_max=0.6)
            ok = j >= 0
            novo = gpu.xp().zeros_like(d)
            novo[v[ok], u[ok]] = d[v[ok], u[ok]]
            tot += len(W)
            mant += int(ok.sum())
            gpu.para_cpu(novo).tofile(f)
        print(os.path.basename(pasta), f"{mant}/{tot} mantidos ({100 * mant / tot:.1f}%)")
    o3d.io.write_point_cloud(
        os.path.join(os.path.dirname(a.T), "nucleo_unido.ply"),
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(Pk))),
    )


if __name__ == "__main__":
    main()
