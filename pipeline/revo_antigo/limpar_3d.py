"""Limpeza 3D dos quadros .dph de dois scans ja alinhados: tira pixels sem apoio na nuvem unida.

Uso: python pipeline/revo_antigo/limpar_3d.py <pastaA> <pastaB> --T T_B_para_A.json [--indice 0] [--raio 1.5 --min-viz 12 --dbscan 1.5]
Escreve nos .dph (rode sobre a copia do projeto). Une os pontos dos quadros (A no seu referencial, B levado a A),
voxel 0,3 mm, tira pontos com < min-viz vizinhos a raio e mantem so o maior componente (DBSCAN); pixel cujo ponto
nao tem vizinho mantido a < 0,6 mm vira 0.
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, pose, poses_registradas, profundidade, pontos


def quadros(pasta, T0):
    K = intrinsecos(pasta)
    reg = poses_registradas(pasta) if os.path.exists(os.path.join(pasta, "cache", "global_register_pose.pose")) else None
    for f in listar(pasta):
        d = profundidade(f)
        p, v, u = pontos(d, K)
        if not len(p):
            yield f, d, p, v, u
            continue
        T = T0 @ (reg[int(f[-8:-4])] if reg else pose(f))
        yield f, d, p @ T[:3, :3].T + T[:3, 3], v, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--T", required=True)
    ap.add_argument("--indice", type=int, default=0)
    ap.add_argument("--raio", type=float, default=1.5); ap.add_argument("--min-viz", type=int, default=12)
    ap.add_argument("--dbscan", type=float, default=1.5)
    a = ap.parse_args()
    TB = np.array(json.load(open(a.T))[a.indice]["T"])
    pastas = [(a.a, np.eye(4)), (a.b, TB)]
    todos = []
    for pasta, T0 in pastas:
        for f, d, W, v, u in quadros(pasta, T0):
            if len(W):
                todos.append(W.astype(np.float32))
    P = np.concatenate(todos); print("pontos brutos", len(P))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.3)
    print("voxel 0,3:", len(pc.points))
    pc, _ = pc.remove_radius_outlier(a.min_viz, a.raio)
    lab = np.array(pc.cluster_dbscan(a.dbscan, 10))
    cont = np.bincount(lab[lab >= 0]); big = int(np.argmax(cont))
    print("componentes", len(cont), "maior", cont[big], "total", len(lab))
    Pk = np.asarray(pc.points)[lab == big]
    t = cKDTree(Pk)
    for pasta, T0 in pastas:
        tot = mant = 0
        for f, d, W, v, u in quadros(pasta, T0):
            if not len(W):
                continue
            dist, _ = t.query(W, distance_upper_bound=0.6)
            ok = np.isfinite(dist)
            novo = np.zeros_like(d); novo[v[ok], u[ok]] = d[v[ok], u[ok]]
            tot += len(W); mant += int(ok.sum())
            novo.tofile(f)
        print(os.path.basename(pasta), f"{mant}/{tot} mantidos ({100*mant/tot:.1f}%)")
    o3d.io.write_point_cloud(os.path.join(os.path.dirname(a.T), "nucleo_unido.ply"), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pk)))


if __name__ == "__main__":
    main()
