"""Limpa os .dph de scans em que o proprio Revo ja fundiu so a peca: pixel a > --tol mm do fuse.ply do scan (sem as ilhas soltas da malha) vira 0.

Uso: python pipeline/revo_antigo/limpar_por_fuse.py <projeto> --saida <pasta nova> --scans id1 id2 [--tol 1.5]
Serve para scans que preenchem o campo de visao (sem componentes isolados para separar); a pose por quadro vem de
dph.pose_do_quadro (registrada ou .inf, a que concorda com o fuse). O original nao e tocado.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, poses_registradas, pose_do_quadro, profundidade, pontos

ap = argparse.ArgumentParser()
ap.add_argument("projeto"); ap.add_argument("--saida", required=True); ap.add_argument("--scans", nargs="+", required=True)
ap.add_argument("--tol", type=float, default=1.5)
a = ap.parse_args()
if os.path.exists(a.saida):
    sys.exit("destino ja existe")
shutil.copytree(a.projeto, a.saida)
for s in a.scans:
    p = os.path.join(a.saida, "data", s)
    K = intrinsecos(p); reg = poses_registradas(p)
    # regua = fuse.ply do Revo SEM as ilhas: pontos do fuse mais perto de um componente solto da malha do que do maior
    # (ilhas soltas, ex.: 183931, sao lixo; a malha sozinha nao serve de regua porque tem furos onde ha pontos reais)
    F = np.asarray(o3d.io.read_point_cloud(os.path.join(p, "fuse.ply")).points)
    m = o3d.io.read_triangle_mesh(os.path.join(p, "fuse_mesh.ply")); cl, n, _ = m.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
    big = int(np.argmax(n))
    def amostra(mask):
        q = o3d.geometry.TriangleMesh(m); q.remove_triangles_by_mask(mask); q.remove_unreferenced_vertices()
        return cKDTree(np.asarray(q.sample_points_uniformly(max(20000, int(len(q.triangles) * 2))).points))
    dprin = amostra(cl != big).query(F)[0]
    if (n >= 1).sum() > 1:
        dilha = amostra(cl == big).query(F)[0]
        F = F[~((dilha < dprin) & (dilha < 3.0))]
    print(f"{s}: fuse {len(F)} pontos de referencia (componentes da malha: {sorted(n.tolist(), reverse=True)[:4]})")
    t = cKDTree(F)
    tot = mant = 0; usou = {"reg": 0, "inf": 0}
    for f in listar(p):
        d = profundidade(f); T, q = pose_do_quadro(f, reg, K, t); usou[q] += 1
        pts, v, u = pontos(d, K)
        if not len(pts):
            continue
        ok = t.query(pts @ T[:3, :3].T + T[:3, 3], distance_upper_bound=a.tol)[0] < np.inf
        novo = np.zeros_like(d); novo[v[ok], u[ok]] = d[v[ok], u[ok]]; novo.tofile(f)
        tot += len(pts); mant += int(ok.sum())
    print(f"{s}: {mant}/{tot} pixels mantidos ({100*mant/tot:.2f}%); poses usadas {usou}", flush=True)
