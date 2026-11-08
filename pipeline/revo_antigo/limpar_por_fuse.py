"""Limpa os .dph de scans em que o proprio Revo ja fundiu so a peca: pixel a > --tol mm do fuse.ply do scan (sem as ilhas soltas da malha) vira 0.

Uso: python pipeline/revo_antigo/limpar_por_fuse.py <projeto> --saida <pasta nova> --scans id1 id2 [--tol 1.5]
Serve para scans que preenchem o campo de visao (sem componentes isolados para separar); a pose por quadro vem de
dph.pose_do_quadro (registrada ou .inf, a que concorda com o fuse). O original nao e tocado.
"""

import argparse
import os
import shutil
import sys

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

ap = argparse.ArgumentParser()
ap.add_argument("projeto")
ap.add_argument("--saida", required=True)
ap.add_argument("--scans", nargs="+", required=True)
ap.add_argument("--tol", type=float, default=1.5)
a = ap.parse_args()
if os.path.exists(a.saida):
    sys.exit("destino ja existe")
shutil.copytree(a.projeto, a.saida)
for s in a.scans:
    p = os.path.join(a.saida, "data", s)
    K = intrinsecos(p)
    reg = poses_registradas(p)
    # regua = fuse.ply do Revo SEM as ilhas: pontos do fuse mais perto de um componente solto da malha do que do maior
    # (ilhas soltas, ex.: 183931, sao lixo; a malha sozinha nao serve de regua porque tem furos onde ha pontos reais)
    F = gpu.para_gpu(np.asarray(o3d.io.read_point_cloud(os.path.join(p, "fuse.ply")).points), np.float64)
    m = o3d.io.read_triangle_mesh(os.path.join(p, "fuse_mesh.ply"))
    V = np.asarray(m.vertices)
    Tri = np.asarray(m.triangles)
    cl, n = gg.componentes_triangulos(Tri, len(V))
    n = np.asarray(n)  # componente 0 = o maior
    big = 0

    def amostra(mask):
        """Amostra de pontos (GPU) das superficies dos triangulos com mask False."""
        sel = Tri[~mask]
        return gg.amostrar_malha(V, sel, max(20000, int(len(sel) * 2)), saida_gpu=True)[0]

    dprin = gg.distancia_ate(amostra(cl != big), F, saida_gpu=True)
    if (n >= 1).sum() > 1:
        dilha = gg.distancia_ate(amostra(cl == big), F, saida_gpu=True)
        F = F[~((dilha < dprin) & (dilha < 3.0))]
    print(f"{s}: fuse {len(F)} pontos de referencia (componentes da malha: {sorted(n.tolist(), reverse=True)[:4]})")
    tot = mant = 0
    usou = {"reg": 0, "inf": 0}
    for f in listar(p):
        d = gpu.para_gpu(profundidade(f))
        T, q = pose_do_quadro(f, reg, K, F)
        usou[q] += 1
        pts, v, u = pontos_gpu(d, K)
        if not len(pts):
            continue
        ok = gpu.vizinho_mais_proximo(F, para_mundo(pts, T), dist_max=a.tol)[1] >= 0
        novo = gpu.xp().zeros_like(d)
        novo[v[ok], u[ok]] = d[v[ok], u[ok]]
        gpu.para_cpu(novo).tofile(f)
        tot += len(pts)
        mant += int(ok.sum())
    print(f"{s}: {mant}/{tot} pixels mantidos ({100 * mant / tot:.2f}%); poses usadas {usou}", flush=True)
