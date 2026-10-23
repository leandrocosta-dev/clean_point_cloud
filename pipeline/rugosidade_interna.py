"""Rugosidade local (RMS do plano ajustado em raio r) da malha, mediana e p90 - para comparar versoes (a vista do MeshInspector 'aspera' = p90 alto).
    python pipeline/rugosidade_interna.py malha.ply [--raio 2.5] [--n 60000]
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
ap = argparse.ArgumentParser(); ap.add_argument("malha"); ap.add_argument("--raio", type=float, default=2.5); ap.add_argument("--n", type=int, default=60000)
a = ap.parse_args(); m = o3d.io.read_triangle_mesh(a.malha); V = np.asarray(m.vertices); t = cKDTree(V)
idx = np.random.default_rng(0).choice(len(V), min(a.n, len(V)), replace=False); r = []
for i in idx:
    l = t.query_ball_point(V[i], a.raio)
    if len(l) < 8: continue
    Q = V[l] - V[l].mean(0); w = np.linalg.svd(Q, compute_uv=False); r.append(w[2] / np.sqrt(len(l)))
r = np.array(r) * 1000; print("%s: rugosidade (um) mediana %.0f p90 %.0f p99 %.0f" % (a.malha.split("/")[-1], np.median(r), np.percentile(r, 90), np.percentile(r, 99)))
