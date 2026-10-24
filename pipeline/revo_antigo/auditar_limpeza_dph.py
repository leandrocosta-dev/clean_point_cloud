"""Audita a regra de limpeza por scan: mantidos/descartados x fuse.ply do Revo (amostra de quadros).

Uso: python pipeline/revo_antigo/auditar_limpeza_dph.py <projeto> [--scans ...] [--n 12]
Imprime por scan: % de pixels mantidos, % dos mantidos a < 1,5 mm do fuse, % dos descartados a < 1,5 mm do fuse
(descartado perto do fuse = superficie boa perdida). Convencao (x,-y,-z) so validada p/ o scanner A23C06525D6I01M17.
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
from limpar_dph import mascara_peca

ap = argparse.ArgumentParser(); ap.add_argument("projeto"); ap.add_argument("--scans", nargs="*"); ap.add_argument("--n", type=int, default=12)
a = ap.parse_args()
data = os.path.join(a.projeto, "data")
for s in (a.scans or sorted(os.listdir(data))):
    p = os.path.join(data, s); fz = os.path.join(p, "fuse.ply")
    prop = json.load(open(os.path.join(p, "property.rvproj")))
    fs = listar(p)
    if not fs or not os.path.exists(fz):
        print(s, "sem quadros ou sem fuse.ply:", len(fs), "quadros"); continue
    K = intrinsecos(p); sc = prop["scan_param"]["depth_scale"]
    reg = poses_registradas(p) if os.path.exists(os.path.join(p, "cache", "global_register_pose.pose")) else None
    t = cKDTree(np.asarray(o3d.io.read_point_cloud(fz).points))
    tot = mant = pm = nd = pd = 0
    for f in fs[::max(1, len(fs) // a.n)]:
        d = profundidade(f); m = mascara_peca(d)
        pts, v, u = pontos(d, K, escala=sc)
        if not len(pts): continue
        T = reg[int(f[-8:-4])] if reg else pose(f)
        dist, _ = t.query(pts @ T[:3, :3].T + T[:3, 3]); k = m[v, u]; perto = dist < 1.5
        tot += len(pts); mant += k.sum(); pm += (perto & k).sum(); nd += (~k).sum(); pd += (perto & ~k).sum()
    print(f"{s}: f={K[0]:.0f} quadros={len(fs)} mantidos {100*mant/tot:.1f}% | mantidos perto do fuse {100*pm/max(mant,1):.1f}% | descartados perto do fuse {100*pd/max(nd,1):.1f}% ({pd} px)", flush=True)
