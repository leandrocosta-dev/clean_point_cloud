"""Rugosidade local (RMS do plano ajustado em raio r) da malha, mediana e p90 - para comparar versoes (a vista do MeshInspector 'aspera' = p90 alto).
python pipeline/rugosidade_interna.py malha.ply [--raio 2.5] [--n 60000]
"""

import argparse
import sys

import numpy as np
import open3d as o3d
from metricas import rugosidade

# GPU: rugosidade em lote (metricas.rugosidade: covariancia por soma dos vizinhos no raio + autovalores 3x3 de todos os pontos).
# CPU so a leitura da malha. Antes: laco Python com query_ball_point + SVD por vertice.
ap = argparse.ArgumentParser()
ap.add_argument("malha")
ap.add_argument("--raio", type=float, default=2.5)
ap.add_argument("--n", type=int, default=60000)
a = ap.parse_args()
m = o3d.io.read_triangle_mesh(a.malha)
V = np.asarray(m.vertices)
r = rugosidade(V, raio=a.raio, n=a.n, min_viz=8)
if not r.get("n"):
    sys.exit("nenhum vertice com >= 8 vizinhos no raio")
print(
    "%s: rugosidade (um) mediana %.0f p90 %.0f p99 %.0f"
    % (a.malha.split("/")[-1], r["mediana_um"], r["p90_um"], r["p99_um"])
)
