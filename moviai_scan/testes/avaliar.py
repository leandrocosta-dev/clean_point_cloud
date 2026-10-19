"""Compara a nuvem limpa com a peca verdade (so para o teste sintetico)."""
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

limpo = np.asarray(o3d.io.read_point_cloud(sys.argv[1]).points)
verdade = np.asarray(o3d.io.read_point_cloud(sys.argv[2]).points)

d_out, _ = cKDTree(verdade).query(limpo, workers=-1)
d_cov, _ = cKDTree(limpo).query(verdade, workers=-1)
print(f"pontos limpos: {len(limpo):,}")
print(f"precisao: {np.mean(d_out < 0.2) * 100:.2f}% dos pontos a menos de 0.2 mm da peca real")
print(f"erro mediano: {np.median(d_out):.3f} mm | p99: {np.percentile(d_out, 99):.3f} mm")
print(f"cobertura: {np.mean(d_cov < 0.5) * 100:.2f}% da peca real reconstruida (tolerancia 0.5 mm)")
