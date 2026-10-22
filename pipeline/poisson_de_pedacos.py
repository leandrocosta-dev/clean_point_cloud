"""Costura a malha por cobertura (base + pedacos) com Poisson: amostra pontos LISOS (um scan por regiao) com normais p/ fora e reconstroi.

    python pipeline/poisson_de_pedacos.py <base_e_pedacos.ply> --saida <malha.ply> [--amostras 6000000] [--profundidade 11] [--dist-max 1.5]

As emendas entre pedacos de scans diferentes tem degraus de ~0,5-1 mm (deriva de cada scan); o Poisson junta tudo numa superficie so,
sem duplicar camadas, e como cada regiao vem de UM scan (ruido 10-30 um) a malha final fica lisa. Gera tambem <saida>_reparada.ply (MeshLib).
"""
import argparse, os, sys, time
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from malha_meshlib import metricas, reparar
from meshlib import mrmeshpy as mm

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("malha"); ap.add_argument("--saida", required=True)
ap.add_argument("--amostras", type=int, default=6_000_000); ap.add_argument("--profundidade", type=int, default=11); ap.add_argument("--dist-max", type=float, default=1.5)
a = ap.parse_args(); t0 = time.time()
M = o3d.io.read_triangle_mesh(a.malha); M.compute_triangle_normals()
pc = M.sample_points_uniformly(a.amostras, use_triangle_normal=True)
print(f"{len(pc.points):,} pontos amostrados, normais p/ fora ({time.time()-t0:.0f}s)", flush=True)
P = np.asarray(pc.points)
R, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=a.profundidade, width=0, scale=1.05, linear_fit=False, n_threads=8)
dens = np.asarray(dens); V = np.asarray(R.vertices); d, _ = cKDTree(P).query(V, workers=8)
R.remove_vertices_by_mask((dens < np.quantile(dens, 0.005)) | (d > a.dist_max)); R.remove_degenerate_triangles(); R.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh(a.saida, R, write_vertex_normals=False)
ml = mm.loadMesh(a.saida); print("MeshLib antes :", metricas(ml)); reparar(ml, min_faces=500, furo_max=6.0); print("MeshLib depois:", metricas(ml))
mm.saveMesh(ml, a.saida.replace(".ply", "_reparada.ply")); print(f"pronto ({time.time()-t0:.0f}s)")
