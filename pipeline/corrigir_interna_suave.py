"""Corrige a posicao da interna para a espessura medida com um campo de deslocamento MUITO SUAVE (grade grossa), sem criar rugosidade.

    python pipeline/corrigir_interna_suave.py interna.ply externa.ply nuvem_externa.ply --espessura 2.0 --saida out.ply [--celula 20] [--sigma 1.5] [--passadas 3]

Substitui `corrigir_interna_local.py` (mediana por vertice em raios de 5-25 mm injetou rugosidade: a interna ficou "aspera"). Aqui: delta = -(s + espessura)
por vertice de pele -> mediana por celula de --celula mm -> suavizacao gaussiana da grade (sigma em celulas, normalizada pelas celulas validas) ->
interpolacao trilinear por vertice -> deslocamento ao longo da normal externa. O campo varia devagar, entao o relevo/rugosidade original dos scans nao muda.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna"); ap.add_argument("externa"); ap.add_argument("nuvem"); ap.add_argument("--saida", required=True)
ap.add_argument("--espessura", type=float, default=2.0); ap.add_argument("--celula", type=float, default=20.0); ap.add_argument("--sigma", type=float, default=1.5)
ap.add_argument("--passadas", type=int, default=3); ap.add_argument("--perto", type=float, default=8.0)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna); X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem); Pn = np.asarray(E.points); Nn = np.asarray(E.normals)
P = np.asarray(X.sample_points_uniformly(2_000_000).points); _, j = cKDTree(Pn).query(P, workers=8); N = Nn[j]; t = cKDTree(P)
V = np.asarray(I.vertices).copy()
lo = V.min(0) - 2 * a.celula; hi = V.max(0) + 2 * a.celula
for it in range(a.passadas):
    d, k = t.query(V, workers=8); n = N[k]; s = np.einsum("ij,ij->i", V - P[k], n)
    pele = (d < a.perto) & (np.abs(s + a.espessura) < a.perto); delta = -(s + a.espessura)
    ij = np.floor((V[pele] - lo) / a.celula).astype(int); shp = np.ceil((hi - lo) / a.celula).astype(int) + 1
    flat = np.ravel_multi_index(ij.T, shp); cnt = np.bincount(flat, minlength=shp.prod())
    soma = np.zeros(shp.prod()); np.add.at(soma, flat, delta[pele])
    # mediana por celula e aproximada pela media aparada: usa a media (suavizada depois); celulas com <50 vertices ficam vazias
    ok = cnt >= 50; g = np.zeros(shp.prod()); g[ok] = soma[ok] / cnt[ok]; g = g.reshape(shp); w = ok.reshape(shp).astype(float)
    gs = ndi.gaussian_filter(g * w, a.sigma); ws = ndi.gaussian_filter(w, a.sigma); campo = np.where(ws > 1e-3, gs / np.maximum(ws, 1e-3), 0.0)
    eixos = [lo[i] + a.celula * (np.arange(shp[i]) + 0.5) for i in range(3)]
    f = RegularGridInterpolator(eixos, campo, bounds_error=False, fill_value=0.0)
    dv = f(V); dv[d > 30] = 0.0
    V = V + dv[:, None] * n
    print(f"passada {it+1}: delta de pele mediano {np.median(delta[pele]):.2f} | deslocamento aplicado mediano {np.median(dv):.2f} p10 {np.percentile(dv,10):.2f} p90 {np.percentile(dv,90):.2f}", flush=True)
I.vertices = o3d.utility.Vector3dVector(V)
d, k = t.query(V, workers=8); s = np.einsum("ij,ij->i", V - P[k], N[k]); p = d < a.perto
print("s final (perto da pele): mediana %.2f p10 %.2f p25 %.2f p75 %.2f p90 %.2f" % (np.median(s[p]), *np.percentile(s[p], [10, 25, 75, 90])))
o3d.io.write_triangle_mesh(a.saida, I)
