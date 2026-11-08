"""Corrige a posicao da interna para a espessura medida com um campo de deslocamento MUITO SUAVE (grade grossa), sem criar rugosidade.

    python pipeline/corrigir_interna_suave.py interna.ply externa.ply nuvem_externa.ply --espessura 2.0 --saida out.ply [--celula 20] [--sigma 1.5] [--passadas 3]

Substitui `corrigir_interna_local.py` (mediana por vertice em raios de 5-25 mm injetou rugosidade: a interna ficou "aspera"). Aqui: delta = -(s + espessura)
por vertice de pele -> mediana por celula de --celula mm -> suavizacao gaussiana da grade (sigma em celulas, normalizada pelas celulas validas) ->
interpolacao trilinear por vertice -> deslocamento ao longo da normal externa. O campo varia devagar, entao o relevo/rugosidade original dos scans nao muda.
"""

import argparse

import gpu
import numpy as np
import open3d as o3d

# GPU: vizinhos, soma por celula, filtro gaussiano e interpolacao trilinear (cupyx.scipy.ndimage); CPU so leitura/gravacao e amostragem da malha (Open3D).
if gpu.em_gpu():
    from cupyx.scipy import ndimage as ndi
else:
    from scipy import ndimage as ndi

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("interna")
ap.add_argument("externa")
ap.add_argument("nuvem")
ap.add_argument("--saida", required=True)
ap.add_argument("--espessura", type=float, default=2.0)
ap.add_argument("--celula", type=float, default=20.0)
ap.add_argument("--sigma", type=float, default=1.5)
ap.add_argument("--passadas", type=int, default=3)
ap.add_argument("--perto", type=float, default=8.0)
a = ap.parse_args()
I = o3d.io.read_triangle_mesh(a.interna)
X = o3d.io.read_triangle_mesh(a.externa)
E = o3d.io.read_point_cloud(a.nuvem)
Pn = np.asarray(E.points)
Nn = np.asarray(E.normals)
m = gpu.xp()
P = gpu.para_gpu(np.asarray(X.sample_points_uniformly(2_000_000).points))
_, j = gpu.vizinho_mais_proximo(Pn, P)
N = gpu.para_gpu(Nn)[j]
V = gpu.para_gpu(np.asarray(I.vertices).copy())
lo = V.min(0) - 2 * a.celula
hi = V.max(0) + 2 * a.celula
shp = [int(x) for x in np.ceil(gpu.para_cpu((hi - lo) / a.celula)).astype(int) + 1]
for it in range(a.passadas):
    d, k = gpu.vizinho_mais_proximo(P, V)
    n = N[k]
    s = ((V - P[k]) * n).sum(1)
    pele = (d < a.perto) & (m.abs(s + a.espessura) < a.perto)
    delta = -(s + a.espessura)
    ij = m.floor((V[pele] - lo) / a.celula).astype(m.int64)
    flat = (ij[:, 0] * shp[1] + ij[:, 1]) * shp[2] + ij[:, 2]
    nflat = shp[0] * shp[1] * shp[2]
    cnt = m.bincount(flat, minlength=nflat)
    soma = gpu.somar_por_grupo(delta[pele], flat, nflat)
    # mediana por celula e aproximada pela media aparada: usa a media (suavizada depois); celulas com <50 vertices ficam vazias
    ok = cnt >= 50
    g = m.zeros(nflat)
    g[ok] = soma[ok] / cnt[ok]
    g = g.reshape(shp)
    w = ok.reshape(shp).astype(m.float64)
    gs = ndi.gaussian_filter(g * w, a.sigma)
    ws = ndi.gaussian_filter(w, a.sigma)
    campo = m.where(ws > 1e-3, gs / m.maximum(ws, 1e-3), 0.0)
    # trilinear entre centros de celula (indice = (V - lo)/celula - 0.5), 0 fora da grade (= RegularGridInterpolator(fill_value=0))
    dv = ndi.map_coordinates(campo, ((V - lo) / a.celula - 0.5).T, order=1, mode="constant", cval=0.0)
    dv[d > 30] = 0.0
    V = V + dv[:, None] * n
    print(
        f"passada {it + 1}: delta de pele mediano {float(m.median(delta[pele])):.2f} | deslocamento aplicado mediano {float(m.median(dv)):.2f} p10 {float(m.percentile(dv, 10)):.2f} p90 {float(m.percentile(dv, 90)):.2f}",
        flush=True,
    )
I.vertices = o3d.utility.Vector3dVector(gpu.para_cpu(V))
d, k = gpu.vizinho_mais_proximo(P, V)
s = ((V - P[k]) * N[k]).sum(1)
p = d < a.perto
print(
    "s final (perto da pele): mediana %.2f p10 %.2f p25 %.2f p75 %.2f p90 %.2f"
    % (float(m.median(s[p])), *[float(x) for x in m.percentile(s[p], m.asarray([10, 25, 75, 90]))])
)
o3d.io.write_triangle_mesh(a.saida, I)
