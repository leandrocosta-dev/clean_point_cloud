"""Une duas malhas do Revo da mesma superficie quando o scan menor (A) tem DERIVA de poucos mm em relacao ao maior (B, base).

Uso: python pipeline/revo_antigo/unir_malhas_deriva.py A.ply B.ply T_B_para_A.npy --saida out.ply [--celula 25 --lacuna 1.2 --min-tri 300]
(criterio de lacuna: deslocamento lateral, nao distancia). 1. leva A ao referencial de B; 2. campo de correcao SUAVE (mediana por celula de --celula mm do deslocamento normal A->B,
gaussiana, trilinear; so baixa frequencia: nao rugosifica, ver corrigir_interna_suave); 3. da A corrigida entram so os
triangulos a > --lacuna mm de B, em componentes >= --min-tri encostados na base (<= 2 mm). B fica intocada; nada fecha furo.
"""
import argparse
import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("T"); ap.add_argument("--saida", required=True)
ap.add_argument("--celula", type=float, default=25.0); ap.add_argument("--lacuna", type=float, default=1.2)
ap.add_argument("--min-tri", type=int, default=300); ap.add_argument("--passadas", type=int, default=3)
a = ap.parse_args()
mA = o3d.io.read_triangle_mesh(a.a); mB = o3d.io.read_triangle_mesh(a.b); mB.compute_vertex_normals()
T = np.load(a.T); mA.transform(np.linalg.inv(T))   # T leva B->A; aqui A -> B
VA = np.asarray(mA.vertices).copy(); B = np.asarray(mB.vertices); nB = np.asarray(mB.vertex_normals); tB = cKDTree(B)
def campo(V):
    d, i = tB.query(V); m = d < 4.0
    vec = ((B[i[m]] - V[m]) * nB[i[m]]).sum(1)[:, None] * nB[i[m]]
    lo = V.min(0) - a.celula; n = np.ceil((V.max(0) + a.celula - lo) / a.celula).astype(int) + 1
    idx = np.floor((V[m] - lo) / a.celula).astype(int)
    S = np.zeros((*n, 3)); N = np.zeros(n)
    flat = np.ravel_multi_index(idx.T, n)
    for k in np.unique(flat):
        sel = flat == k
        if sel.sum() >= 30:
            ijk = np.unravel_index(k, n); S[ijk] = np.median(vec[sel], 0); N[ijk] = 1
    Ss = np.stack([ndi.gaussian_filter(S[..., c], 1.0) for c in range(3)], -1); Ns = ndi.gaussian_filter(N, 1.0)
    ok = Ns > 0.05; F = np.zeros_like(Ss); F[ok] = Ss[ok] / Ns[ok][:, None]
    _, near = ndi.distance_transform_edt(~ok, return_indices=True); F = F[tuple(near)]   # fora da sobreposicao: vizinho preenchido mais proximo
    g = [lo[c] + a.celula * np.arange(n[c]) for c in range(3)]
    return RegularGridInterpolator(g, F, bounds_error=False, fill_value=None)
def stats(V, rot):
    d, i = tB.query(V); m = d < 4; s = ((V[m] - B[i[m]]) * nB[i[m]]).sum(1)
    print(f"  {rot}: |d| mediana {np.median(np.abs(s)):.2f} mm | p90 {np.percentile(np.abs(s),90):.2f} | A a < 0,5 mm de B: {100*(d<0.5).mean():.1f}%")
stats(VA, "antes")
for k in range(a.passadas):
    VA = VA + campo(VA)(VA); stats(VA, f"passada {k+1}")
mA.vertices = o3d.utility.Vector3dVector(VA)
F = np.asarray(mA.triangles); cen = VA[F].mean(1)
tS = cKDTree(np.asarray(mB.sample_points_uniformly(2_000_000).points))
# lacuna = deslocamento LATERAL ao vertice mais proximo de B > --lacuna: prolongamento real alem da borda da base. Pele paralela
# sobre B (deriva) se afasta so na direcao da normal e nao entra (60% dos triangulos a < 3 mm de B eram pele duplicada).
dv, iv = tB.query(cen); disp = cen - B[iv]; nrm = ((disp * nB[iv]).sum(1))
lat = np.sqrt(np.maximum((disp ** 2).sum(1) - nrm ** 2, 0)); lac = (lat > a.lacuna) | (dv > 6.0)
viz = cKDTree(cen).query_ball_point(cen[lac], 3.0); frac = np.array([lac[v].mean() for v in viz])
keep = np.zeros(len(F), bool); keep[np.flatnonzero(lac)[frac >= 0.6]] = True
P = o3d.geometry.TriangleMesh(mA); P.remove_triangles_by_mask(~keep); P.remove_unreferenced_vertices()
cl, n, _ = P.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
Pv = np.asarray(P.vertices); Pt = np.asarray(P.triangles); ok = []
for c in np.flatnonzero(n >= a.min_tri):
    vs = np.unique(Pt[cl == c]); dm = tB.query(Pv[vs])[0].min()
    if dm <= 2.0: ok.append(c)
P.remove_triangles_by_mask(~np.isin(cl, ok)); P.remove_unreferenced_vertices()
print(f"pedacos de A: {len(P.triangles)} triangulos em {len(ok)} componentes encostados na base; base {len(mB.triangles)}")
o3d.io.write_triangle_mesh(a.saida, mB + P, write_vertex_normals=False)
