"""Base + N malhas (ja no mesmo referencial) completando so as lacunas, uma a uma.

Uso: python pipeline/revo_antigo/completar_varias.py base.ply outra1.ply outra2.ply ... --saida out.ply [--lacuna 1.2 --min-tri 300 --corrigir]
Cada malha: (opcional --corrigir) campo suave de deriva (mediana por celula de 20 mm, gaussiana, 2 passadas) em relacao a UNIAO atual;
entram so triangulos (a) cujo deslocamento LATERAL ao vertice mais proximo da uniao > --lacuna (prolongamento alem da borda) ou (b) paredes: normal com |cos| < --cos-parede em relacao a da base e a > 1 mm, com
>= 60% da vizinhanca de 3 mm tambem lacuna, em componentes >= --min-tri encostados na uniao (<= 2 mm). A base fica intocada;
nada fecha furo; componentes minusculos nao entram (sem ilhas).
"""
import argparse
import numpy as np
import open3d as o3d
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree


def campo(V, B, nB, tB, celula=20.0):
    d, i = tB.query(V); m = d < 4.0
    vec = ((B[i[m]] - V[m]) * nB[i[m]]).sum(1)[:, None] * nB[i[m]]
    lo = V.min(0) - celula; n = np.ceil((V.max(0) + celula - lo) / celula).astype(int) + 1
    idx = np.floor((V[m] - lo) / celula).astype(int); S = np.zeros((*n, 3)); N = np.zeros(n); flat = np.ravel_multi_index(idx.T, n)
    for k in np.unique(flat):
        sel = flat == k
        if sel.sum() >= 30:
            ijk = np.unravel_index(k, n); S[ijk] = np.median(vec[sel], 0); N[ijk] = 1
    Ss = np.stack([ndi.gaussian_filter(S[..., c], 1.0) for c in range(3)], -1); Ns = ndi.gaussian_filter(N, 1.0)
    ok = Ns > 0.05; F = np.zeros_like(Ss); F[ok] = Ss[ok] / Ns[ok][:, None]
    if not ok.any():
        return lambda X: np.zeros_like(X)
    _, near = ndi.distance_transform_edt(~ok, return_indices=True); F = F[tuple(near)]
    g = [lo[c] + celula * np.arange(n[c]) for c in range(3)]
    return RegularGridInterpolator(g, F, bounds_error=False, fill_value=None)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("outras", nargs="*"); ap.add_argument("--saida", required=True)
    ap.add_argument("--lacuna", type=float, default=1.2); ap.add_argument("--min-tri", type=int, default=300); ap.add_argument("--corrigir", action="store_true"); ap.add_argument("--cos-parede", type=float, default=0.8)
    a = ap.parse_args()
    uniao = o3d.io.read_triangle_mesh(a.base); nbase = len(uniao.triangles)
    for arq in a.outras:
        m = o3d.io.read_triangle_mesh(arq); U = uniao; U.compute_vertex_normals()
        B = np.asarray(U.vertices); nB = np.asarray(U.vertex_normals); tB = cKDTree(B); V = np.asarray(m.vertices).copy()
        if a.corrigir:
            for _ in range(2): V = V + campo(V, B, nB, tB)(V)
            m.vertices = o3d.utility.Vector3dVector(V)
        F = np.asarray(m.triangles); cen = V[F].mean(1); dv, iv = tB.query(cen); disp = cen - B[iv]; nrm = (disp * nB[iv]).sum(1)
        lat = np.sqrt(np.maximum((disp ** 2).sum(1) - nrm ** 2, 0)); # parede/degrau: normal do triangulo diferente da da base (pele duplicada tem normal paralela); precisa estar a > 1 mm
        m.compute_triangle_normals(); tn = np.asarray(m.triangle_normals); cosang = np.abs((tn * nB[iv]).sum(1))
        lac = (lat > a.lacuna) | (dv > 6.0) | ((cosang < a.cos_parede) & (dv > 1.0))
        if not lac.any():
            print(f"{arq.split('/')[-1]}: nada fora da uniao"); continue
        viz = cKDTree(cen).query_ball_point(cen[lac], 3.0); frac = np.array([lac[v].mean() for v in viz])
        keep = np.zeros(len(F), bool); keep[np.flatnonzero(lac)[frac >= 0.6]] = True
        P = o3d.geometry.TriangleMesh(m); P.remove_triangles_by_mask(~keep); P.remove_unreferenced_vertices()
        cl, n, _ = P.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n); Pv = np.asarray(P.vertices); Pt = np.asarray(P.triangles); ok = []
        for c in np.flatnonzero(n >= a.min_tri):
            if tB.query(Pv[np.unique(Pt[cl == c])])[0].min() <= 2.0: ok.append(c)
        P.remove_triangles_by_mask(~np.isin(cl, ok)); P.remove_unreferenced_vertices()
        print(f"{arq.split('/')[-1]}: +{len(P.triangles)} triangulos em {len(ok)} componentes"); uniao = uniao + P
    o3d.io.write_triangle_mesh(a.saida, uniao, write_vertex_normals=False)
    print("base", nbase, "-> total", len(uniao.triangles))


if __name__ == "__main__":
    main()
