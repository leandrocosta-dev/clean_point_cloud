"""Base + N malhas (ja no mesmo referencial) completando so as lacunas, uma a uma.

Uso: python pipeline/revo_antigo/completar_varias.py base.ply outra1.ply outra2.ply ... --saida out.ply [--lacuna 1.2 --min-tri 300 --corrigir]
Cada malha: (opcional --corrigir) campo suave de deriva (mediana por celula de 20 mm, gaussiana, 2 passadas) em relacao a UNIAO atual;
entram so triangulos (a) cujo deslocamento LATERAL ao vertice mais proximo da uniao > --lacuna (prolongamento alem da borda) ou (b) paredes: normal com |cos| < --cos-parede em relacao a da base e a > 1 mm, com
>= 60% da vizinhanca de 3 mm tambem lacuna, em componentes >= --min-tri encostados na uniao (<= 2 mm). A base fica intocada;
nada fecha furo; componentes minusculos nao entram (sem ilhas).
"""

import argparse

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d


def campo(V, B, nB, celula=20.0):
    """Campo suave de deriva (GPU); ver geom_gpu.campo_deriva. Devolve f(X) -> deslocamento."""
    return gg.campo_deriva(V, B, nB, celula)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("outras", nargs="*")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--lacuna", type=float, default=1.2)
    ap.add_argument("--min-tri", type=int, default=300)
    ap.add_argument("--corrigir", action="store_true")
    ap.add_argument("--cos-parede", type=float, default=0.8)
    a = ap.parse_args()
    xp = gpu.xp()
    uniao = o3d.io.read_triangle_mesh(a.base)
    nbase = len(uniao.triangles)
    for arq in a.outras:
        m = o3d.io.read_triangle_mesh(arq)
        U = uniao
        Uv, Uf = np.asarray(U.vertices), np.asarray(U.triangles)
        B = gpu.para_gpu(Uv, np.float64)
        nB = gg.normais_vertices(Uv, Uf, saida_gpu=True)
        V = gpu.para_gpu(np.asarray(m.vertices), np.float64)
        if a.corrigir:
            for _ in range(2):
                V = V + campo(V, B, nB)(V)
            m.vertices = o3d.utility.Vector3dVector(gpu.para_cpu(V))
        F = np.asarray(m.triangles)
        cen = gg.centros_triangulos(V, F, saida_gpu=True)
        dv, iv = gpu.vizinho_mais_proximo(B, cen, dist_max=np.inf)
        disp = cen - B[iv]
        nrm = (disp * nB[iv]).sum(1)
        lat = xp.sqrt(
            xp.maximum((disp**2).sum(1) - nrm**2, 0)
        )  # parede/degrau: normal do triangulo diferente da da base (pele duplicada tem normal paralela); precisa estar a > 1 mm
        tn = gg.normais_triangulos(V, F, saida_gpu=True)
        cosang = xp.abs((tn * nB[iv]).sum(1))
        lac = (lat > a.lacuna) | (dv > 6.0) | ((cosang < a.cos_parede) & (dv > 1.0))
        if not bool(lac.any()):
            print(f"{arq.split('/')[-1]}: nada fora da uniao")
            continue
        # fracao de triangulos "lacuna" entre os vizinhos a <= 3 mm de cada centroide-lacuna (contagens exatas, sem listas)
        cl_ = cen[lac]
        frac = gpu.contar_no_raio(cl_, cl_, 3.0) / gpu.contar_no_raio(cen, cl_, 3.0)
        keep = xp.zeros(len(F), bool)
        keep[xp.flatnonzero(lac)[frac >= 0.6]] = True
        P = o3d.geometry.TriangleMesh(m)
        P.remove_triangles_by_mask(~gpu.para_cpu(keep))
        P.remove_unreferenced_vertices()
        Pv, Pt = np.asarray(P.vertices), np.asarray(P.triangles)
        cl, n = gg.componentes_triangulos(Pt, len(Pv), saida_gpu=True)
        perto = (gpu.vizinho_mais_proximo(B, Pv, dist_max=2.0)[1] >= 0)[gpu.para_gpu(Pt)].any(
            1
        )  # triangulo com vertice a <= 2 mm da uniao
        toca = xp.bincount(cl, weights=perto.astype(xp.float64), minlength=len(n)) > 0  # componente encostada na uniao
        ok = xp.flatnonzero((n >= a.min_tri) & toca)
        P.remove_triangles_by_mask(~gpu.para_cpu(xp.isin(cl, ok)))
        P.remove_unreferenced_vertices()
        print(f"{arq.split('/')[-1]}: +{len(P.triangles)} triangulos em {len(ok)} componentes")
        uniao = uniao + P
    o3d.io.write_triangle_mesh(a.saida, uniao, write_vertex_normals=False)
    print("base", nbase, "-> total", len(uniao.triangles))


if __name__ == "__main__":
    main()
