"""Une duas malhas do Revo da mesma superficie quando o scan menor (A) tem DERIVA de poucos mm em relacao ao maior (B, base).

Uso: python pipeline/revo_antigo/unir_malhas_deriva.py A.ply B.ply T_B_para_A.npy --saida out.ply [--celula 25 --lacuna 1.2 --min-tri 300]
(criterio de lacuna: deslocamento lateral, nao distancia). 1. leva A ao referencial de B; 2. campo de correcao SUAVE (mediana por celula de --celula mm do deslocamento normal A->B,
gaussiana, trilinear; so baixa frequencia: nao rugosifica, ver corrigir_interna_suave); 3. da A corrigida entram so os
triangulos a > --lacuna mm de B, em componentes >= --min-tri encostados na base (<= 2 mm). B fica intocada; nada fecha furo.
"""

import argparse

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("T")
ap.add_argument("--saida", required=True)
ap.add_argument("--celula", type=float, default=25.0)
ap.add_argument("--lacuna", type=float, default=1.2)
ap.add_argument("--min-tri", type=int, default=300)
ap.add_argument("--passadas", type=int, default=3)
a = ap.parse_args()
xp = gpu.xp()
mA = o3d.io.read_triangle_mesh(a.a)
mB = o3d.io.read_triangle_mesh(a.b)
T = np.load(a.T)
mA.transform(np.linalg.inv(T))  # T leva B->A; aqui A -> B
VA = gpu.para_gpu(np.asarray(mA.vertices), np.float64)
Bv, Bf = np.asarray(mB.vertices), np.asarray(mB.triangles)
B = gpu.para_gpu(Bv, np.float64)
nB = gg.normais_vertices(Bv, Bf, saida_gpu=True)


def stats(V, rot):
    d, i = gpu.vizinho_mais_proximo(B, V, dist_max=np.inf)
    m = d < 4
    s = ((V[m] - B[i[m]]) * nB[i[m]]).sum(1)
    print(
        f"  {rot}: |d| mediana {float(xp.median(xp.abs(s))):.2f} mm | p90 {float(xp.percentile(xp.abs(s), 90)):.2f} | A a < 0,5 mm de B: {100 * float((d < 0.5).mean()):.1f}%"
    )


stats(VA, "antes")
for k in range(a.passadas):
    VA = VA + gg.campo_deriva(VA, B, nB, a.celula)(VA)
    stats(VA, f"passada {k + 1}")
mA.vertices = o3d.utility.Vector3dVector(gpu.para_cpu(VA))
F = np.asarray(mA.triangles)
cen = gg.centros_triangulos(VA, F, saida_gpu=True)
# lacuna = deslocamento LATERAL ao vertice mais proximo de B > --lacuna: prolongamento real alem da borda da base. Pele paralela
# sobre B (deriva) se afasta so na direcao da normal e nao entra (60% dos triangulos a < 3 mm de B eram pele duplicada).
dv, iv = gpu.vizinho_mais_proximo(B, cen, dist_max=np.inf)
disp = cen - B[iv]
nrm = (disp * nB[iv]).sum(1)
lat = xp.sqrt(xp.maximum((disp**2).sum(1) - nrm**2, 0))
lac = (lat > a.lacuna) | (dv > 6.0)
cl_ = cen[lac]
frac = gpu.contar_no_raio(cl_, cl_, 3.0) / gpu.contar_no_raio(
    cen, cl_, 3.0
)  # fracao de lacuna entre os vizinhos a <= 3 mm
keep = xp.zeros(len(F), bool)
keep[xp.flatnonzero(lac)[frac >= 0.6]] = True
P = o3d.geometry.TriangleMesh(mA)
P.remove_triangles_by_mask(~gpu.para_cpu(keep))
P.remove_unreferenced_vertices()
Pv = np.asarray(P.vertices)
Pt = np.asarray(P.triangles)
cl, n = gg.componentes_triangulos(Pt, len(Pv), saida_gpu=True)
perto = (gpu.vizinho_mais_proximo(B, Pv, dist_max=2.0)[1] >= 0)[gpu.para_gpu(Pt)].any(
    1
)  # triangulo com vertice a <= 2 mm da base
toca = xp.bincount(cl, weights=perto.astype(xp.float64), minlength=len(n)) > 0  # componente encostada na base
ok = xp.flatnonzero((n >= a.min_tri) & toca)
P.remove_triangles_by_mask(~gpu.para_cpu(xp.isin(cl, ok)))
P.remove_unreferenced_vertices()
print(
    f"pedacos de A: {len(P.triangles)} triangulos em {len(ok)} componentes encostados na base; base {len(mB.triangles)}"
)
o3d.io.write_triangle_mesh(a.saida, mB + P, write_vertex_normals=False)
