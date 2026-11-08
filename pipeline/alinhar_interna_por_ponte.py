"""Alinha os scans da face INTERNA usando um scan-PONTE (que enxerga as duas faces) para preservar a espessura real da parede.

    python pipeline/alinhar_interna_por_ponte.py <projeto> <pasta_seg> <direto.json> <T_seg_revo.npy> <nuvem_externa.ply> --ponte 20260921_190200 --saida <pasta> [--extras json]

1) ponte: alinha so as faces VOLTADAS PARA FORA (perto da nuvem externa) a nuvem externa (ICP ponto-plano robusto); o scan inteiro recebe essa pose,
   entao a parte interna da ponte cai na posicao correta em relacao a pele externa (medida do proprio scan, sem espessura assumida);
2) referencia R = parte interna da ponte; cada outro scan interno e alinhado contra R (interna x interna, sem degenerescencia de espessura),
   do que mais se sobrepoe para o que menos; cada scan alinhado entra em R;
3) grava os scans alinhados e aparados (<= --tol-limpa da nuvem externa) em <saida>/scans/<nome>.ply: o cache que construcao_por_cobertura.py --so-internos reutiliza.
Relatorio: espessura (s mediano da pele interna em relacao a externa) por scan.
"""

import argparse
import json
import os

import gpu
import numpy as np

# Open3D so para ler/gravar malhas e para cluster_connected_triangles (topologia de malha, sem equivalente na gpu.py).
# Todo o calculo numerico (transformar, normais de triangulo, amostragem, vizinhos, ICP, encaixe) roda na GPU.
import open3d as o3d
from alinhar_varios import baixa_prioridade
from registro_gpu import Nuvem, avaliar_encaixe, icp, transformar

xp = gpu.xp()
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for k in ("projeto", "pasta_seg", "direto", "tseg", "externa"):
    ap.add_argument(k)
ap.add_argument("--ponte", required=True)
ap.add_argument("--saida", required=True)
ap.add_argument("--extras")
ap.add_argument("--tol-limpa", type=float, default=40.0)
a = ap.parse_args()
baixa_prioridade()
os.makedirs(os.path.join(a.saida, "scans"), exist_ok=True)
ind = json.load(open(os.path.join(a.pasta_seg, "segmentos_indice.json")))
pj = json.load(open(a.direto))["segmentos"]
Tr = np.load(a.tseg)
E = o3d.io.read_point_cloud(a.externa)
P = gpu.para_gpu(np.asarray(E.points), xp.float64)
N = gpu.para_gpu(np.asarray(E.normals), xp.float64)
Ep = Nuvem(P, N)  # normais ja orientadas para fora


class Malha:
    """Vertices (n, 3) float64 e triangulos (m, 3) int64 na GPU."""

    def __init__(self, V, F):
        self.V, self.F = V, F

    @classmethod
    def ler(cls, caminho):
        M = o3d.io.read_triangle_mesh(caminho)
        return cls(gpu.para_gpu(np.asarray(M.vertices), xp.float64), gpu.para_gpu(np.asarray(M.triangles), xp.int64))

    def transformar(self, T):
        self.V = transformar(self.V, T)

    def normais(self):
        A, B, C = (self.V[self.F[:, k]] for k in range(3))
        n = xp.cross(B - A, C - A)
        return n / xp.maximum(xp.linalg.norm(n, axis=1), 1e-300)[:, None]

    def centros(self):
        return self.V[self.F].mean(1)

    def filtrar(self, mascara):
        return Malha(self.V, self.F[mascara])

    def amostra(self, n, semente=0):
        """n pontos uniformes na superficie (triangulo por area, coordenadas baricentricas) com a normal do triangulo."""
        A, B, C = (self.V[self.F[:, k]] for k in range(3))
        area = 0.5 * xp.linalg.norm(xp.cross(B - A, C - A), axis=1)
        g = xp.random.default_rng(semente)
        cum = xp.cumsum(area)
        t = xp.minimum(xp.searchsorted(cum, g.random(n) * cum[-1]), len(area) - 1)
        u, v = g.random(n), g.random(n)
        dobra = u + v > 1
        u, v = xp.where(dobra, 1 - u, u), xp.where(dobra, 1 - v, v)
        pts = A[t] + u[:, None] * (B[t] - A[t]) + v[:, None] * (C[t] - A[t])
        return Nuvem(pts, self.normais()[t])

    def para_o3d(self):
        M = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(gpu.para_cpu(self.V)),
            o3d.utility.Vector3iVector(gpu.para_cpu(self.F).astype(np.int32)),
        )
        return M


def carregar(nome):
    if nome.startswith("extra_"):
        e = next(x for x in json.load(open(a.extras)) if "extra_" + x["scan"] == nome)
        M = Malha.ler(os.path.join(e["proj"], "data", e["scan"], "fuse_mesh.ply"))
        M.transformar(np.array(e["T"]))
    else:
        seg = max(ind[nome], key=lambda s: len(np.load(s["arquivo"])))
        T0 = Tr @ np.array(next(x for x in pj[nome] if x["seg"] == seg["seg"])["T"])
        M = Malha.ler(os.path.join(a.projeto, "data", nome, "fuse_mesh.ply"))
        M.transformar(T0)
    return M


def amostra(M, n):
    return M.amostra(n)


def sinal(M):
    T = M.normais()
    C = M.centros()
    d, k = gpu.vizinho_mais_proximo(P, C)  # vizinho mais proximo da pele externa (sem limite de distancia)
    return d, ((C - P[k]) * N[k]).sum(1), (T * N[k]).sum(1)


def icp_multi(src, dst, taus, T=None, it=40):
    T = np.eye(4) if T is None else T
    for tol in taus:
        T, _ = icp(src, dst, T, tol, it, robusto=tol / 3)
    fit, rmse, _ = avaliar_encaixe(src, dst, taus[-1], T)  # como r.fitness / r.inlier_rmse do Open3D
    return T, (fit, rmse)


def med(x):
    return float(xp.median(x))


# 1. ponte
B = carregar(a.ponte)
d, s, dot = sinal(B)
fora = (d < 4) & (dot > 0.3)
Bo = B.filtrar(fora)
S = amostra(Bo, 600_000)
TB, (fit, rmse) = icp_multi(S, Ep, (6, 3, 1.5))
print(f"ponte {a.ponte}: faces externas {int(fora.sum()):,} | ICP fitness {fit:.3f} rmse {rmse:.3f}", flush=True)
B.transformar(TB)
d, s, dot = sinal(B)
dentro = (dot < -0.3) & (d < 12)
q = gpu.para_cpu(xp.percentile(s[dentro], xp.asarray([25.0, 75.0])))
print(
    f"ponte: faces internas perto da pele {int(dentro.sum()):,} | s (interna - externa) mediana {med(s[dentro]):.2f} p25 {q[0]:.2f} p75 {q[1]:.2f}",
    flush=True,
)
Bi = B.filtrar(dentro)
R = amostra(Bi, 700_000)
resultados = {a.ponte: (TB, med(s[dentro]))}
guard = [(a.ponte, B)]
# 2. demais scans internos
cand = [n for n in sorted(ind)] + (["extra_" + e["scan"] for e in json.load(open(a.extras))] if a.extras else [])
cand = [n for n in cand if n != a.ponte]
internos = {}
for n in cand:
    M = carregar(n)
    d, s, dot = sinal(M)
    perto = d < 5
    if int(perto.sum()) and int((dot[perto] < -0.3).sum()) / int(perto.sum()) > 0.5:
        internos[n] = M
    else:
        print(n, "externo, ignorado", flush=True)
print("scans internos a alinhar:", len(internos), flush=True)
saidas = {a.ponte: B}
while internos:
    melhor, mf = None, -1
    for n, M in internos.items():
        S = amostra(M, 200_000)
        fit_n = avaliar_encaixe(S, R, 4.0)[0]
        if fit_n > mf:
            melhor, mf = n, fit_n
    M = internos.pop(melhor)
    S = amostra(M, 600_000)
    Tn, (fit, rmse) = icp_multi(S, R, (8, 4, 2, 1.2))
    M.transformar(Tn)
    d, s, dot = sinal(M)
    perto = (d < 12) & (dot < -0.3)
    print(
        f"{melhor}: sobreposicao inicial {mf:.2f} | ICP fitness {fit:.3f} rmse {rmse:.3f} | s mediano {med(s[perto]):.2f}",
        flush=True,
    )
    if fit < 0.15:
        print("  (baixa sobreposicao com a referencia; mantido, mas confira)", flush=True)
    saidas[melhor] = M
    A2 = amostra(M, 250_000)
    R = Nuvem(xp.concatenate([R.pts, A2.pts]), xp.concatenate([R.normais, A2.normais]))
# 3. grava
for n, M in saidas.items():
    d, _, _ = sinal(M)
    O = M.filtrar(d <= a.tol_limpa).para_o3d()
    O.remove_unreferenced_vertices()
    cl, nn, _ = O.cluster_connected_triangles()
    cl = np.asarray(cl)
    nn = np.asarray(nn)
    O.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(nn < 500)))
    O.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(os.path.join(a.saida, "scans", n + ".ply"), O, write_vertex_normals=False)
print("gravado em", os.path.join(a.saida, "scans"))
