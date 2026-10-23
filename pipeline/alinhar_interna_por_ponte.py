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
import sys
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for k in ("projeto", "pasta_seg", "direto", "tseg", "externa"):
    ap.add_argument(k)
ap.add_argument("--ponte", required=True); ap.add_argument("--saida", required=True); ap.add_argument("--extras")
ap.add_argument("--tol-limpa", type=float, default=40.0)
a = ap.parse_args(); baixa_prioridade()
os.makedirs(os.path.join(a.saida, "scans"), exist_ok=True)
ind = json.load(open(os.path.join(a.pasta_seg, "segmentos_indice.json"))); pj = json.load(open(a.direto))["segmentos"]; Tr = np.load(a.tseg)
E = o3d.io.read_point_cloud(a.externa); P = np.asarray(E.points); N = np.asarray(E.normals); tE = cKDTree(P)
Ep = E  # normais ja orientadas para fora


def carregar(nome):
    if nome.startswith("extra_"):
        e = next(x for x in json.load(open(a.extras)) if "extra_" + x["scan"] == nome)
        M = o3d.io.read_triangle_mesh(os.path.join(e["proj"], "data", e["scan"], "fuse_mesh.ply")); M.transform(np.array(e["T"]))
    else:
        seg = max(ind[nome], key=lambda s: len(np.load(s["arquivo"]))); T0 = Tr @ np.array(next(x for x in pj[nome] if x["seg"] == seg["seg"])["T"])
        M = o3d.io.read_triangle_mesh(os.path.join(a.projeto, "data", nome, "fuse_mesh.ply")); M.transform(T0)
    return M


def amostra(M, n):
    S = M.sample_points_uniformly(n, use_triangle_normal=True); return S


def sinal(M):
    M.compute_triangle_normals(); T = np.asarray(M.triangle_normals); C = np.asarray(M.vertices)[np.asarray(M.triangles)].mean(1)
    d, k = tE.query(C, workers=8); return d, np.einsum("ij,ij->i", C - P[k], N[k]), np.einsum("ij,ij->i", T, N[k])


def icp(src, dst, taus, T=None, it=40):
    T = np.eye(4) if T is None else T
    for tol in taus:
        r = o3d.pipelines.registration.registration_icp(src, dst, tol, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(tol / 3)), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T = r.transformation
    return T, r


# 1. ponte
B = carregar(a.ponte); d, s, dot = sinal(B)
fora = (d < 4) & (dot > 0.3)
Bo = o3d.geometry.TriangleMesh(B); Bo.remove_triangles_by_mask(~fora); Bo.remove_unreferenced_vertices()
S = amostra(Bo, 600_000)
TB, r = icp(S, Ep, (6, 3, 1.5))
print(f"ponte {a.ponte}: faces externas {fora.sum():,} | ICP fitness {r.fitness:.3f} rmse {r.inlier_rmse:.3f}", flush=True)
B.transform(TB); d, s, dot = sinal(B)
dentro = (dot < -0.3) & (d < 12)
print(f"ponte: faces internas perto da pele {dentro.sum():,} | s (interna - externa) mediana {np.median(s[dentro]):.2f} p25 {np.percentile(s[dentro],25):.2f} p75 {np.percentile(s[dentro],75):.2f}", flush=True)
Bi = o3d.geometry.TriangleMesh(B); Bi.remove_triangles_by_mask(~dentro); Bi.remove_unreferenced_vertices()
R = amostra(Bi, 700_000); resultados = {a.ponte: (TB, np.median(s[dentro]))}
guard = [(a.ponte, B)]
# 2. demais scans internos
cand = [n for n in sorted(ind)] + (["extra_" + e["scan"] for e in json.load(open(a.extras))] if a.extras else [])
cand = [n for n in cand if n != a.ponte]
internos = {}
for n in cand:
    M = carregar(n); d, s, dot = sinal(M); perto = d < 5
    if perto.sum() and (dot[perto] < -0.3).sum() / perto.sum() > 0.5:
        internos[n] = M
    else:
        print(n, "externo, ignorado", flush=True)
print("scans internos a alinhar:", len(internos), flush=True)
saidas = {a.ponte: B}
while internos:
    melhor, mf = None, -1
    for n, M in internos.items():
        S = amostra(M, 200_000); ev = o3d.pipelines.registration.evaluate_registration(S, R, 4.0, np.eye(4))
        if ev.fitness > mf:
            melhor, mf = n, ev.fitness
    M = internos.pop(melhor); S = amostra(M, 600_000)
    Tn, r = icp(S, R, (8, 4, 2, 1.2))
    M.transform(Tn); d, s, dot = sinal(M); perto = (d < 12) & (dot < -0.3)
    print(f"{melhor}: sobreposicao inicial {mf:.2f} | ICP fitness {r.fitness:.3f} rmse {r.inlier_rmse:.3f} | s mediano {np.median(s[perto]):.2f}", flush=True)
    if r.fitness < 0.15:
        print("  (baixa sobreposicao com a referencia; mantido, mas confira)", flush=True)
    saidas[melhor] = M
    R = R + amostra(M, 250_000)
# 3. grava
for n, M in saidas.items():
    d, _, _ = sinal(M); M.remove_triangles_by_mask(d > a.tol_limpa); M.remove_unreferenced_vertices()
    cl, nn, _ = M.cluster_connected_triangles(); cl = np.asarray(cl); nn = np.asarray(nn)
    M.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(nn < 500))); M.remove_unreferenced_vertices()
    o3d.io.write_triangle_mesh(os.path.join(a.saida, "scans", n + ".ply"), M, write_vertex_normals=False)
print("gravado em", os.path.join(a.saida, "scans"))
