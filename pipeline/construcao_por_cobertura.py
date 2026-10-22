"""Malha por CONSTRUCAO POR COBERTURA: um scan de base (o de maior cobertura) + so os pedacos de outros scans que preenchem lacunas.

    python pipeline/construcao_por_cobertura.py <projeto_original> <pasta_seg> <direto.json> <T_seg_revo.npy> <nuvem_externa.ply> --saida <pasta>
        [--tol-limpa 1.5] [--tol-lacuna 1.5] [--ganho-min 0.003] [--margem 2.0] [--min-tri 300]

Por que: promediar 20 scans de mao gera nuvem "grossa" (desencontro fino) e malha rugosa/pontilhada (v5..v12: 126-580 um contra 22 um de UM scan).
A malha original do Revo de um scan (fuse_mesh.ply) e lisa. Aqui nenhuma superficie e promediada:
 1. cada scan: malha original -> referencial da peca (pose do segmento mais pesado + ICP rigido contra a nuvem externa) -> aparada
    (so triangulos a <= --tol-limpa mm da nuvem externa: tira mesa/cenario/parede interna) -> cobertura da pele externa a 1 mm;
 2. base = scan de maior cobertura; 3. guloso: o scan que cobre mais nuvem ainda NAO coberta; dele so entram os triangulos a > --tol-lacuna mm
    da malha atual (mais uma margem de --margem mm para sobrepor), por componente conexo >= --min-tri, cada um realinhado localmente (ICP rigido
    so com a vizinhanca de 25 mm contra a malha atual) para nao deixar degrau; 4. solda vertices, repara com MeshLib, mede.
Saida: <pasta>/base_e_pedacos.ply (+ relatorio.json com a ordem e os ganhos). Nada e escrito nos projetos originais.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade


def alinhar_e_limpar(proj, scan, ind, pj, Tr, E, tE, tol_limpa, T0=None):
    if T0 is None:
        seg = max(ind, key=lambda s: len(np.load(s["arquivo"])))
        T0 = Tr @ np.array(next(x for x in pj if x["seg"] == seg["seg"])["T"])
    M = o3d.io.read_triangle_mesh(os.path.join(proj, "data", scan, "fuse_mesh.ply")); M.transform(T0)
    S = M.sample_points_uniformly(600_000); S.estimate_normals()
    T = np.eye(4)
    for tol in (8, 4, 2, 1.2):
        r = o3d.pipelines.registration.registration_icp(S, E, tol, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(tol / 3)), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        T = r.transformation
    M.transform(T)
    d, _ = tE.query(np.asarray(M.vertices), workers=8)
    keep = (d <= tol_limpa)[np.asarray(M.triangles)].all(1)
    M.remove_triangles_by_mask(~keep); M.remove_unreferenced_vertices()
    cl, n, _ = M.cluster_connected_triangles(); cl = np.asarray(cl); n = np.asarray(n)
    M.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(n < 500))); M.remove_unreferenced_vertices()
    return M, r.fitness, r.inlier_rmse


def amostrar(M, n=1_500_000):
    return np.asarray(M.sample_points_uniformly(n).points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("projeto", "pasta_seg", "direto", "tseg", "externa"):
        ap.add_argument(k)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--tol-limpa", type=float, default=1.5); ap.add_argument("--tol-lacuna", type=float, default=1.5)
    ap.add_argument("--ganho-min", type=float, default=0.003); ap.add_argument("--margem", type=float, default=1.0)
    ap.add_argument("--so-externos", action="store_true", help="ignora scans cuja malha esta orientada p/ dentro (viram a parede por dentro: sao da face INTERNA, ~1,6 mm deslocados)")
    ap.add_argument("--so-internos", action="store_true", help="o inverso: usa so os scans da face INTERNA, sem inverter (malha fica com normais para dentro da cavidade)")
    ap.add_argument("--cob-tol", type=float, default=1.0, help="mm: distancia a nuvem externa que conta como cobertura (use ~4 para a face interna)")
    ap.add_argument("--espessura", type=float, default=0.0, help="mm: alinha/apara contra a nuvem externa deslocada para DENTRO por essa espessura (face interna)")
    ap.add_argument("--extras", help="json: lista de {proj, scan, T} de scans de OUTROS projetos/dias (T = pose inicial scan->referencial)")
    ap.add_argument("--apenas", nargs="*", help="so estes scans do projeto principal (nomes)")
    ap.add_argument("--min-tri", type=int, default=500); ap.add_argument("--abertura", type=float, default=3.0)
    a = ap.parse_args(); baixa_prioridade(); t0 = time.time()
    os.makedirs(os.path.join(a.saida, "scans"), exist_ok=True)
    ind = json.load(open(os.path.join(a.pasta_seg, "segmentos_indice.json"))); pj = json.load(open(a.direto))["segmentos"]; Tr = np.load(a.tseg)
    E = o3d.io.read_point_cloud(a.externa)
    if a.espessura:
        En0 = np.asarray(E.normals).copy(); E.points = o3d.utility.Vector3dVector(np.asarray(E.points) - a.espessura * En0); E.normals = o3d.utility.Vector3dVector(En0)
    E.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    Ep = np.asarray(E.points); rng = np.random.default_rng(0); Ea = Ep[rng.choice(len(Ep), 1_500_000, replace=False)]   # amostra para cobertura
    tE = cKDTree(Ep); En = np.asarray(E.normals)
    # 1. alinhar e aparar cada scan
    meshes, cobs = {}, {}
    fontes = [(scan, a.projeto, scan, ind[scan], pj[scan], None) for scan in sorted(ind) if not a.apenas or scan in a.apenas]
    if a.extras:
        for e in json.load(open(a.extras)):
            fontes.append(("extra_" + e["scan"], e["proj"], e["scan"], None, None, np.array(e["T"])))
    for scan, proj_f, real, ind_s, pj_s, T_ini in fontes:
        f = os.path.join(a.saida, "scans", scan + ".ply")
        if os.path.exists(f):
            M = o3d.io.read_triangle_mesh(f)
        else:
            if not os.path.exists(os.path.join(proj_f, "data", real, "fuse_mesh.ply")):
                print(scan, "sem fuse_mesh, ignorado"); continue
            M, fit, rmse = alinhar_e_limpar(proj_f, real, ind_s, pj_s, Tr, E, tE, a.tol_limpa, T_ini)
            o3d.io.write_triangle_mesh(f, M, write_vertex_normals=False)
        if len(M.triangles) == 0:
            continue
        M.compute_triangle_normals(); Tn = np.asarray(M.triangle_normals); Cn = np.asarray(M.vertices)[np.asarray(M.triangles)].mean(1)
        _, jn = tE.query(Cn[::7], workers=8); conc = float(np.mean(np.einsum("ij,ij->i", Tn[::7], En[jn]) > 0))
        if conc < 0.5 and a.so_externos:
            print(f"{scan}: orientacao concorda {conc*100:.0f}% -> scan da face INTERNA, ignorado (--so-externos)", flush=True); continue
        if conc >= 0.5 and a.so_internos:
            print(f"{scan}: orientacao concorda {conc*100:.0f}% -> scan externo, ignorado (--so-internos)", flush=True); continue
        if conc < 0.5 and not a.so_internos:                       # malha do scan orientada para dentro: inverte o scan inteiro (faces de verso viram de frente)
            Tt = np.asarray(M.triangles)[:, ::-1].copy(); M.triangles = o3d.utility.Vector3iVector(Tt); M.compute_triangle_normals()
        print(f"{scan}: orientacao concorda com a nuvem externa em {conc*100:.0f}%" + (" -> invertida" if conc < 0.5 and not a.so_internos else ""), flush=True)
        meshes[scan] = M
        d, _ = cKDTree(amostrar(M)).query(Ea, workers=8); cobs[scan] = d < a.cob_tol
        print(f"{scan}: {len(M.triangles):,} tri | cobre {cobs[scan].mean()*100:.1f}% da pele externa a 1 mm ({time.time()-t0:.0f}s)", flush=True)
    # 2. base + guloso
    base = max(cobs, key=lambda s: cobs[s].mean()); ordem = [dict(scan=base, papel="base", cobertura=float(cobs[base].mean()))]
    B = meshes[base]; V = [np.asarray(B.vertices)]; Tri = [np.asarray(B.triangles)]; nv = len(V[0]); Lab = [np.zeros(len(Tri[0]), np.int16)]; nomes = [base]
    cobertos = cobs[base].copy(); restantes = [s for s in meshes if s != base]
    cur = o3d.geometry.TriangleMesh(B)
    while restantes:
        ganho = {s: (cobs[s] & ~cobertos).mean() for s in restantes}
        s = max(ganho, key=ganho.get)
        if ganho[s] < a.ganho_min:
            break
        restantes.remove(s)
        Cs = amostrar(cur, 2_500_000); tC = cKDTree(Cs)
        S = meshes[s]; Sv = np.asarray(S.vertices); St = np.asarray(S.triangles); cen = Sv[St].mean(1)
        dc, _ = tC.query(cen, workers=8); lac = dc > a.tol_lacuna
        # componentes das triangulos de lacuna (adjacencia por vertice)
        # abertura: so o interior das lacunas (>= 60% da vizinhanca de 3 mm tambem e lacuna) -> descarta tiras finas e bordas rasgadas
        tcen = cKDTree(cen); viz_l = tcen.query_ball_point(cen[lac], a.abertura)
        frac = np.array([lac[v].mean() for v in viz_l]); keep_i = np.flatnonzero(lac)[frac >= 0.6]
        idx = keep_i
        if len(idx) == 0:
            continue
        vv = St[idx].ravel(); tid = np.repeat(np.arange(len(idx)), 3)
        _, inv = np.unique(vv, return_inverse=True)
        G = sparse.coo_matrix((np.ones(len(inv)), (tid, inv)), shape=(len(idx), inv.max() + 1)).tocsr(); A = G @ G.T
        nc, lab = connected_components(A, directed=False); tam = np.bincount(lab)
        adds = 0; tS = cKDTree(Sv)
        for c in np.flatnonzero(tam >= a.min_tri):
            tri_c = idx[lab == c]
            # margem: triangulos de S cujo centroide esta a <= margem dos triangulos da lacuna
            cc = cen[tri_c]; dm, _ = cKDTree(cc).query(cen, workers=8); tri_m = np.flatnonzero(dm <= a.margem)
            P = Sv[np.unique(St[tri_m])]
            # realinhamento local: vertices de S a <= 25 mm do remendo contra a malha atual
            viz = np.unique(np.concatenate(tS.query_ball_point(cc[::max(1, len(cc)//300)], 25.0)))
            pc_s = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Sv[viz])); pc_s.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
            dcur, _ = tC.query(Sv[viz], workers=8); ok = dcur < 3.0
            Tl = np.eye(4)
            if ok.sum() > 2000:
                src = pc_s.select_by_index(np.flatnonzero(ok).tolist()); cur_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Cs[::2])); cur_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
                rr = o3d.pipelines.registration.registration_icp(src, cur_pc, 1.5, np.eye(4), o3d.pipelines.registration.TransformationEstimationPointToPlane(
                    o3d.pipelines.registration.TukeyLoss(0.4)), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                if np.linalg.norm(rr.transformation[:3, 3]) < 2.0:
                    Tl = rr.transformation
            sub = o3d.geometry.TriangleMesh(S); m = np.ones(len(St), bool); m[tri_m] = False
            sub.remove_triangles_by_mask(m); sub.remove_unreferenced_vertices(); sub.transform(Tl)
            V.append(np.asarray(sub.vertices)); Tri.append(np.asarray(sub.triangles) + nv); nv += len(sub.vertices); adds += len(sub.triangles)
            if s not in nomes: nomes.append(s)
            Lab.append(np.full(len(sub.triangles), nomes.index(s), np.int16))
        cobertos |= cobs[s]
        cur = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.vstack(V)), o3d.utility.Vector3iVector(np.vstack(Tri)))
        ordem.append(dict(scan=s, papel="preenche lacunas", ganho=float(ganho[s]), triangulos_acrescentados=int(adds), cobertura_acumulada=float(cobertos.mean())))
        print(f"+ {s}: ganho {ganho[s]*100:.2f}% | +{adds:,} tri | cobertura acumulada {cobertos.mean()*100:.1f}% ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(a.saida, "fonte_tri.npy"), np.concatenate(Lab)); json.dump(nomes, open(os.path.join(a.saida, "fonte_nomes.json"), "w"))
    out = os.path.join(a.saida, "base_e_pedacos.ply")
    o3d.io.write_triangle_mesh(out, cur, write_vertex_normals=False)
    json.dump(dict(ordem=ordem, parametros=vars(a), cobertura_final=float(cobertos.mean())), open(os.path.join(a.saida, "relatorio.json"), "w"), indent=1)
    print(f"gravado {out}: {len(cur.triangles):,} tri, cobertura final {cobertos.mean()*100:.1f}% | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
