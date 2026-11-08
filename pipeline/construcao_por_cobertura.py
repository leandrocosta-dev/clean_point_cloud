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
import time

import gpu
import numpy as np
import open3d as o3d
from metricas import componentes_grafo, icp_robusto, normais_da_nuvem, normais_hibridas
from recursos import baixa_prioridade

# Calculo numerico na GPU (gpu.py): vizinhos, ICP, normais, componentes, centroides. Fica na CPU so o que e malha do
# Open3D: leitura/gravacao, transform, sample_points_uniformly, remove_*_by_mask, cluster_connected_triangles.


def alinhar_e_limpar(proj, scan, ind, pj, Tr, Ep, En, tol_limpa, T0=None):
    """Ep, En: pontos e normais (numpy) da nuvem externa."""
    if T0 is None:
        seg = max(ind, key=lambda s: len(np.load(s["arquivo"])))
        T0 = Tr @ np.array(next(x for x in pj if x["seg"] == seg["seg"])["T"])
    M = o3d.io.read_triangle_mesh(os.path.join(proj, "data", scan, "fuse_mesh.ply"))
    M.transform(T0)
    S = np.asarray(M.sample_points_uniformly(600_000).points)
    T, info = icp_robusto(S, Ep, En, tols=(8, 4, 2, 1.2), iteracoes=40)  # ICP ponto-plano robusto na GPU
    M.transform(T)
    _, j = gpu.vizinho_mais_proximo(Ep, gpu.para_gpu(np.asarray(M.vertices)), dist_max=tol_limpa)
    keep = gpu.para_cpu((j >= 0)[gpu.para_gpu(np.asarray(M.triangles))].all(1))
    M.remove_triangles_by_mask(~keep)
    M.remove_unreferenced_vertices()
    cl, n, _ = M.cluster_connected_triangles()
    cl = np.asarray(cl)
    n = np.asarray(n)
    M.remove_triangles_by_mask(np.isin(cl, np.flatnonzero(n < 500)))
    M.remove_unreferenced_vertices()
    return M, info["fitness"], info["rmse"]


def amostrar(M, n=1_500_000):
    return np.asarray(M.sample_points_uniformly(n).points)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("projeto", "pasta_seg", "direto", "tseg", "externa"):
        ap.add_argument(k)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--tol-limpa", type=float, default=1.5)
    ap.add_argument("--tol-lacuna", type=float, default=1.5)
    ap.add_argument("--ganho-min", type=float, default=0.003)
    ap.add_argument("--margem", type=float, default=1.0)
    ap.add_argument(
        "--so-externos",
        action="store_true",
        help="ignora scans cuja malha esta orientada p/ dentro (viram a parede por dentro: sao da face INTERNA, ~1,6 mm deslocados)",
    )
    ap.add_argument(
        "--so-internos",
        action="store_true",
        help="o inverso: usa so os scans da face INTERNA, sem inverter (malha fica com normais para dentro da cavidade)",
    )
    ap.add_argument(
        "--cob-tol",
        type=float,
        default=1.0,
        help="mm: distancia a nuvem externa que conta como cobertura (use ~4 para a face interna)",
    )
    ap.add_argument(
        "--espessura",
        type=float,
        default=0.0,
        help="mm: alinha/apara contra a nuvem externa deslocada para DENTRO por essa espessura (face interna)",
    )
    ap.add_argument(
        "--extras",
        help="json: lista de {proj, scan, T} de scans de OUTROS projetos/dias (T = pose inicial scan->referencial)",
    )
    ap.add_argument("--apenas", nargs="*", help="so estes scans do projeto principal (nomes)")
    ap.add_argument("--min-tri", type=int, default=500)
    ap.add_argument("--abertura", type=float, default=3.0)
    a = ap.parse_args()
    baixa_prioridade()
    t0 = time.time()
    os.makedirs(os.path.join(a.saida, "scans"), exist_ok=True)
    ind = json.load(open(os.path.join(a.pasta_seg, "segmentos_indice.json")))
    pj = json.load(open(a.direto))["segmentos"]
    Tr = np.load(a.tseg)
    E = o3d.io.read_point_cloud(a.externa)
    if a.espessura:
        En0 = np.asarray(E.normals).copy()
        E.points = o3d.utility.Vector3dVector(np.asarray(E.points) - a.espessura * En0)
        E.normals = o3d.utility.Vector3dVector(En0)
    normais_da_nuvem(E, 4, 30)  # GPU; mantem o lado das normais ja gravadas
    Ep = np.asarray(E.points)
    rng = np.random.default_rng(0)
    Ea = gpu.para_gpu(Ep[rng.choice(len(Ep), min(len(Ep), 1_500_000), replace=False)])  # amostra para cobertura
    En = np.asarray(E.normals)
    mg = gpu.xp()
    # 1. alinhar e aparar cada scan
    meshes, cobs = {}, {}
    fontes = [
        (scan, a.projeto, scan, ind[scan], pj[scan], None) for scan in sorted(ind) if not a.apenas or scan in a.apenas
    ]
    if a.extras:
        for e in json.load(open(a.extras)):
            fontes.append(("extra_" + e["scan"], e["proj"], e["scan"], None, None, np.array(e["T"])))
    for scan, proj_f, real, ind_s, pj_s, T_ini in fontes:
        f = os.path.join(a.saida, "scans", scan + ".ply")
        if os.path.exists(f):
            M = o3d.io.read_triangle_mesh(f)
        else:
            if not os.path.exists(os.path.join(proj_f, "data", real, "fuse_mesh.ply")):
                print(scan, "sem fuse_mesh, ignorado")
                continue
            M, fit, rmse = alinhar_e_limpar(proj_f, real, ind_s, pj_s, Tr, Ep, En, a.tol_limpa, T_ini)
            o3d.io.write_triangle_mesh(f, M, write_vertex_normals=False)
        if len(M.triangles) == 0:
            continue
        M.compute_triangle_normals()
        Tn = gpu.para_gpu(np.asarray(M.triangle_normals))
        Cn = gpu.para_gpu(np.asarray(M.vertices))[gpu.para_gpu(np.asarray(M.triangles))].mean(1)
        _, jn = gpu.vizinho_mais_proximo(Ep, Cn[::7])
        conc = float((mg.einsum("ij,ij->i", Tn[::7], gpu.para_gpu(En)[jn]) > 0).mean())
        if conc < 0.5 and a.so_externos:
            print(
                f"{scan}: orientacao concorda {conc * 100:.0f}% -> scan da face INTERNA, ignorado (--so-externos)",
                flush=True,
            )
            continue
        if conc >= 0.5 and a.so_internos:
            print(
                f"{scan}: orientacao concorda {conc * 100:.0f}% -> scan externo, ignorado (--so-internos)", flush=True
            )
            continue
        if (
            conc < 0.5 and not a.so_internos
        ):  # malha do scan orientada para dentro: inverte o scan inteiro (faces de verso viram de frente)
            Tt = np.asarray(M.triangles)[:, ::-1].copy()
            M.triangles = o3d.utility.Vector3iVector(Tt)
            M.compute_triangle_normals()
        print(
            f"{scan}: orientacao concorda com a nuvem externa em {conc * 100:.0f}%"
            + (" -> invertida" if conc < 0.5 and not a.so_internos else ""),
            flush=True,
        )
        meshes[scan] = M
        _, jc = gpu.vizinho_mais_proximo(amostrar(M), Ea, dist_max=a.cob_tol)
        cobs[scan] = jc >= 0
        print(
            f"{scan}: {len(M.triangles):,} tri | cobre {float(cobs[scan].mean()) * 100:.1f}% da pele externa a 1 mm ({time.time() - t0:.0f}s)",
            flush=True,
        )
    # 2. base + guloso
    base = max(cobs, key=lambda s: float(cobs[s].mean()))
    ordem = [dict(scan=base, papel="base", cobertura=float(cobs[base].mean()))]
    B = meshes[base]
    V = [np.asarray(B.vertices)]
    Tri = [np.asarray(B.triangles)]
    nv = len(V[0])
    Lab = [np.zeros(len(Tri[0]), np.int16)]
    nomes = [base]
    cobertos = cobs[base].copy()
    restantes = [s for s in meshes if s != base]
    cur = o3d.geometry.TriangleMesh(B)
    while restantes:
        ganho = {s: float((cobs[s] & ~cobertos).mean()) for s in restantes}
        s = max(ganho, key=ganho.get)
        if ganho[s] < a.ganho_min:
            break
        restantes.remove(s)
        Cs = amostrar(cur, 2_500_000)
        S = meshes[s]
        Sv = np.asarray(S.vertices)
        St = np.asarray(S.triangles)
        Svg = gpu.para_gpu(Sv)
        cen = Svg[gpu.para_gpu(St)].mean(1)
        _, jc = gpu.vizinho_mais_proximo(Cs, cen, dist_max=a.tol_lacuna)
        lac = jc < 0
        # componentes das triangulos de lacuna (adjacencia por vertice)
        # abertura: so o interior das lacunas (>= 60% da vizinhanca de 3 mm tambem e lacuna) -> descarta tiras finas e bordas rasgadas
        # fracao de lacuna na vizinhanca = (centroides de lacuna no raio) / (todos os centroides no raio), por contagem na GPU
        if not bool(lac.any()):
            continue
        cl = cen[lac]
        frac = gpu.contar_no_raio(cl, cl, a.abertura) / gpu.contar_no_raio(cen, cl, a.abertura)
        idx = gpu.para_cpu(mg.flatnonzero(lac)[frac >= 0.6])
        if len(idx) == 0:
            continue
        _, inv, _ = gpu.agrupar(gpu.para_gpu(St[idx].ravel()))
        inv = inv.reshape(-1, 3)
        _, lab = componentes_grafo(
            int(inv.max()) + 1, mg.concatenate([inv[:, 0], inv[:, 1]]), mg.concatenate([inv[:, 1], inv[:, 2]])
        )
        lab = lab[gpu.para_cpu(inv[:, 0])]
        tam = np.bincount(lab)  # triangulos que compartilham vertice = mesmo componente
        adds = 0
        alvo_icp = None
        for c in np.flatnonzero(tam >= a.min_tri):
            tri_c = idx[lab == c]
            # margem: triangulos de S cujo centroide esta a <= margem dos triangulos da lacuna
            cc = cen[gpu.para_gpu(tri_c)]
            _, jm = gpu.vizinho_mais_proximo(cc, cen, dist_max=a.margem)
            tri_m = gpu.para_cpu(mg.flatnonzero(jm >= 0))
            # realinhamento local: vertices de S a <= 25 mm do remendo contra a malha atual
            _, jv = gpu.vizinho_mais_proximo(cc[:: max(1, len(cc) // 300)], Svg, dist_max=25.0)
            viz = gpu.para_cpu(mg.flatnonzero(jv >= 0))
            _, jd = gpu.vizinho_mais_proximo(Cs, Svg[gpu.para_gpu(viz)], dist_max=3.0)
            ok = gpu.para_cpu(jd >= 0)
            Tl = np.eye(4)
            if ok.sum() > 2000:
                if alvo_icp is None:  # alvo (malha atual a 1/2) e suas normais: uma vez por scan
                    alvo_icp = (Cs[::2], normais_hibridas(Cs[::2], 4, 30))
                Trr, _ = gpu.icp_ponto_plano(
                    Sv[viz][ok],
                    alvo_icp[0],
                    alvo_icp[1],
                    T0=np.eye(4),
                    dist_max=1.5,
                    iteracoes=30,
                    perda="tukey",
                    c=0.4,
                )
                if np.linalg.norm(Trr[:3, 3]) < 2.0:
                    Tl = Trr
            sub = o3d.geometry.TriangleMesh(S)
            m = np.ones(len(St), bool)
            m[tri_m] = False
            sub.remove_triangles_by_mask(m)
            sub.remove_unreferenced_vertices()
            sub.transform(Tl)
            V.append(np.asarray(sub.vertices))
            Tri.append(np.asarray(sub.triangles) + nv)
            nv += len(sub.vertices)
            adds += len(sub.triangles)
            if s not in nomes:
                nomes.append(s)
            Lab.append(np.full(len(sub.triangles), nomes.index(s), np.int16))
        cobertos |= cobs[s]
        cobertura_ac = float(cobertos.mean())
        cur = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.vstack(V)), o3d.utility.Vector3iVector(np.vstack(Tri))
        )
        ordem.append(
            dict(
                scan=s,
                papel="preenche lacunas",
                ganho=float(ganho[s]),
                triangulos_acrescentados=int(adds),
                cobertura_acumulada=cobertura_ac,
            )
        )
        print(
            f"+ {s}: ganho {ganho[s] * 100:.2f}% | +{adds:,} tri | cobertura acumulada {cobertura_ac * 100:.1f}% ({time.time() - t0:.0f}s)",
            flush=True,
        )
    np.save(os.path.join(a.saida, "fonte_tri.npy"), np.concatenate(Lab))
    json.dump(nomes, open(os.path.join(a.saida, "fonte_nomes.json"), "w"))
    out = os.path.join(a.saida, "base_e_pedacos.ply")
    o3d.io.write_triangle_mesh(out, cur, write_vertex_normals=False)
    json.dump(
        dict(ordem=ordem, parametros=vars(a), cobertura_final=float(cobertos.mean())),
        open(os.path.join(a.saida, "relatorio.json"), "w"),
        indent=1,
    )
    print(
        f"gravado {out}: {len(cur.triangles):,} tri, cobertura final {float(cobertos.mean()) * 100:.1f}% | {time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
