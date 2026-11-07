"""Fusao propria de alta nitidez para pecas pequenas com detalhes finos (2-4 scans ja limpos).

    python pipeline/fundir_nitido.py <projeto_limpo> --poses poses.json --saida PASTA [--voxel 0.1]

poses.json: {guid: 4x4} leva cada scan ao referencial comum (partida; e refinada aqui por ICP multi-vista).
1. Todos os pixels dos quadros, com UMA calibracao para todos os scans (foco gravado + centro ajustado).
2. Normal local por superficie de centroides de 0,2 mm, orientada para o lado visto pelas cameras.
3. Observacoes boas: nao rasantes (cos >= --cos-min) e perto do plano local (<= --dist-max). Viram uma grade fina
   de 0,05 mm com peso cos^2 e direcao de visada.
4. Amostras = centroides de --voxel mm com >= --min-obs observacoes, sem pontos soltos (--viz-min vizinhos a 0,3 mm).
5. Mean-shift ao longo da normal sobre a grade fina (gaussiana tangencial --sigma-t, faixa de altura decrescente):
   colapsa a espessura da superficie, que a media por voxel NAO colapsa (camadas de voxels empilhadas), sem puxar
   vizinhos do outro lado de arestas nem a face oposta. Depois, uma amostra por voxel de --voxel mm.
6. Apoio fraco: superficies com poucas observacoes (paredes de cavidade) entram em voxel de --voxel-fraco mm, longe
   (> --raio-fraco) da superficie principal e so se >= --scans-fraco scans independentes as confirmam.
7. Remove componentes soltos pequenos (--comp-min da peca).
Saida: nuvem_nitida.ply (com normais), nuvem_apoio_fraco.ply (todas as observacoes fracas confirmadas, mais ruidosas:
paredes de cavidade, sem o filtro de lamina), poses_finais.json, relatorio.json. Nao altera o projeto de entrada.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(AQUI, "revo_frames"))
from alinhar_varios import baixa_prioridade  # noqa: E402
from revo_projeto import RevoScan  # noqa: E402

WORKERS = 8
reg = o3d.pipelines.registration


def grupos(chaves):
    _, inv, n = np.unique(chaves, axis=0, return_inverse=True, return_counts=True)
    return inv.ravel(), n


def centroides(X, voxel, w=None, min_obs=1):
    inv, n = grupos(np.floor(X / voxel).astype(np.int64))
    pesos = np.ones(len(X)) if w is None else w
    soma = np.bincount(inv, pesos, len(n))
    C = np.stack([np.bincount(inv, X[:, i] * pesos, len(n)) / soma for i in range(3)], 1)
    return C[n >= min_obs], n[n >= min_obs], inv


def vizinhos(arv, Q, k, raio):
    d, i = arv.query(Q, k=k, distance_upper_bound=raio, workers=WORKERS)
    fora = ~np.isfinite(d) | (i >= arv.n)                       # maquina instavel: valida os indices devolvidos
    i = np.where(fora, 0, i)
    return d, i, fora


def normais_locais(C, ref, k=40, raio=0.5, sigma=0.25, bloco=200000):
    """PCA ponderado por gaussiana; devolve normal e razao menor/maior autovalor do plano local."""
    arv = cKDTree(C)
    N = np.zeros_like(C)
    for a in range(0, len(C), bloco):
        Q = C[a:a + bloco]
        d, i, fora = vizinhos(arv, Q, k, raio)
        w = np.exp(-(np.where(fora, 0, d) ** 2) / (2 * sigma ** 2))
        w[fora] = 0
        Y = C[i] - Q[:, None, :]
        media = (w[..., None] * Y).sum(1) / w.sum(1, keepdims=True)
        Y = Y - media[:, None, :]
        M = np.einsum("nki,nk,nkj->nij", Y, w, Y)
        _, U = np.linalg.eigh(M)
        N[a:a + bloco] = U[:, :, 0]
    return N * np.sign(np.sum(N * ref, 1, keepdims=True) + 1e-12)


def carregar(pasta, tmp, K):
    sc = RevoScan(pasta, tmp)
    ks = [k for k in range(len(sc.ids)) if sc.tem_prof[k]]
    pts, cam = [], []
    for k in ks:
        P = sc.pontos(k, K)[0]
        pts.append(P)
        cam.append(np.repeat(sc.T[k][:3, 3][None], len(P), axis=0).astype(np.float32))
    sc.c.close()
    return np.vstack(pts), np.vstack(cam), len(ks)


def icp_multivista(nuv, T):
    guids = list(nuv)
    est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=0.15))
    for tol in (1.0, 0.6, 0.4, 0.3, 0.25, 0.2, 0.2, 0.2):
        for g in guids[1:]:                                     # o primeiro scan e a referencia
            outros = o3d.geometry.PointCloud()
            for o in guids:
                if o != g:
                    q = o3d.geometry.PointCloud(nuv[o])
                    q.transform(T[o])
                    outros += q
            r = reg.registration_icp(nuv[g], outros, tol, T[g], est, reg.ICPConvergenceCriteria(max_iteration=60))
            T[g] = np.array(r.transformation)
    return T


def sem_soltos(C, ref, viz_min, raio=0.3):
    n = np.array(cKDTree(C).query_ball_point(C, raio, workers=WORKERS, return_length=True))
    ok = n > viz_min
    return C[ok], ref[ok], int((~ok).sum())


def espessura_local(C, raio, k=80, bloco=100000):
    """RMS (mm) do plano local e numero de vizinhos: superficie = fina; nuvem de multipercurso = grossa."""
    arv = cKDTree(C)
    e, nv = np.zeros(len(C)), np.zeros(len(C), int)
    for a in range(0, len(C), bloco):
        Q = C[a:a + bloco]
        _, i, fora = vizinhos(arv, Q, k, raio)
        w = (~fora).astype(float)
        Y = C[i] - Q[:, None, :]
        soma = np.maximum(w.sum(1), 1)
        Y = Y - ((w[..., None] * Y).sum(1) / soma[:, None])[:, None, :]
        M = np.einsum("nki,nk,nkj->nij", Y, w, Y)
        e[a:a + bloco] = np.sqrt(np.maximum(np.linalg.eigvalsh(M)[:, 0], 0) / soma)
        nv[a:a + bloco] = w.sum(1)
    return e, nv


def mean_shift(S, N, F, W, VF, sigma_t, sigma_h, raio=0.35, k=200, max_passo=0.1, bloco=50000):
    """Leva cada amostra S ao longo da normal N ate a moda de altura da grade fina F (pesos W, visada VF)."""
    arv = cKDTree(F)
    D = np.zeros(len(S))
    for a in range(0, len(S), bloco):
        Q, Nq = S[a:a + bloco], N[a:a + bloco]
        d, i, fora = vizinhos(arv, Q, k, raio)
        h = np.einsum("nkj,nj->nk", F[i] - Q[:, None, :], Nq)
        t2 = np.maximum(np.where(fora, 0, d) ** 2 - h ** 2, 0)
        w = W[i] * np.exp(-t2 / (2 * sigma_t ** 2)) * np.exp(-(h ** 2) / (2 * sigma_h ** 2))
        w *= np.einsum("nkj,nj->nk", VF[i], Nq) > 0.05          # face oposta de parede fina nao conta
        w[fora] = 0
        D[a:a + bloco] = np.clip((w * h).sum(1) / np.maximum(w.sum(1), 1e-9), -max_passo, max_passo)
    return S + D[:, None] * N, D


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.1, help="amostras do mean-shift")
    ap.add_argument("--voxel-saida", type=float, default=0.07, help="espacamento final depois do colapso")
    ap.add_argument("--esp-max", type=float, default=0.07, help="mm; RMS local (r=0,3) acima disso e nuvem, nao superficie")
    ap.add_argument("--esp-max-fraco", type=float, default=0.05, help="idem na camada de apoio fraco (r=0,6)")
    ap.add_argument("--cos-min", type=float, default=0.3)
    ap.add_argument("--dist-max", type=float, default=0.25)
    ap.add_argument("--min-obs", type=int, default=6)
    ap.add_argument("--viz-min", type=int, default=12)
    ap.add_argument("--sigma-t", type=float, default=0.12)
    ap.add_argument("--voxel-fraco", type=float, default=0.15)
    ap.add_argument("--min-obs-fraco", type=int, default=5)
    ap.add_argument("--raio-fraco", type=float, default=0.3)
    ap.add_argument("--scans-fraco", type=int, default=2)
    ap.add_argument("--comp-min", type=float, default=0.002)
    a = ap.parse_args()
    baixa_prioridade()
    t0 = time.time()
    if os.path.exists(a.saida) and os.listdir(a.saida):
        sys.exit(f"destino ja existe: {a.saida}")
    os.makedirs(a.saida, exist_ok=True)
    poses0 = {g: np.array(T) for g, T in json.load(open(a.poses)).items()}
    rel = dict(parametros=vars(a))

    with tempfile.TemporaryDirectory() as tmp:
        amostra = RevoScan(os.path.join(a.projeto, "data", next(iter(poses0))), tmp)
        f = amostra.foco_gravado(0)
        centros = []
        amostra.c.close()
        for g in poses0:
            sc = RevoScan(os.path.join(a.projeto, "data", g), tmp, reusar=True)
            K, _ = sc.calibrar(0)
            centros.append(K[2:])
            sc.c.close()
        cx, cy = np.median(centros, 0)
        K = (f, f, float(cx), float(cy))
        rel["K"] = K
        print("K", np.round(K, 3), flush=True)
        dados = {g: carregar(os.path.join(a.projeto, "data", g), tmp, K) for g in poses0}
    rel["quadros"] = {g: d[2] for g, d in dados.items()}
    rel["pontos_entrada"] = {g: len(d[0]) for g, d in dados.items()}

    nuv = {}
    for g, (X, _, _) in dados.items():
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(centroides(X, 0.2, min_obs=3)[0]))
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=40))
        nuv[g] = p
    T = icp_multivista(nuv, {g: x.copy() for g, x in poses0.items()})
    json.dump({g: x.tolist() for g, x in T.items()}, open(os.path.join(a.saida, "poses_finais.json"), "w"), indent=1)
    for g, x in T.items():
        D = np.linalg.inv(poses0[g]) @ x
        print(g, "ICP: %.3f graus %.3f mm" % (np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))), np.linalg.norm(D[:3, 3])), flush=True)

    X = np.vstack([d[0] @ T[g][:3, :3].T + T[g][:3, 3] for g, d in dados.items()])
    V = np.vstack([(np.asarray(d[1], float) @ T[g][:3, :3].T + T[g][:3, 3]) for g, d in dados.items()]) - X
    rot = np.concatenate([np.full(len(d[0]), i, np.int8) for i, d in enumerate(dados.values())])
    del dados
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    print("observacoes", len(X), flush=True)

    # superficie grossa e normais orientadas
    C0, _, inv = centroides(X, 0.2, min_obs=3)
    inv_todos, n_todos = grupos(np.floor(X / 0.2).astype(np.int64))
    v_med = np.stack([np.bincount(inv_todos, V[:, i], len(n_todos)) for i in range(3)], 1)[n_todos >= 3]
    N0 = normais_locais(C0, v_med)
    arv = cKDTree(C0)
    dist, idx = arv.query(X, workers=WORKERS)
    if idx.max() >= len(C0):
        raise RuntimeError("cKDTree devolveu indice invalido")
    N_obs = N0[idx]
    cosv = np.sum(N_obs * V, 1)
    h = np.sum((X - C0[idx]) * N_obs, 1)
    plano_ok = (dist < 0.4) & (np.abs(h) <= a.dist_max)
    ok = plano_ok & (cosv >= a.cos_min)
    rel["obs_boas"] = float(ok.mean())
    rel["obs_rasantes"] = float((cosv < a.cos_min).mean())
    rel["obs_fora_plano"] = float((np.abs(h) > a.dist_max).mean())
    print("obs boas %.1f%% (rasantes %.1f%%, fora do plano %.1f%%)" % (100 * ok.mean(), 100 * (cosv < a.cos_min).mean(), 100 * (np.abs(h) > a.dist_max).mean()), flush=True)

    def por_voxel(sel, w, voxel, minimo):
        Xs, Vs, ws = X[sel], V[sel], w[sel]
        inv_, nn = grupos(np.floor(Xs / voxel).astype(np.int64))
        soma = np.bincount(inv_, ws, len(nn))
        C_ = np.stack([np.bincount(inv_, Xs[:, i] * ws, len(nn)) / soma for i in range(3)], 1)
        V_ = np.stack([np.bincount(inv_, Vs[:, i] * ws, len(nn)) for i in range(3)], 1)
        m = nn >= minimo
        return C_[m], V_[m] / np.maximum(np.linalg.norm(V_[m], axis=1, keepdims=True), 1e-12), soma[m], nn[m]

    pesos = np.maximum(cosv, 0.05) ** 2
    F, VF, W, _ = por_voxel(ok, pesos, 0.05, 1)
    S, ref, _, n = por_voxel(ok, pesos, a.voxel_saida, a.min_obs)
    rel["grade_fina"] = len(F)
    rel["amostras_iniciais"] = len(S)
    rel["obs_por_voxel_mediana"] = float(np.median(n))
    print("grade fina", len(F), "| amostras", len(S), "| obs/voxel", np.median(n), flush=True)
    S, ref, soltos = sem_soltos(S, ref, a.viz_min)     # grade mais densa: so tira o que esta realmente solto
    rel["soltos_removidos"] = soltos
    print("soltos removidos", soltos, flush=True)

    for sig_h in (0.12, 0.10, 0.08, 0.06, 0.05):
        N = normais_locais(S, ref, k=48, raio=0.4, sigma=0.18)
        S, D = mean_shift(S, N, F, W, VF, a.sigma_t, sig_h)
        rel[f"desloc_h{sig_h}_um"] = dict(mediana=float(np.median(np.abs(D)) * 1000), p99=float(np.percentile(np.abs(D), 99) * 1000))
        print("mean-shift", sig_h, rel[f"desloc_h{sig_h}_um"], flush=True)
    del F, VF, W

    # uma amostra por voxel: as camadas colapsadas caem no mesmo voxel
    inv_, nn = grupos(np.floor(S / a.voxel_saida).astype(np.int64))
    C = np.stack([np.bincount(inv_, S[:, i], len(nn)) / nn for i in range(3)], 1)
    ref = np.stack([np.bincount(inv_, ref[:, i], len(nn)) for i in range(3)], 1)
    rel["amostras_apos_colapso"] = len(C)
    print("apos colapso", len(S), "->", len(C), flush=True)
    C, ref, soltos = sem_soltos(C, ref, 12)
    rel["soltos_removidos_fim"] = soltos
    e, _ = espessura_local(C, 0.3)
    fina = e <= a.esp_max
    rel["nuvem_removida_principal"] = int((~fina).sum())
    print("espessura local: mediana %.1f um p99 %.1f um | removidos > %.0f um: %d" % (np.median(e) * 1000, np.percentile(e, 99) * 1000, a.esp_max * 1000, (~fina).sum()), flush=True)
    C, ref = C[fina], ref[fina]
    C, ref, soltos = sem_soltos(C, ref, 12)
    rel["soltos_removidos_fim2"] = soltos

    # apoio fraco: confirmado por >= scans-fraco scans e longe da superficie principal
    d_princ = cKDTree(C).query(X, workers=WORKERS)[0]
    fracos = np.flatnonzero((d_princ > a.raio_fraco) & (np.abs(h) <= 1.0))
    if len(fracos):
        qf = np.floor(X[fracos] / a.voxel_fraco).astype(np.int64)
        inv_, nn = grupos(qf)
        cont = np.zeros((len(nn), len(np.unique(rot))), bool)
        cont[inv_, rot[fracos]] = True
        m = (nn >= a.min_obs_fraco) & (cont.sum(1) >= a.scans_fraco)
        Cf = np.stack([np.bincount(inv_, X[fracos, i], len(nn)) / nn for i in range(3)], 1)[m]
        Vf = np.stack([np.bincount(inv_, V[fracos, i], len(nn)) for i in range(3)], 1)[m]
        if len(Cf):
            camada = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Cf))   # todas as confirmadas, em arquivo separado
            camada.normals = o3d.utility.Vector3dVector(normais_locais(Cf, Vf, k=60, raio=0.6, sigma=0.3))
            o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_apoio_fraco.ply"), camada, write_ascii=False)
            ef, nv = espessura_local(Cf, 0.6)
            sup = (ef <= a.esp_max_fraco) & (nv >= 20)         # lamina, nao volume: multipercurso e descartado
            n_antes = len(Cf)
            Cf, Vf = Cf[sup], Vf[sup]
            Nf = normais_locais(Cf, Vf, k=60, raio=0.6, sigma=0.3)
            Xf, Vxf = X[fracos], V[fracos]
            for sig_h in (0.1, 0.07, 0.05):                    # mesmo colapso, sobre as observacoes da propria camada
                Cf, _ = mean_shift(Cf, Nf, Xf, np.ones(len(Xf)), Vxf, 0.2, sig_h, raio=0.5, k=120)
                Nf = normais_locais(Cf, Vf, k=60, raio=0.6, sigma=0.3)
        rel["apoio_fraco"] = dict(obs=len(fracos), voxels_candidatos=len(nn), voxels_um_scan=int(((nn >= a.min_obs_fraco) & (cont.sum(1) < a.scans_fraco)).sum()), confirmados=n_antes if len(Cf) else 0, pontos=len(Cf))
        print("apoio fraco", rel["apoio_fraco"], flush=True)
        C, ref = np.vstack([C, Cf]), np.vstack([ref, Vf])
    del X, V
    N = normais_locais(C, ref, k=48, raio=0.5, sigma=0.22)

    # componentes soltos (grade de 0,5 mm)
    g = np.floor((C - C.min(0)) / 0.5).astype(np.int64)
    vol = np.zeros(g.max(0) + 1, bool)
    vol[tuple(g.T)] = True
    lab, ncomp = ndimage.label(vol, structure=np.ones((3, 3, 3)))
    lp = lab[tuple(g.T)]
    tam = np.bincount(lp, minlength=ncomp + 1)
    manter = tam[lp] >= a.comp_min * len(C)
    rel["componentes"] = dict(total=int(ncomp), pontos_removidos=int((~manter).sum()), maiores=sorted(tam[1:].tolist(), reverse=True)[:8])
    print("componentes", rel["componentes"], flush=True)
    C, N = C[manter], N[manter]

    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(C))
    p.normals = o3d.utility.Vector3dVector(N)
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_nitida.ply"), p, write_ascii=False)
    rel["pontos_saida"] = len(C)
    rel["segundos"] = round(time.time() - t0)
    json.dump(rel, open(os.path.join(a.saida, "relatorio.json"), "w"), indent=1, default=float)
    print("pontos saida", len(C), "| extensao", np.round(np.ptp(C, axis=0), 2), "|", rel["segundos"], "s", flush=True)


if __name__ == "__main__":
    main()
