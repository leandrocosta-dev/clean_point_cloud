"""Limpeza da nuvem: base, ruido, ilhas e trechos deslocados.

ORDEM IMPORTA: o realinhamento acontece ANTES de descartar clusters
secundarios. Senao um trecho deslocado (que e peca real) seria jogado fora
como se fosse ilha de ruido.

Rotulos por ponto:
    0          peca (cluster principal)
    1..k       trecho deslocado realinhado (transformacao k)
    100+i      candidato a trecho deslocado (modo sem alinhamento)
   -2          base / mesa
   -3          ruido fino (SOR)
   -4          ilha flutuante ou trecho rejeitado
   -5          duplicado na sobreposicao apos realinhar
   -6          trecho com encaixe ambiguo (salvo a parte para revisao manual)
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .geom import to_pcd, plane_basis, apply_T
from .align import register

MAIN, BASE, NOISE, CLUSTER, DUP, AMBIG = 0, -2, -3, -4, -5, -6
CAND_OFFSET = 100


def _trimmed_area(p2, q=2.0):
    """Area do retangulo entre os percentis q e 100-q (ignora outliers)."""
    lo = np.percentile(p2, q, axis=0)
    hi = np.percentile(p2, 100 - q, axis=0)
    return float(np.prod(hi - lo))


def detect_base(pts, spacing, P, mode, seed):
    """Detecta o plano da base. Retorna dict(n, d, band) ou None.

    No modo auto, so aceita como base um plano que:
      1. deixa >= 90% do resto da nuvem de um lado so;
      2. tem area projetada maior que a da peca (mesa e maior que a peca).
    O criterio 2 evita confundir a face plana de uma peca com a base.
    """
    if mode == "off":
        return None
    o3d.utility.random.seed(seed)
    dist = P.base_dist_mult * spacing
    model, inl = to_pcd(pts).segment_plane(dist, 3, 3000)
    n = np.array(model[:3], float)
    k = np.linalg.norm(n)
    n, d = n / k, model[3] / k
    inl = np.asarray(inl)
    if mode == "auto" and len(inl) < 0.05 * len(pts):
        return None
    sd = pts @ n + d
    rest = np.ones(len(pts), bool)
    rest[inl] = False
    pos = float(np.mean(sd[rest] > 0)) if rest.any() else 1.0
    if pos < 0.5:
        n, d, sd, pos = -n, -d, -sd, 1 - pos
    if mode == "auto":
        if pos < 0.9:
            return None
        obj = sd > 2 * dist
        if obj.sum() < 100:
            return None
        u, v = plane_basis(n)
        a_base = _trimmed_area(np.c_[pts[inl] @ u, pts[inl] @ v])
        a_obj = _trimmed_area(np.c_[pts[obj] @ u, pts[obj] @ v])
        if a_base < 1.3 * a_obj:
            return None
    band = dist + P.base_margin_mult * spacing
    return {"n": n.tolist(), "d": float(d), "band": float(band), "n_inliers": int(len(inl))}


def run_clean(pts, spacing, P, cfg, align=True, base_fixa=None, escolher=None):
    """Executa a limpeza. Retorna (labels, info, transforms{k: T}).

    base_fixa: dict(n, d, band) ja conhecido (ex.: prato achado pelas cameras no projeto Revo);
        substitui a deteccao.
    escolher(idx, cl, sizes) -> (id do cluster principal, ids candidatos a trecho em ordem de tamanho).
        Padrao: o maior cluster e a peca e todos os outros sao candidatos.
    """
    o3d.utility.random.seed(cfg.seed)
    n = len(pts)
    labels = np.full(n, MAIN, int)
    info = {"fragmentos": []}
    transforms = {}

    # 1. Base
    base = base_fixa if base_fixa is not None else detect_base(pts, spacing, P, cfg.base_mode, cfg.seed)
    info["base"] = base
    if base:
        sd = pts @ np.array(base["n"]) + base["d"]
        labels[sd < base["band"]] = BASE

    # 2. Ruido fino
    idx = np.where(labels == MAIN)[0]
    if len(idx) < 50:
        info["erro"] = "quase nada sobrou apos remover a base"
        return labels, info, transforms
    _, keep = to_pcd(pts[idx]).remove_statistical_outlier(int(P.sor_nb), float(P.sor_std))
    m = np.zeros(len(idx), bool)
    m[np.asarray(keep, int)] = True
    labels[idx[~m]] = NOISE
    idx = idx[m]

    # 3. Clusters
    cl = np.asarray(to_pcd(pts[idx]).cluster_dbscan(P.dbscan_eps_mult * spacing, int(P.dbscan_min_points)))
    if cl.max() < 0:
        labels[idx] = CLUSTER
        info["erro"] = "DBSCAN nao encontrou nenhum cluster"
        return labels, info, transforms
    sizes = np.bincount(cl[cl >= 0])
    if escolher:
        main_id, candidatos = escolher(idx, cl, sizes)
    else:
        order = np.argsort(sizes)[::-1]
        main_id, candidatos = order[0], order[1:]
    labels[idx[cl != main_id]] = CLUSTER
    main_pts = pts[idx[cl == main_id]]
    info["n_clusters"] = int(len(sizes))

    # 4. Trechos deslocados
    k = 0
    for i, cid in enumerate(candidatos):
        if sizes[cid] < P.fragment_min_frac * sizes[main_id]:
            break
        fidx = idx[cl == cid]
        if not align:
            labels[fidx] = CAND_OFFSET + i
            info["fragmentos"].append({"n_pontos": int(len(fidx)), "status": "candidato"})
            continue
        T, fit, rmse, extra = register(pts[fidx], main_pts, spacing, cfg.seed,
                                       cfg.align_min_fitness, cfg.align_max_rmse_mult, base=base)
        ok = fit >= cfg.align_min_fitness and rmse <= cfg.align_max_rmse_mult * spacing
        rec = {"n_pontos": int(len(fidx)), "fitness": round(fit, 3),
               "rmse": round(rmse, 4), "status": "realinhado" if ok else "rejeitado",
               **extra}
        if ok and extra.get("ambiguo"):
            # Nao mistura na nuvem: um encaixe errado contamina a peca.
            # Guarda a melhor sugestao para revisao manual.
            labels[fidx] = AMBIG
            rec["status"] = "ambiguo"
            info.setdefault("sugestoes", []).append({"idx": fidx, "T": T})
        elif ok:
            k += 1
            moved = apply_T(T, pts[fidx])
            dd, _ = cKDTree(main_pts).query(moved, k=1, workers=-1)
            dup = dd < cfg.dedupe_mult * spacing
            labels[fidx[~dup]] = k
            labels[fidx[dup]] = DUP
            main_pts = np.vstack([main_pts, moved[~dup]])
            transforms[k] = T
            rec.update(label=k, duplicados=int(dup.sum()), T=np.round(T, 6).tolist())
        info["fragmentos"].append(rec)

    return labels, info, transforms


def final_points(pts, labels, transforms, normals=None):
    """Pontos mantidos, ja com os trechos realinhados na posicao correta.

    Retorna (pontos, mascara dos mantidos, normais giradas junto ou None).
    """
    keep = labels >= 0
    out = pts[keep].copy()
    nrm = normals[keep].copy() if normals is not None else None
    lk = labels[keep]
    for k, T in transforms.items():
        m = lk == k
        out[m] = apply_T(T, out[m])
        if nrm is not None:
            nrm[m] = nrm[m] @ np.asarray(T)[:3, :3].T
    return out, keep, nrm
