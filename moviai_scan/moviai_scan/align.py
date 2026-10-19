"""Realinhamento de trechos deslocados por perda de rastreamento.

PROBLEMA DO CRITERIO INGENUO: escolher o encaixe com maior fitness (mais
pontos encostados) falha em pecas com simetria. O trecho deslocado "cabe"
espelhado sobre outra regiao da peca e cobre MAIS area que o encaixe certo,
que so tem sobreposicao parcial. Foi exatamente o que o teste sintetico pegou.

CRITERIO USADO AQUI: costura de bordas.
Quando o rastreamento cai, a nuvem principal termina numa borda aberta
(borda de corte) e o trecho deslocado comeca em outra. No encaixe correto,
essas bordas de corte ficam por dentro da superficie da outra parte e
deixam de ser borda ("cicatrizam"). No encaixe errado, pelo menos uma delas
continua aberta. Bordas reais (ex.: aro de baixo apoiado na base) continuam
abertas nos dois casos, entao nao atrapalham a comparacao.

Geracao de candidatos:
  - RANSAC com FPFH na nuvem inteira (varias sementes)
  - RANSAC so na regiao perto das bordas abertas (onde fica a sobreposicao)
Cada candidato e refinado por ICP ponto-plano e pontuado pela costura.
"""
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from scipy.cluster.vq import kmeans2

from .geom import to_pcd, apply_T

reg = o3d.pipelines.registration
STAB_MIN = 0.02  # abaixo disso o encaixe e considerado instavel


def boundary_mask(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Pontos de borda aberta (fim da superficie escaneada)."""
    pc = o3d.t.geometry.PointCloud(o3d.core.Tensor(pts.astype(np.float32)))
    pc.estimate_normals(radius=4 * spacing, max_nn=30)
    _, mask = pc.compute_boundary_points(4 * spacing, 30, 90.0)
    return mask.numpy().astype(bool)


def _prep(pts, voxel):
    pcd = to_pcd(pts).voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    fpfh = reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return pcd, fpfh


def _ransac(src_pts, tgt_pts, voxel, seed):
    o3d.utility.random.seed(seed)
    src, sf = _prep(src_pts, voxel)
    tgt, tf = _prep(tgt_pts, voxel)
    dist = voxel * 1.5
    r = reg.registration_ransac_based_on_feature_matching(
        src, tgt, sf, tf, True, dist,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         reg.CorrespondenceCheckerBasedOnDistance(dist)],
        reg.RANSACConvergenceCriteria(200_000, 0.999))
    return np.asarray(r.transformation)


def _icp(src_fine, tgt_fine, T, voxel, spacing):
    for thr in (voxel * 1.5, spacing * 2.0):
        T = reg.registration_icp(src_fine, tgt_fine, thr, T,
                                 reg.TransformationEstimationPointToPlane(),
                                 reg.ICPConvergenceCriteria(max_iteration=60)).transformation
    ev = reg.evaluate_registration(src_fine, tgt_fine, spacing * 1.5, T)
    return np.asarray(T), float(ev.fitness), float(ev.inlier_rmse)


def _near(pts, mask, dist):
    """Pontos a menos de dist de algum ponto marcado em mask."""
    if mask.sum() == 0:
        return np.zeros(len(pts), bool)
    d, _ = cKDTree(pts[mask]).query(pts, k=1, workers=-1)
    return d < dist


def _near_base(pts, base, spacing):
    if not base:
        return np.zeros(len(pts), bool)
    sd = pts @ np.asarray(base["n"]) + base["d"]
    return sd < base["band"] + 4 * spacing


def seam_score(src_pts, tgt_pts, T, b_src, b_tgt, spacing, base=None):
    """Fracao das bordas de CORTE que cicatrizaram apos a uniao (0 a 1).

    Bordas encostadas na base (aro de baixo da peca) sao bordas naturais:
    nunca fecham, entao ficam fora da conta.
    """
    moved = apply_T(T, src_pts)
    b_tgt = b_tgt & ~_near_base(tgt_pts, base, spacing)
    b_src = b_src & ~_near_base(moved, base, spacing)
    union = np.vstack([tgt_pts, moved])
    b_union = boundary_mask(union, spacing)
    bt = b_union[:len(tgt_pts)]
    bs = b_union[len(tgt_pts):]
    healed_t = (b_tgt & ~bt).sum() / max(1, b_tgt.sum())
    healed_s = (b_src & ~bs).sum() / max(1, b_src.sum())
    # Media geometrica: exige que os DOIS lados cicatrizem
    return float(np.sqrt(healed_t * healed_s)), float(healed_t), float(healed_s)


def stability(src_pts, tgt_pcd, T, spacing):
    """Estabilidade do encaixe (analise de degenerescencia do ICP ponto-plano).

    Usa so a regiao de sobreposicao. Se as normais ali nao "travam" alguma
    direcao (ex.: parede reta, extrusao, cilindro liso), o trecho pode
    deslizar ou girar sem piorar o encaixe. Retorna razao min/max dos
    autovalores de translacao e de rotacao (perto de 0 = instavel).
    """
    moved = apply_T(T, src_pts)
    tgt = np.asarray(tgt_pcd.points)
    nrm = np.asarray(tgt_pcd.normals)
    d, j = cKDTree(tgt).query(moved, k=1, workers=-1)
    ok = d < 1.5 * spacing
    if ok.sum() < 30:
        return 0.0, 0.0
    p = moved[ok]
    n = nrm[j[ok]]
    p = p - p.mean(0)
    scale = np.sqrt((p ** 2).sum(1).mean()) + 1e-9
    tr = n.T @ n
    rc = np.cross(p / scale, n)
    rot = rc.T @ rc
    et = np.linalg.eigvalsh(tr)
    er = np.linalg.eigvalsh(rot)
    return float(et[0] / et[-1]), float(er[0] / (er[-1] + 1e-12))


def _same_T(a, b, ang_deg=3.0, trans=1.0):
    R = a[:3, :3].T @ b[:3, :3]
    ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    return ang < ang_deg and np.linalg.norm(a[:3, 3] - b[:3, 3]) < trans


def register(source_pts, target_pts, spacing, seed=42, min_fitness=0.1, max_rmse_mult=1.5,
             feature_mult=3.0, n_seeds=3, base=None):
    """Alinha source sobre target escolhendo pelo criterio de costura.

    Retorna (T, fitness, rmse, extra) onde extra traz a pontuacao de costura,
    candidatos avaliados e se o resultado e ambiguo.
    """
    voxel = spacing * feature_mult
    # Trabalha numa resolucao intermediaria para pontuar rapido
    w = spacing * 1.5
    src_w = np.asarray(to_pcd(source_pts).voxel_down_sample(w).points)
    tgt_w = np.asarray(to_pcd(target_pts).voxel_down_sample(w).points)
    sw = w  # espacamento aproximado da nuvem reduzida
    b_src = boundary_mask(src_w, sw)
    b_tgt = boundary_mask(tgt_w, sw)

    # Regiao de costura: perto das bordas abertas
    diag = np.linalg.norm(src_w.max(0) - src_w.min(0))
    band = 0.35 * diag
    seam_src = src_w[_near(src_w, b_src, band)]
    seam_tgt = tgt_w[_near(tgt_w, b_tgt, band)]

    src_fine, tgt_fine = to_pcd(src_w), to_pcd(tgt_w)
    for p in (src_fine, tgt_fine):
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=w * 3, max_nn=30))

    raw = []
    for i in range(n_seeds):
        raw.append(("inteira", _ransac(src_w, tgt_w, voxel, seed + i)))
        if len(seam_src) > 200 and len(seam_tgt) > 200:
            raw.append(("costura", _ransac(seam_src, seam_tgt, voxel, seed + 100 + i)))
    # Pedacos do trecho registrados separadamente: evita que a maior
    # simetria da peca domine todos os candidatos
    n_patch = 6 if len(src_w) > 6 * 400 else 0
    if n_patch:
        _, lab = kmeans2(src_w, n_patch, seed=seed, minit="++")
        for k in range(n_patch):
            if (lab == k).sum() > 300:
                raw.append(("pedaco", _ransac(src_w[lab == k], tgt_w, voxel, seed + 200 + k)))

    cands = []
    for origem, T0 in raw:
        T, fit, rmse = _icp(src_fine, tgt_fine, T0, voxel, spacing)
        if fit < min_fitness or rmse > max_rmse_mult * spacing:
            continue
        if any(_same_T(T, c["T"]) for c in cands):
            continue
        sc, ht, hs = seam_score(src_w, tgt_w, T, b_src, b_tgt, sw, base)
        st_t, st_r = stability(src_w, tgt_fine, T, spacing)
        cands.append({"T": T, "fitness": fit, "rmse": rmse, "costura": sc,
                      "cicatriz_principal": ht, "cicatriz_trecho": hs, "origem": origem,
                      "estab_t": st_t, "estab_r": st_r,
                      "estavel": st_t >= STAB_MIN and st_r >= STAB_MIN})

    if not cands:
        return np.eye(4), 0.0, 9e9, {"candidatos": 0, "ambiguo": False}

    # Ranking: encaixes estaveis primeiro, depois maior costura.
    # Um encaixe instavel pode estar deslizado mesmo com costura alta.
    cands.sort(key=lambda c: (c["estavel"], c["costura"], c["fitness"]), reverse=True)
    best = cands[0]
    st_t, st_r = best["estab_t"], best["estab_r"]
    instavel = not best["estavel"]
    # Ambiguidade:
    #  - rival ESTAVEL que fecha quase tao bem: simetria discreta (ex.: espelhado)
    #  - rival INSTAVEL que fecha MELHOR: sobreposicao lisa, o escolhido e so
    #    um dos muitos encaixes possiveis ao longo do deslizamento.
    #  Rival instavel que fecha igual ou pior e so uma versao deslizada do
    #  proprio escolhido, nao conta.
    rivais = [c for c in cands[1:]
              if (c["estavel"] and c["costura"] >= 0.85 * best["costura"])
              or (not c["estavel"] and c["costura"] > best["costura"] + 0.05)]
    concorrente = bool(rivais)
    ambiguo = instavel or concorrente or best["costura"] < 0.1
    motivo = []
    if instavel:
        motivo.append("sobreposicao sem detalhe geometrico: encaixe pode deslizar/girar")
    if any(c["estavel"] for c in rivais):
        motivo.append("outro encaixe diferente pontua quase igual (simetria)")
    if any(not c["estavel"] for c in rivais):
        motivo.append("sobreposicao lisa: existe encaixe deslizado que fecha as bordas melhor")
    if best["costura"] < 0.1:
        motivo.append("bordas de corte nao fecharam")
    extra = {
        "costura": round(best["costura"], 3),
        "estabilidade_translacao": round(st_t, 4),
        "estabilidade_rotacao": round(st_r, 4),
        "candidatos": len(cands),
        "ambiguo": bool(ambiguo),
        "motivo_ambiguidade": motivo,
        "alternativas": [{k: (round(v, 3) if isinstance(v, float) else v)
                          for k, v in c.items() if k != "T"} for c in cands[:4]],
    }
    extra["_todos"] = cands
    return best["T"], best["fitness"], best["rmse"], extra


def merge_scans(scans: list, spacing: float, seed: int = 42, min_fitness: float = 0.2):
    """Mescla varias varreduras independentes (captura dividida em partes).

    Cada varredura e alinhada sobre o acumulado. Retorna (pts_mesclados, relatorio).
    """
    acc = scans[0]
    rel = [{"scan": 0, "status": "referencia"}]
    for i, s in enumerate(scans[1:], start=1):
        T, fit, rmse, extra = register(s, acc, spacing, seed, min_fitness=min_fitness)
        ok = fit >= min_fitness and rmse <= 1.5 * spacing
        rel.append({"scan": i, "fitness": round(fit, 3), "rmse": round(rmse, 4),
                    "costura": extra.get("costura"), "ambiguo": extra.get("ambiguo"),
                    "status": "mesclado" if ok else "rejeitado", "T": np.round(T, 6).tolist()})
        if ok:
            acc = np.vstack([acc, apply_T(T, s)])
    return acc, rel
