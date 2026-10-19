"""Metricas objetivas de qualidade da limpeza."""
import numpy as np
from scipy.spatial import cKDTree

from .clean import BASE, NOISE, CLUSTER, final_points
from .geom import to_pcd
import open3d as o3d


def neighbor_counts(pts, radius):
    return cKDTree(pts).query_ball_point(pts, radius, return_length=True, workers=-1)


def compute(pts, labels, info, transforms, spacing, raw_counts, gates, seed=42):
    n = len(pts)
    n_base = int((labels == BASE).sum())
    noise = (labels == NOISE) | (labels == CLUSTER)
    nonbase = max(1, n - n_base)
    fin, keep = final_points(pts, labels, transforms)
    n_keep = len(fin)
    m = {"n_total": n, "n_base": n_base, "n_ruido": int(noise.sum()), "n_final": n_keep}
    if n_keep < 50:
        m.update(passou=False, motivo="nuvem final vazia")
        return m

    # Fracao da peca removida como ruido
    m["noise_removed_frac"] = float(noise.sum() / nonbase)

    # Pontos isolados que sobraram
    r = 2.5 * spacing
    cnt = neighbor_counts(fin, r)
    m["isolated_frac"] = float(np.mean(cnt < 4))

    # Superficie real removida por engano: ponto removido que estava em regiao densa
    dense_thr = 0.6 * np.median(raw_counts[keep])
    m["removed_dense_frac"] = float((noise & (raw_counts >= dense_thr)).sum() / nonbase)

    # Restos de base: pontos perto do plano com normal paralela a ele
    m["base_residual_frac"] = 0.0
    if info.get("base"):
        nb = np.array(info["base"]["n"])
        sd = fin @ nb + info["base"]["d"]
        near = (sd >= 0) & (sd < info["base"]["band"] + 6 * spacing)
        if near.sum() > 20:
            pcd = to_pcd(fin[near])
            pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * spacing, max_nn=30))
            nr = np.abs(np.asarray(pcd.normals) @ nb)
            m["base_residual_frac"] = float((nr > 0.9).sum() / n_keep)

    # Alerta (nao reprova): suspeita de superficie dupla. Detecta trecho
    # deslocado que ficou ENCOSTADO na peca (sem vao), caso em que o
    # DBSCAN nao consegue separa-lo. Quinas vivas tambem elevam esse valor.
    rng = np.random.default_rng(seed)
    sample = rng.choice(n_keep, size=min(3000, n_keep), replace=False)
    tree = cKDTree(fin)
    nbrs = tree.query_ball_point(fin[sample], 4 * spacing, workers=-1)
    thick = []
    for nb_idx in nbrs:
        if len(nb_idx) < 10:
            continue
        q = fin[nb_idx] - fin[nb_idx].mean(0)
        ev = np.linalg.eigvalsh(q.T @ q / len(q))
        thick.append(np.sqrt(max(ev[0], 0)))
    thick = np.array(thick)
    m["suspeita_superficie_dupla_frac"] = float(np.mean(thick > 1.5 * spacing)) if len(thick) else 0.0

    # Alerta: maior plano da nuvem final. Se a base nao foi detectada e
    # um plano domina a nuvem, provavelmente a base ficou (ou a peca e uma chapa).
    _, pin = to_pcd(fin).segment_plane(2.5 * spacing, 3, 1000)
    m["maior_plano_frac"] = float(len(pin) / n_keep)

    frs = info.get("fragmentos", [])
    m["fragmentos_realinhados"] = sum(f["status"] == "realinhado" for f in frs)
    m["fragmentos_rejeitados"] = sum(f["status"] == "rejeitado" for f in frs)
    m["fragmentos_ambiguos"] = sum(f["status"] == "ambiguo" for f in frs)

    g = gates
    checks = {
        "noise_removed_frac": (m["noise_removed_frac"], g.max_noise_removed_frac),
        "isolated_frac": (m["isolated_frac"], g.max_isolated_frac),
        "base_residual_frac": (m["base_residual_frac"], g.max_base_residual_frac),
        "removed_dense_frac": (m["removed_dense_frac"], g.max_removed_dense_frac),
    }
    m["reprovado_em"] = [k for k, (v, lim) in checks.items() if v > lim]
    m["passou"] = not m["reprovado_em"]
    m["_checks"] = {k: [round(v, 5), lim] for k, (v, lim) in checks.items()}
    return m


def cost(m, gates):
    """Custo para o otimizador: qualquer resultado aprovado < qualquer reprovado."""
    if "_checks" not in m:
        return 1e6
    pen = sum(max(0.0, v / lim - 1.0) for v, lim in m["_checks"].values())
    soft = 0.5 * m["removed_dense_frac"] / gates.max_removed_dense_frac \
        + 0.5 * m["isolated_frac"] / gates.max_isolated_frac
    return soft / 2 if pen == 0 else 1.0 + 10 * pen
