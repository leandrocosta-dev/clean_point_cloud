"""Orquestracao: diagnostico -> limpeza -> metricas -> ajuste -> saida."""
import hashlib
import json
import os
import time
from dataclasses import asdict

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from .config import Config
from .geom import mean_spacing, make_working_cloud, apply_T
from .clean import run_clean, BASE, NOISE, CLUSTER, DUP, AMBIG
from . import metrics as M
from .optimize import tune
from .render import render_views
from . import inspect_llm


def _hash(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _propagate(orig_pts, work_pts, work_labels):
    """Leva os rotulos da nuvem reduzida para a nuvem original (vizinho mais proximo)."""
    _, nn = cKDTree(work_pts).query(orig_pts, k=1, workers=-1)
    return work_labels[nn]


def _save_outputs(pcd, labels, transforms, out_dir, stem):
    pts = np.asarray(pcd.points)
    keep = labels >= 0
    out = o3d.geometry.PointCloud()
    p = pts[keep].copy()
    lk = labels[keep]
    nrm = np.asarray(pcd.normals)[keep].copy() if pcd.has_normals() else None
    for k, T in transforms.items():
        m = lk == k
        p[m] = apply_T(T, p[m])
        if nrm is not None:
            nrm[m] = nrm[m] @ np.asarray(T)[:3, :3].T
    out.points = o3d.utility.Vector3dVector(p)
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[keep])
    if nrm is not None:
        out.normals = o3d.utility.Vector3dVector(nrm)
    path_ok = os.path.join(out_dir, f"{stem}_limpo.ply")
    o3d.io.write_point_cloud(path_ok, out)

    # Removidos coloridos por motivo, para conferencia
    rem = ~keep
    cores = {BASE: [0.35, 0.63, 0.88], NOISE: [0.88, 0.19, 0.19],
             CLUSTER: [0.94, 0.55, 0.0], DUP: [0.6, 0.3, 0.8], AMBIG: [0.9, 0.8, 0.1]}
    rp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[rem]))
    rp.colors = o3d.utility.Vector3dVector(np.array([cores.get(l, [0, 0, 0]) for l in labels[rem]]))
    path_rem = os.path.join(out_dir, f"{stem}_removidos.ply")
    o3d.io.write_point_cloud(path_rem, rp)
    return path_ok, path_rem


def process(input_path, out_dir, cfg: Config = None, use_llm=False, log=print):
    cfg = cfg or Config()
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    t0 = time.time()
    rel = {"arquivo": input_path, "hash": _hash(input_path), "inicio": time.strftime("%Y-%m-%d %H:%M:%S")}

    pcd = o3d.io.read_point_cloud(input_path)
    orig = np.asarray(pcd.points)
    if len(orig) == 0:
        raise ValueError("nuvem vazia ou formato nao suportado")
    work, reduced = make_working_cloud(orig, cfg.work_max_points, cfg.seed)
    spacing = mean_spacing(work, seed=cfg.seed)
    raw_counts = M.neighbor_counts(work, 2.5 * spacing)
    rel["diagnostico"] = {"n_original": int(len(orig)), "n_trabalho": int(len(work)),
                          "reduzida": reduced, "espacamento_mm": round(spacing, 4)}
    log(f"[1] {len(orig):,} pontos | trabalho {len(work):,} | espacamento {spacing:.4f} mm")

    render_views(work, np.zeros(len(work), int), {},
                 os.path.join(out_dir, f"{stem}_vistas_antes.png"), title=f"{stem}: nuvem crua")

    # 2. Tentativa com parametros por regra (sem LLM, sem busca)
    P = cfg.params
    labels, info, T = run_clean(work, spacing, P, cfg, align=False)
    m = M.compute(work, labels, info, T, spacing, raw_counts, cfg.gates, cfg.seed)
    rel["tentativa_padrao"] = {"passou": m.get("passou"), "reprovado_em": m.get("reprovado_em")}
    log(f"[2] parametros padrao: passou={m.get('passou')} reprovado_em={m.get('reprovado_em')}")

    # 3. Se reprovou, busca de parametros
    if not m.get("passou"):
        log(f"[3] buscando parametros (ate {cfg.optimize_trials} tentativas)...")
        P, best_cost, n_trials = tune(work, spacing, cfg, raw_counts, log=log)
        rel["otimizacao"] = {"tentativas": n_trials, "melhor_custo": round(best_cost, 4)}

    # 4. Rodada final com realinhamento de trechos deslocados
    log("[4] rodada final com realinhamento...")
    labels, info, T = run_clean(work, spacing, P, cfg, align=cfg.align_fragments)
    m = M.compute(work, labels, info, T, spacing, raw_counts, cfg.gates, cfg.seed)
    for f in info["fragmentos"]:
        log(f"    trecho {f['n_pontos']:,} pts: {f['status']} (fitness {f.get('fitness')}, "
            f"rmse {f.get('rmse')}, costura {f.get('costura')}, candidatos {f.get('candidatos')})")
        for alt in f.get("alternativas", []):
            log(f"      candidato {alt['origem']}: costura={alt['costura']} fitness={alt['fitness']} "
                f"estavel={alt['estavel']}")

    img = os.path.join(out_dir, f"{stem}_vistas.png")
    render_views(work, labels, T, img, title=f"{stem}: limpeza")

    # 5. Inspecao opcional por LLM (1 chamada; no maximo 1 rodada extra)
    if use_llm:
        img_final = os.path.join(out_dir, f"{stem}_vistas_final.png")
        render_views(work, labels, T, img_final, title=stem, only_final=True)
        verdict = inspect_llm.inspect(img_final, m)
        rel["llm"] = verdict
        if verdict is None:
            log("[5] LLM indisponivel (pacote anthropic ou ANTHROPIC_API_KEY ausente), seguindo sem ela")
        elif not verdict.get("aprovado"):
            P2, changed = inspect_llm.apply_feedback(P, verdict)
            log(f"[5] LLM reprovou: {verdict.get('problemas')}")
            if changed:
                l2, i2, T2 = run_clean(work, spacing, P2, cfg, align=cfg.align_fragments)
                m2 = M.compute(work, l2, i2, T2, spacing, raw_counts, cfg.gates, cfg.seed)
                if m2.get("passou"):
                    labels, info, T, m, P = l2, i2, T2, m2, P2
                    render_views(work, labels, T, img, title=f"{stem}: limpeza (ajuste LLM)")
                    log("[5] ajuste aplicado")
        else:
            log("[5] LLM aprovou")

    # 6. Status final
    status = "aprovado"
    if not m.get("passou"):
        status = "revisar"
    if m.get("fragmentos_rejeitados", 0) > 0:
        status = "revisar"  # existe trecho que parece peca mas nao alinhou
    if m.get("suspeita_superficie_dupla_frac", 0) > 0.05:
        status = "revisar"
    alertas = []
    if not info.get("base") and m.get("maior_plano_frac", 0) > 0.35:
        alertas.append("base nao detectada, mas um plano domina a nuvem final: conferir se a base ficou")
        status = "revisar"
    for f in info.get("fragmentos", []):
        if f["status"] == "ambiguo":
            alertas.append("trecho deslocado com encaixe ambiguo, NAO foi mesclado: "
                           + "; ".join(f.get("motivo_ambiguidade", [])))
            status = "revisar"
    if m.get("fragmentos_rejeitados", 0) > 0:
        alertas.append("existe trecho grande que nao alinhou: possivel perda de rastreamento sem sobreposicao")
    if m.get("suspeita_superficie_dupla_frac", 0) > 0.05:
        alertas.append("suspeita de superficie dupla: trecho deslocado pode estar encostado na peca")
    for a in alertas:
        log(f"    ALERTA: {a}")
    if use_llm and rel.get("llm") and not rel["llm"].get("aprovado") and status == "aprovado":
        status = "revisar"

    # 7. Leva rotulos para a nuvem original e salva
    full_labels = _propagate(orig, work, labels) if reduced else labels
    path_ok, path_rem = _save_outputs(pcd, full_labels, T, out_dir, stem)
    sugestoes = []
    for i, sg in enumerate(info.get("sugestoes", []), start=1):
        sp = os.path.join(out_dir, f"{stem}_trecho{i}_sugestao.ply")
        o3d.io.write_point_cloud(sp, o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(apply_T(sg["T"], work[sg["idx"]]))))
        sugestoes.append(sp)
    info.pop("sugestoes", None)

    rel.update({
        "status": status,
        "alertas": alertas,
        "parametros_finais": asdict(P),
        "metricas": {k: v for k, v in m.items() if not k.startswith("_")},
        "criterios": m.get("_checks"),
        "base": info.get("base"),
        "fragmentos": info.get("fragmentos"),
        "saidas": {"limpo": path_ok, "removidos": path_rem, "vistas": img, "sugestoes": sugestoes},
        "tempo_s": round(time.time() - t0, 1),
    })
    with open(os.path.join(out_dir, f"{stem}_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)
    # Historico acumulado: base para calibrar criterios e treinar modelo no futuro
    with open(os.path.join(out_dir, "historico_execucoes.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({k: rel[k] for k in ("arquivo", "hash", "inicio", "status",
                                                 "parametros_finais", "metricas", "tempo_s")},
                           ensure_ascii=False) + "\n")
    log(f"[ok] status={status} | {m.get('n_final', 0):,} pontos (trabalho) | {rel['tempo_s']} s")
    return rel
