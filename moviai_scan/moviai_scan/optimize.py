"""Ajuste automatico de parametros por busca (Optuna), no lugar da LLM.

Reproduzivel (seed fixa), com limite de tentativas e criterio objetivo.
Roda sem realinhamento (mais rapido); trechos candidatos contam como peca.
"""
from dataclasses import replace, asdict
import optuna

from .clean import run_clean
from . import metrics as M


def tune(pts, spacing, cfg, raw_counts, log=print):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    base_p = cfg.params

    def objective(trial):
        P = replace(base_p,
                    sor_nb=trial.suggest_int("sor_nb", 10, 50),
                    sor_std=trial.suggest_float("sor_std", 0.8, 4.0),
                    dbscan_eps_mult=trial.suggest_float("dbscan_eps_mult", 2.0, 7.0),
                    base_dist_mult=trial.suggest_float("base_dist_mult", 1.5, 6.0),
                    base_margin_mult=trial.suggest_float("base_margin_mult", 0.5, 4.0))
        labels, info, T = run_clean(pts, spacing, P, cfg, align=False)
        m = M.compute(pts, labels, info, T, spacing, raw_counts, cfg.gates, cfg.seed)
        trial.set_user_attr("passou", bool(m.get("passou")))
        c = M.cost(m, cfg.gates)
        log(f"  tentativa {trial.number:02d}: custo={c:.3f} passou={m.get('passou')} "
            f"reprovado_em={m.get('reprovado_em')}")
        return c

    def stop_when_good(study, trial):
        # Para cedo: ja tem resultado aprovado e pelo menos 10 tentativas
        n_ok = sum(t.user_attrs.get("passou", False) for t in study.trials)
        if n_ok >= 1 and len(study.trials) >= 10:
            study.stop()

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    d = asdict(base_p)
    study.enqueue_trial({k: d[k] for k in
                         ("sor_nb", "sor_std", "dbscan_eps_mult", "base_dist_mult", "base_margin_mult")})
    study.optimize(objective, n_trials=cfg.optimize_trials, callbacks=[stop_when_good])
    best = replace(base_p, **study.best_params)
    return best, study.best_value, len(study.trials)
