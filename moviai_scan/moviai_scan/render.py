"""Render em 3 vistas (topo, frente, lateral) para conferencia e para a LLM.

Linha de cima: nuvem final (peca + trechos realinhados).
Linha de baixo: o que foi removido, por motivo.
Usa matplotlib, roda sem placa de video e sem janela.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .clean import BASE, NOISE, CLUSTER, AMBIG, final_points

CORES = {
    "nuvem crua": "#555555",
    "peca": "#8a8a8a",
    "realinhado": "#1a9e5a",
    "base": "#5aa0e0",
    "ruido": "#e03030",
    "ilha/rejeitado": "#f08c00",
    "trecho ambiguo": "#c9a400",
}
DENSOS = {"nuvem crua", "peca", "realinhado", "base", "trecho ambiguo"}
VISTAS = [("Topo (X-Y)", 0, 1), ("Frente (X-Z)", 0, 2), ("Lateral (Y-Z)", 1, 2)]


def _row(axes, layers, rng, lims, max_pts, row_title=None):
    total = sum(len(p) for n, p in layers if n in DENSOS)
    frac = min(1.0, max_pts / max(total, 1))
    for ax, (name, a, b) in zip(axes, VISTAS):
        for lname, p in layers:
            if len(p) == 0:
                continue
            dense = lname in DENSOS
            k = max(1, int(len(p) * frac)) if dense else len(p)
            sel = p[rng.choice(len(p), size=min(k, len(p)), replace=False)]
            ax.scatter(sel[:, a], sel[:, b], s=0.5 if dense else 3, c=CORES[lname],
                       label=f"{lname} ({len(p):,})", linewidths=0, rasterized=True)
        ax.set_xlim(lims[a])
        ax.set_ylim(lims[b])
        ax.set_aspect("equal")
        ax.set_title(name if row_title is None else f"{row_title}: {name}", fontsize=10)
        ax.set_xlabel("XYZ"[a] + " (mm)", fontsize=8)
        ax.set_ylabel("XYZ"[b] + " (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[-1].legend(h, l, loc="upper right", markerscale=10, fontsize=8, framealpha=0.9)


def render_views(pts, labels, transforms, path, title="", max_pts=80_000, seed=42, only_final=False):
    rng = np.random.default_rng(seed)
    fin, keep = final_points(pts, labels, transforms)
    lk = labels[keep]
    crua = title.endswith("crua")
    final_layers = [("nuvem crua" if crua else "peca", fin[lk == 0]), ("realinhado", fin[lk > 0])]
    rem_layers = [("base", pts[labels == BASE]), ("trecho ambiguo", pts[labels == AMBIG]),
                  ("ilha/rejeitado", pts[labels == CLUSTER]), ("ruido", pts[labels == NOISE])]

    allp = fin if (only_final and len(fin)) else (np.vstack([fin, pts]) if len(fin) else pts)
    lo, hi = np.percentile(allp, 0.2, axis=0) - 5, np.percentile(allp, 99.8, axis=0) + 5
    lims = list(zip(lo, hi))

    two_rows = not (only_final or crua)
    nrows = 2 if two_rows else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(18, 6.0 * nrows), dpi=100, squeeze=False)
    _row(axes[0], final_layers, rng, lims, max_pts, "Resultado" if two_rows else None)
    if two_rows:
        _row(axes[1], rem_layers, rng, lims, max_pts, "Removido")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
