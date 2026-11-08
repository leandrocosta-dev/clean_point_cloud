"""Cobertura de cada scan no referencial comum e ordem de construcao (cobertura gulosa).

    python pipeline/cobertura.py <pasta_segmentos> <segmentos.json> --saida <pasta> [--por-scan por_scan.json]

Cada scan (segmentos ja alinhados) vira o conjunto de voxels de 2 mm que ele cobre. A ordem de construcao
escolhe, a cada passo, o scan que acrescenta mais superficie NOVA (empate: menor vies de alinhamento);
mostra a cobertura acumulada em relacao a uniao de todas as capturas e o conjunto minimo para 99% / 99,5%.
Gera cobertura.json e imagens passo a passo (azul = ja coberto, vermelho = ainda falta coberto so pelas capturas
restantes).
"""

import argparse
import json
import os

import gpu
import numpy as np
from alinhar_varios import baixa_prioridade
from registro_gpu import transformar

VOX = 2.0


def chaves(P, origem):
    """Chave int64 do voxel de VOX mm de cada ponto (grade ancorada em `origem`); GPU."""
    return gpu.chaves_voxel(P, VOX, origem)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("segmentos")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--por-scan")
    ap.add_argument(
        "--min-voxels", type=int, default=50, help="voxels vistos por ao menos N pontos (descarta ruido isolado)"
    )
    a = ap.parse_args()
    baixa_prioridade()
    os.makedirs(a.saida, exist_ok=True)
    ind = json.load(open(os.path.join(a.pasta, "segmentos_indice.json")))
    pj = json.load(open(a.segmentos))["segmentos"]
    vies = {}
    if a.por_scan and os.path.exists(a.por_scan):
        vies = {g: (v or {}).get("mediana_um", 999) for g, v in json.load(open(a.por_scan)).items()}
    scans = sorted(g for g in ind if g in pj)
    # nuvens globais por scan (centroides de 1 mm) -> centroides de 2 mm
    xp = gpu.xp()
    P = {}
    for g in scans:
        partes = []
        for s in ind[g]:
            X = np.load(s["arquivo"]).astype(np.float64)
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            partes.append(transformar(gpu.para_gpu(X, xp.float64), T))
        P[g] = xp.concatenate(partes)
    origem = xp.min(xp.stack([p.min(0) for p in P.values()]), axis=0) - 10
    V = {}  # voxels (chaves ordenadas) vistos por >= 3 centroides, na GPU
    for g in scans:
        k, _, c = gpu.agrupar(chaves(P[g], origem))
        V[g] = k[c >= 3]
    uniao = gpu.agrupar(xp.concatenate(list(V.values())))[0]
    print(
        f"{len(scans)} scans | uniao de todas as capturas: {len(uniao):,} voxels de {VOX:g} mm (~{len(uniao) * VOX * VOX / 1e6:.2f} m2 de superficie)",
        flush=True,
    )
    for g in scans:
        print(
            f"   {g}: {len(V[g]):7,} voxels ({len(V[g]) / len(uniao) * 100:4.1f}% da uniao) | vies {vies.get(g, '?')} um",
            flush=True,
        )
    cobertos, ordem, passos = xp.empty(0, xp.int64), [], []
    restantes = list(scans)
    while restantes:
        gan = {g: int((~xp.isin(V[g], cobertos)).sum()) for g in restantes}
        melhor = max(restantes, key=lambda g: (gan[g], -vies.get(g, 999)))
        if gan[melhor] == 0:
            break
        cobertos = gpu.agrupar(xp.concatenate([cobertos, V[melhor]]))[0]
        restantes.remove(melhor)
        ordem.append(melhor)
        passos.append(
            dict(
                scan=melhor,
                ganho_voxels=gan[melhor],
                ganho_pct=gan[melhor] / len(uniao) * 100,
                acumulado_pct=len(cobertos) / len(uniao) * 100,
            )
        )
        print(
            f"  passo {len(ordem):2d}: +{melhor} ganha {gan[melhor]:7,} ({gan[melhor] / len(uniao) * 100:4.1f}%) -> cobertura {len(cobertos) / len(uniao) * 100:5.1f}%",
            flush=True,
        )
    minimo = {}
    for alvo in (95.0, 99.0, 99.5):
        n = next((i + 1 for i, p in enumerate(passos) if p["acumulado_pct"] >= alvo), len(passos))
        minimo[str(alvo)] = ordem[:n]
        print(f"  conjunto minimo para {alvo}% da uniao: {n} scans: {[g[-6:] for g in ordem[:n]]}")
    json.dump(
        dict(
            uniao_voxels=len(uniao),
            voxel_mm=VOX,
            ordem=ordem,
            passos=passos,
            minimo=minimo,
            por_scan={
                g: dict(voxels=len(V[g]), pct_uniao=len(V[g]) / len(uniao) * 100, vies_um=vies.get(g)) for g in scans
            },
        ),
        open(os.path.join(a.saida, "cobertura.json"), "w"),
        indent=1,
    )
    # imagens passo a passo (vista de cima e lateral no eixo da peca)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ku = uniao  # ja ordenada
    C = gpu.decodificar_voxel(ku, VOX, origem)
    c0 = C.mean(0)
    Y = C - c0
    w, Vv = np.linalg.eigh(gpu.para_cpu(Y.T @ Y / max(len(C) - 1, 1)))
    Q = gpu.para_cpu(Y @ xp.asarray(Vv[:, ::-1]))
    n_img = min(len(ordem), 6)
    escolha = np.unique(np.linspace(0, len(ordem) - 1, n_img).round().astype(int))
    fig, ax = plt.subplots(2, len(escolha), figsize=(4.2 * len(escolha), 8), dpi=80, squeeze=False)
    acum = xp.empty(0, xp.int64)
    for p, g in enumerate(ordem):
        acum = gpu.agrupar(xp.concatenate([acum, V[g]]))[0]
        if p not in escolha:
            continue
        c = list(escolha).index(p)
        cob = gpu.para_cpu(xp.isin(ku, acum))
        for r, (u, v) in enumerate(((0, 1), (0, 2))):
            ax[r, c].scatter(Q[~cob, u], Q[~cob, v], s=1.2, c="#d62728", linewidths=0)
            ax[r, c].scatter(Q[cob, u], Q[cob, v], s=1.2, c="#1f77b4", linewidths=0)
            ax[r, c].set_aspect("equal")
            ax[r, c].axis("off")
        ax[0, c].set_title(f"passo {p + 1}: +{g[-6:]}\ncobertura {len(acum) / len(uniao) * 100:.1f}%", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(a.saida, "construcao_passo_a_passo.png"))
    print("imagem:", os.path.join(a.saida, "construcao_passo_a_passo.png"))


if __name__ == "__main__":
    main()
