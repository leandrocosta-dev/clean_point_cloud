"""Cobertura de cada scan no referencial comum e ordem de construcao (cobertura gulosa).

    python pipeline/cobertura.py <pasta_segmentos> <segmentos.json> --saida <pasta> [--por-scan por_scan.json]

Cada scan (segmentos ja alinhados) vira o conjunto de voxels de 2 mm que ele cobre. A ordem de construcao
escolhe, a cada passo, o scan que acrescenta mais superficie NOVA (empate: menor vies de alinhamento);
mostra a cobertura acumulada em relacao a uniao de todas as capturas e o conjunto minimo para 99% / 99,5%.
Gera cobertura.json e imagens passo a passo (azul = ja coberto, vermelho = ainda falta coberto so pelas capturas
restantes).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade

VOX = 2.0


def chaves(P, origem):
    k = np.floor((P - origem) / VOX).astype(np.int64)
    return (k[:, 0] * 200003 + k[:, 1]) * 200017 + k[:, 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta"); ap.add_argument("segmentos")
    ap.add_argument("--saida", required=True); ap.add_argument("--por-scan")
    ap.add_argument("--min-voxels", type=int, default=50, help="voxels vistos por ao menos N pontos (descarta ruido isolado)")
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
    P = {}
    for g in scans:
        partes = []
        for s in ind[g]:
            X = np.load(s["arquivo"]).astype(np.float64)
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            partes.append(X @ T[:3, :3].T + T[:3, 3])
        P[g] = np.vstack(partes)
    todos = np.vstack(list(P.values()))
    origem = todos.min(0) - 10
    V = {}
    for g in scans:
        k, c = np.unique(chaves(P[g], origem), return_counts=True)
        V[g] = set(k[c >= 3].tolist())
    uniao = set().union(*V.values())
    print(f"{len(scans)} scans | uniao de todas as capturas: {len(uniao):,} voxels de {VOX:g} mm (~{len(uniao)*VOX*VOX/1e6:.2f} m2 de superficie)", flush=True)
    for g in scans:
        print(f"   {g}: {len(V[g]):7,} voxels ({len(V[g])/len(uniao)*100:4.1f}% da uniao) | vies {vies.get(g,'?')} um", flush=True)
    cobertos, ordem, passos = set(), [], []
    restantes = list(scans)
    while restantes:
        gan = {g: len(V[g] - cobertos) for g in restantes}
        melhor = max(restantes, key=lambda g: (gan[g], -vies.get(g, 999)))
        if gan[melhor] == 0:
            break
        cobertos |= V[melhor]
        restantes.remove(melhor)
        ordem.append(melhor)
        passos.append(dict(scan=melhor, ganho_voxels=gan[melhor], ganho_pct=gan[melhor] / len(uniao) * 100,
                           acumulado_pct=len(cobertos) / len(uniao) * 100))
        print(f"  passo {len(ordem):2d}: +{melhor} ganha {gan[melhor]:7,} ({gan[melhor]/len(uniao)*100:4.1f}%) -> cobertura {len(cobertos)/len(uniao)*100:5.1f}%", flush=True)
    minimo = {}
    for alvo in (95.0, 99.0, 99.5):
        n = next((i + 1 for i, p in enumerate(passos) if p["acumulado_pct"] >= alvo), len(passos))
        minimo[str(alvo)] = ordem[:n]
        print(f"  conjunto minimo para {alvo}% da uniao: {n} scans: {[g[-6:] for g in ordem[:n]]}")
    json.dump(dict(uniao_voxels=len(uniao), voxel_mm=VOX, ordem=ordem, passos=passos, minimo=minimo,
                   por_scan={g: dict(voxels=len(V[g]), pct_uniao=len(V[g]) / len(uniao) * 100, vies_um=vies.get(g)) for g in scans}),
              open(os.path.join(a.saida, "cobertura.json"), "w"), indent=1)
    # imagens passo a passo (vista de cima e lateral no eixo da peca)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ku = np.array(sorted(uniao)); 
    def centro(k):
        z = k % 200017; xy = k // 200017; y = xy % 200003; x = xy // 200003
        return (np.stack([x, y, z], 1) + 0.5) * VOX + origem
    C = centro(ku)
    c0 = C.mean(0); w, Vv = np.linalg.eigh(np.cov((C - c0).T)); Q = (C - c0) @ Vv[:, ::-1]
    idx = {k: i for i, k in enumerate(ku.tolist())}
    n_img = min(len(ordem), 6)
    escolha = np.unique(np.linspace(0, len(ordem) - 1, n_img).round().astype(int))
    fig, ax = plt.subplots(2, len(escolha), figsize=(4.2 * len(escolha), 8), dpi=80)
    ax = np.atleast_2d(ax)
    acum = set()
    for p, g in enumerate(ordem):
        acum |= V[g]
        if p not in escolha:
            continue
        c = list(escolha).index(p)
        cob = np.array([k in acum for k in ku.tolist()])
        for r, (u, v) in enumerate(((0, 1), (0, 2))):
            ax[r, c].scatter(Q[~cob, u], Q[~cob, v], s=1.2, c="#d62728", linewidths=0)
            ax[r, c].scatter(Q[cob, u], Q[cob, v], s=1.2, c="#1f77b4", linewidths=0)
            ax[r, c].set_aspect("equal"); ax[r, c].axis("off")
        ax[0, c].set_title(f"passo {p+1}: +{g[-6:]}\ncobertura {len(acum)/len(uniao)*100:.1f}%", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.saida, "construcao_passo_a_passo.png"))
    print("imagem:", os.path.join(a.saida, "construcao_passo_a_passo.png"))


if __name__ == "__main__":
    main()
