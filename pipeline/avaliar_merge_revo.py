"""Avalia um Merge Alignment feito no Revo Scan (sem abrir o Revo): alinhamento e sujeira por scan.

    python pipeline/avaliar_merge_revo.py <pasta_ou_Merge_XX.revox> [--saida PASTA] [--tol 0.4]

Le `point_clouds[0].merge_depends` (caminho do .revox de cada scan + pose 4x4 scan -> merge, linha a linha),
aplica a pose na fusao de cada scan (pointclouds/*.ply da pasta do scan) e mede:
  - encaixe de cada scan contra os outros (mediana/p90 da distancia, so pontos com vizinho < 1 mm; % a > 1 mm);
  - confirmacao: em quantos OUTROS scans cada ponto tem vizinho a --tol mm; "lixo" = sem nenhuma confirmacao e a
    > 0,3 mm de superficie confirmada por >= 2 scans (fiapos, migalhas, apoio, copias deslocadas).
Saida (--saida): fused_<guid8>.npy (alinhada), nap_<guid8>.npy (n. de confirmacoes), poses_final.json
(guid -> pose), avaliacao.json e lixo.png. Esses arquivos alimentam a limpeza seguinte dos quadros.
"""
import argparse
import glob
import json
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("merge")
    ap.add_argument("--saida")
    ap.add_argument("--tol", type=float, default=0.4)
    a = ap.parse_args()
    revox = a.merge if a.merge.endswith(".revox") else glob.glob(os.path.join(a.merge, "*.revox"))[0]
    saida = a.saida or os.path.join(os.path.dirname(os.path.abspath(revox)), "_avaliacao")
    os.makedirs(saida, exist_ok=True)
    F, poses, nomes = {}, {}, {}

    def expandir(revox_path, T_pai):
        """Um item do merge pode ser outro Merge_XX.revox: abre recursivamente compondo as poses (pai @ filho)."""
        rr = json.load(open(revox_path, encoding="utf-8"))
        for m in rr["point_clouds"][0]["merge_depends"]:
            T = T_pai @ np.array(m["pose"], float).reshape(4, 4)
            sub = json.load(open(m["path"], encoding="utf-8"))
            if sub.get("point_clouds") and sub["point_clouds"][0].get("merge_depends"):
                print("merge dentro do merge:", os.path.basename(m["path"]))
                expandir(m["path"], T)
                continue
            d = os.path.dirname(m["path"])
            prop = os.path.join(d, "property.rvproj")
            g = json.load(open(prop, encoding="utf-8"))["guid"] if os.path.exists(prop) else os.path.basename(m["path"])[:-6]
            plys = sorted(glob.glob(os.path.join(d, "pointclouds", "*.ply")), key=os.path.getmtime)
            if not plys:
                print("sem fusao:", m["path"]); continue
            P = np.asarray(o3d.io.read_point_cloud(plys[-1]).points)
            F[g] = P @ T[:3, :3].T + T[:3, 3]; poses[g] = T.tolist(); nomes[g] = os.path.basename(m["path"])[:-6]
    expandir(revox, np.eye(4))
    ks = sorted(F)
    arv = {k: cKDTree(F[k]) for k in ks}
    rel = {}
    print(f"{'scan':10} {'nome':24} {'encaixe med':>11} {'p90':>6} {'>1mm':>6} | {'sem conf.':>9} {'>=2':>6} {'lixo':>6}")
    for k in ks:
        O = cKDTree(np.vstack([F[j][::2] for j in ks if j != k]))
        d = O.query(F[k][::3], workers=4)[0]
        n = np.zeros(len(F[k]), int)
        for j in ks:
            if j != k:
                n += arv[j].query(F[k], distance_upper_bound=a.tol, workers=4)[0] < a.tol
        boa, sem = F[k][n >= 2], F[k][n == 0]
        lixo = int((cKDTree(boa).query(sem, workers=4)[0] > 0.3).sum()) if len(sem) and len(boa) else len(sem)
        np.save(os.path.join(saida, f"fused_{k[:8]}.npy"), F[k]); np.save(os.path.join(saida, f"nap_{k[:8]}.npy"), n)
        dd = d[d < 1] if (d < 1).any() else d
        rel[k] = dict(nome=nomes[k], encaixe_med=float(np.median(dd)), encaixe_p90=float(np.percentile(dd, 90)),
                      acima_1mm=float(np.mean(d > 1)), sem_confirmacao=float(np.mean(n == 0)),
                      confirmado_2=float(np.mean(n >= 2)), lixo=lixo / len(n))
        x = rel[k]
        print(f"{k[:8]:10} {nomes[k][:24]:24} {x['encaixe_med']:11.3f} {x['encaixe_p90']:6.3f} {x['acima_1mm']*100:5.1f}% |"
              f" {x['sem_confirmacao']*100:8.1f}% {x['confirmado_2']*100:5.1f}% {x['lixo']*100:5.2f}%")
    json.dump(poses, open(os.path.join(saida, "poses_final.json"), "w"), indent=1)
    json.dump(rel, open(os.path.join(saida, "avaliacao.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    allP = np.vstack([F[k] for k in ks]); c = allP.mean(0)
    _, _, Vt = np.linalg.svd(allP[::20] - c, full_matrices=False)
    fig, ax = plt.subplots(2, len(ks), figsize=(4 * len(ks), 8), squeeze=False)
    for i, k in enumerate(ks):
        n = np.load(os.path.join(saida, f"nap_{k[:8]}.npy")); R = (F[k] - c) @ Vt.T; m = n == 0
        for j, (u, v) in enumerate(((0, 1), (0, 2))):
            ax[j, i].scatter(R[~m, u][::3], R[~m, v][::3], s=0.02, c="#bbb"); ax[j, i].scatter(R[m, u], R[m, v], s=0.3, c="r")
            ax[j, i].set_aspect("equal")
        ax[0, i].set_title(f"{k[:8]} sem conf. {m.mean()*100:.1f}%")
    plt.tight_layout(); plt.savefig(os.path.join(saida, "lixo.png"), dpi=50)
    print("saida:", saida)


if __name__ == "__main__":
    main()
