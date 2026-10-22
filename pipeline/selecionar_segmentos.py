"""Escolhe o conjunto MINIMO de segmentos (15 quadros) que cobre a peca, com redundancia dupla onde existir.

    python pipeline/selecionar_segmentos.py <pasta_segmentos> <segmentos.json> --saida <selecao.json> [--redundancia 2]

Cobertura em voxels de 2 mm no referencial comum. Cada voxel da uniao de todas as capturas pede
min(redundancia, quantos segmentos o veem). Guloso: a cada passo entra o segmento que mais reduz o que falta
(empate: scan de menor vies). Para quando o que falta cai abaixo de `--folga` da uniao. A redundancia 2 e para
promediar o ruido de medicao; cobertura simples e o minimo para nao cortar a peca.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade

VOX = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta"); ap.add_argument("segmentos"); ap.add_argument("--saida", required=True)
    ap.add_argument("--redundancia", type=int, default=2)
    ap.add_argument("--folga", type=float, default=0.003, help="para quando faltar menos que isto da uniao")
    ap.add_argument("--por-scan")
    ap.add_argument("--apoio-min", type=int, default=12, help="centroides de 1 mm minimos por voxel de 2 mm para contar como superficie")
    a = ap.parse_args()
    baixa_prioridade()
    ind = json.load(open(os.path.join(a.pasta, "segmentos_indice.json")))
    pj = json.load(open(a.segmentos))["segmentos"]
    vies = {g: (v or {}).get("mediana_um", 150) for g, v in json.load(open(a.por_scan)).items()} if a.por_scan else {}
    nos, G = [], []
    for g in sorted(ind):
        if g not in pj:
            continue
        for s in ind[g]:
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            X = np.load(s["arquivo"]).astype(np.float64)
            G.append(X @ T[:3, :3].T + T[:3, 3])
            nos.append(dict(scan=g, seg=s["seg"], k0=s["k0"], k1=s["k1"]))
    origem = np.vstack([x.min(0, keepdims=True) for x in G]).min(0) - 10
    def chaves(P):
        k = np.floor((P - origem) / VOX).astype(np.int64)
        return np.unique((k[:, 0] * 200003 + k[:, 1]) * 200017 + k[:, 2], return_counts=True)
    VC = [chaves(x) for x in G]
    V = [v for v, c in VC]
    # apoio total de cada voxel = soma dos centroides (de 1 mm) de todos os segmentos: superficie solida tem muitos,
    # camada fantasma/reflexo tem poucos. So a superficie solida entra no alvo de cobertura.
    kk = np.concatenate([v for v, c in VC]); cc = np.concatenate([c for v, c in VC])
    todas_k, inv = np.unique(kk, return_inverse=True)
    apoio = np.bincount(inv, weights=cc)
    _, cont = np.unique(kk, return_counts=True)
    solido = apoio >= a.apoio_min
    todas, cont = todas_k[solido], cont[solido]
    print(f"   voxels solidos (apoio >= {a.apoio_min}): {len(todas):,} de {len(todas_k):,} na uniao bruta", flush=True)
    alvo = {int(k): min(a.redundancia, int(c)) for k, c in zip(todas, cont)}
    idx = {int(k): i for i, k in enumerate(todas)}
    falta = np.array([alvo[int(k)] for k in todas], dtype=np.int32)
    total = int(falta.sum())
    I = [np.array([idx[int(k)] for k in v if int(k) in idx], dtype=np.int64) for v in V]
    print(f"{len(nos)} segmentos | {len(todas):,} voxels na uniao | deficit inicial {total:,}", flush=True)
    escolhidos, historico = [], []
    ainda = set(range(len(nos)))
    while falta.sum() > a.folga * total and ainda:
        ganho = {i: int(np.minimum(falta[I[i]], 1).sum()) for i in ainda}
        i = max(ainda, key=lambda j: (ganho[j], -vies.get(nos[j]["scan"], 150)))
        if ganho[i] == 0:
            break
        falta[I[i]] = np.maximum(falta[I[i]] - 1, 0)
        ainda.discard(i)
        escolhidos.append(i)
        cob1 = 1 - np.mean([1 if (c >= 1) else 0 for c in []]) if False else None
        historico.append(dict(n=len(escolhidos), scan=nos[i]["scan"], seg=nos[i]["seg"], falta_pct=float(falta.sum() / total * 100)))
        if len(escolhidos) % 10 == 0:
            print(f"  {len(escolhidos)} segmentos: ainda falta {falta.sum()/total*100:.2f}% do deficit", flush=True)
    quadros = sum(nos[i]["k1"] - nos[i]["k0"] + 1 for i in escolhidos)
    por_scan = {}
    for i in escolhidos:
        por_scan.setdefault(nos[i]["scan"], []).append(nos[i]["seg"])
    print(f"escolhidos {len(escolhidos)} de {len(nos)} segmentos (~{quadros} quadros) de {len(por_scan)} scans | deficit restante {falta.sum()/total*100:.2f}%")
    for g, v in sorted(por_scan.items()):
        print(f"   {g}: {len(v)} de {len(ind[g])} segmentos")
    json.dump(dict(redundancia=a.redundancia, deficit_restante_pct=float(falta.sum() / total * 100), escolhidos=[nos[i] for i in escolhidos],
                   por_scan=por_scan, historico=historico), open(a.saida, "w"), indent=1)


if __name__ == "__main__":
    main()
