"""Escolhe o conjunto MINIMO de segmentos (15 quadros) que cobre a peca, com redundancia dupla onde existir.

    python pipeline/selecionar_segmentos.py <pasta_segmentos> <segmentos.json> --saida <selecao.json> [--redundancia 2]

Cobertura em voxels de 2 mm no referencial comum. Cada voxel da uniao de todas as capturas pede
min(redundancia, quantos segmentos o veem). Guloso: a cada passo entra o segmento que mais reduz o que falta
(empate: scan de menor vies). Para quando o que falta cai abaixo de `--folga` da uniao. A redundancia 2 e para
promediar o ruido de medicao; cobertura simples e o minimo para nao cortar a peca.
"""

import argparse
import json
import os

import gpu
import numpy as np
from alinhar_varios import baixa_prioridade
from registro_gpu import transformar

VOX = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("segmentos")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--redundancia", type=int, default=2)
    ap.add_argument("--folga", type=float, default=0.003, help="para quando faltar menos que isto da uniao")
    ap.add_argument("--por-scan")
    ap.add_argument(
        "--apoio-min",
        type=int,
        default=12,
        help="centroides de 1 mm minimos por voxel de 2 mm para contar como superficie",
    )
    a = ap.parse_args()
    baixa_prioridade()
    ind = json.load(open(os.path.join(a.pasta, "segmentos_indice.json")))
    pj = json.load(open(a.segmentos))["segmentos"]
    vies = {g: (v or {}).get("mediana_um", 150) for g, v in json.load(open(a.por_scan)).items()} if a.por_scan else {}
    xp = gpu.xp()
    nos, G = [], []
    for g in sorted(ind):
        if g not in pj:
            continue
        for s in ind[g]:
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            X = np.load(s["arquivo"]).astype(np.float64)
            G.append(transformar(gpu.para_gpu(X, xp.float64), T))
            nos.append(dict(scan=g, seg=s["seg"], k0=s["k0"], k1=s["k1"]))
    origem = xp.min(xp.stack([x.min(0) for x in G]), axis=0) - 10
    # voxels (chaves) de cada segmento e quantos centroides de 1 mm cada um tem
    VC = [gpu.agrupar(gpu.chaves_voxel(x, VOX, origem)) for x in G]
    # apoio total de cada voxel = soma dos centroides (de 1 mm) de todos os segmentos: superficie solida tem muitos,
    # camada fantasma/reflexo tem poucos. So a superficie solida entra no alvo de cobertura.
    kk = xp.concatenate([v for v, _, c in VC])
    cc = xp.concatenate([c for v, _, c in VC])
    todas_k, inv, cont = gpu.agrupar(kk)
    apoio = gpu.somar_por_grupo(cc, inv, len(todas_k))
    solido = apoio >= a.apoio_min
    todas, cont = todas_k[solido], cont[solido]
    print(f"   voxels solidos (apoio >= {a.apoio_min}): {len(todas):,} de {len(todas_k):,} na uniao bruta", flush=True)
    falta = xp.minimum(cont, a.redundancia).astype(xp.int32)
    total = int(falta.sum())
    # CSR: posicao (em `todas`) de cada voxel solido de cada segmento
    ids_seg, pos = [], []
    for i, (v, _, _) in enumerate(VC):
        p = xp.clip(xp.searchsorted(todas, v), 0, max(len(todas) - 1, 0))
        ok = (todas[p] == v) if len(todas) else xp.zeros(len(v), bool)
        pos.append(p[ok])
        ids_seg.append(xp.full(int(ok.sum()), i, xp.int64))
    pos, ids_seg = xp.concatenate(pos), xp.concatenate(ids_seg)
    print(f"{len(nos)} segmentos | {len(todas):,} voxels na uniao | deficit inicial {total:,}", flush=True)
    escolhidos, historico = [], []
    ainda = set(range(len(nos)))
    while int(falta.sum()) > a.folga * total and ainda:
        gan = gpu.para_cpu(gpu.somar_por_grupo((falta[pos] >= 1), ids_seg, len(nos))).astype(
            np.int64
        )  # voxels que cada segmento ainda cobre
        ganho = {i: int(gan[i]) for i in ainda}
        i = max(ainda, key=lambda j: (ganho[j], -vies.get(nos[j]["scan"], 150)))
        if ganho[i] == 0:
            break
        pi = pos[ids_seg == i]
        falta[pi] = xp.maximum(falta[pi] - 1, 0)
        ainda.discard(i)
        escolhidos.append(i)
        1 - np.mean([1 if (c >= 1) else 0 for c in []]) if False else None
        historico.append(
            dict(
                n=len(escolhidos),
                scan=nos[i]["scan"],
                seg=nos[i]["seg"],
                falta_pct=float(int(falta.sum()) / total * 100),
            )
        )
        if len(escolhidos) % 10 == 0:
            print(
                f"  {len(escolhidos)} segmentos: ainda falta {int(falta.sum()) / total * 100:.2f}% do deficit",
                flush=True,
            )
    quadros = sum(nos[i]["k1"] - nos[i]["k0"] + 1 for i in escolhidos)
    por_scan = {}
    for i in escolhidos:
        por_scan.setdefault(nos[i]["scan"], []).append(nos[i]["seg"])
    print(
        f"escolhidos {len(escolhidos)} de {len(nos)} segmentos (~{quadros} quadros) de {len(por_scan)} scans | deficit restante {int(falta.sum()) / total * 100:.2f}%"
    )
    for g, v in sorted(por_scan.items()):
        print(f"   {g}: {len(v)} de {len(ind[g])} segmentos")
    json.dump(
        dict(
            redundancia=a.redundancia,
            deficit_restante_pct=float(int(falta.sum()) / total * 100),
            escolhidos=[nos[i] for i in escolhidos],
            por_scan=por_scan,
            historico=historico,
        ),
        open(a.saida, "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
