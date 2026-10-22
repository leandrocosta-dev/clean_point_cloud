"""Monta o scan unico (frames de todos os scans alinhados por segmento) e grava o projeto .revo.

    python pipeline/montar_pecas.py <projeto_limpo> <segmentos.json> --saida <pasta> --nome <guid> [--excluir g1 g2]
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uniao_scans as U

ap = argparse.ArgumentParser()
ap.add_argument("projeto"); ap.add_argument("segmentos")
ap.add_argument("--saida", required=True); ap.add_argument("--nome", required=True)
ap.add_argument("--base"); ap.add_argument("--excluir", nargs="*", default=[])
ap.add_argument("--selecao", help="selecao.json de selecionar_segmentos.py: so esses segmentos entram")
a = ap.parse_args()
pj = json.load(open(a.segmentos))
poses = {g: sorted(v, key=lambda s: s["seg"]) for g, v in pj["segmentos"].items() if g not in a.excluir}
if a.selecao:
    esc = json.load(open(a.selecao))["por_scan"]
    poses = {g: [x for x in v if x["seg"] in esc.get(g, [])] for g, v in poses.items() if g in esc}
base = a.base or pj["referencia"]
if base not in poses:
    base = max(poses, key=lambda g: len(poses[g]))
print("scans:", len(poses), "| segmentos:", sum(len(v) for v in poses.values()), "| base:", base)
n = U.unir_varios(os.path.abspath(a.projeto), poses, base, a.saida, a.nome)
json.dump(dict(base=base, excluidos=a.excluir, quadros=n), open(os.path.join(a.saida, "montagem.json"), "w"), indent=1)
print("quadros:", n["total"], "| saida:", a.saida)
