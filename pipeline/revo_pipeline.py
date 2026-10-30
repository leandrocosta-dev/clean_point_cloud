"""Fusao + malha + exportacao no Revo Scan 6 para todos os scans de um .revo, por codigo.

    python pipeline/revo_pipeline.py PROJ.revo --saida PASTA --ponto 0.41 --malha alta --grade 0.41

Passos (revo_ui.py): importa o .revo na biblioteca do Revo, grava os parametros no .revox de cada scan
importado (sem mexer em painel nenhum), abre o scan, Fusion > Apply, Meshing > Apply, Export Model.
Cada etapa e conferida no log do Revo: o passo de pontos, o modo da malha, o nivel de otimizacao e a
grade realmente usados precisam ser os pedidos, senao o scan vai marcado como divergente no relatorio.
Saida: PASTA/<guid>/*.ply (nuvem fundida e malha) e PASTA/revo_pipeline.json.
O mouse real ainda e usado para os cliques de abrir/Apply/exportar: nao mexa no PC durante a execucao.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revo_ui as R


ENTREGAVEIS = {"nuvem+malha": ("nuvem", "malha"), "nuvem": ("nuvem",), "malha": ("malha",)}
PREFIXO = {"nuvem": "Fused", "malha": "Mesh"}


def guids_do_projeto(revo):
    dados = os.path.join(os.path.dirname(os.path.abspath(revo)), "data")
    marcadores = {n["guid"] for n in json.load(open(os.path.abspath(revo), encoding="utf-8")).get("nodes", []) if n.get("type") == 5}
    return sorted(d for d in os.listdir(dados) if os.path.isdir(os.path.join(dados, d)) and d not in marcadores)


def processar(ui, guid, pasta, args):
    r = {"guid": guid, "pasta_biblioteca": pasta}
    t0 = time.time()
    ui.abrir_scan(guid, pasta)
    if getattr(args, "so_exportar", False):
        r["fusao"] = r["malha"] = None                 # usa o que ja esta no scan (maior contador de nome)
    else:
        r["fusao"] = ui.fundir()
        r["malha"] = ui.malha() if "malha" in ENTREGAVEIS[args.entregaveis] else None
        # malha grande deixa o Revo lento/sem responder ao dialogo de exportacao na mesma sessao: reabre antes de exportar
        ui.fechar()
        ui = R.RevoUI.abrir()
        ui.abrir_scan(guid, pasta)
    prefixos = tuple(PREFIXO[e] for e in ENTREGAVEIS[getattr(args, "entregaveis", "nuvem+malha")])
    r["arquivos"] = ui.exportar(os.path.join(args.saida, guid), prefixos=prefixos)
    r["segundos"] = round(time.time() - t0, 1)
    diverge = []
    if r["fusao"] is None:
        r["divergencias"] = diverge
        return r
    if args.ponto is not None and abs(r["fusao"]["ponto_mm"] - args.ponto) > 0.005:
        diverge.append(f"ponto {r['fusao']['ponto_mm']} != {args.ponto}")
    if r["malha"] is None:                              # so a nuvem foi pedida
        r["divergencias"] = diverge
        return r
    if r["malha"]["modo"] != R.MALHA_MODO[args.malha]:
        diverge.append(f"modo da malha {r['malha']['modo']}")
    if args.malha == "rapida" and r["malha"]["otimizar"] != R.OTIMIZAR[args.otimizar]:
        diverge.append(f"otimizar {r['malha']['otimizar']}")
    if args.malha == "alta" and args.grade is not None and abs(r["malha"]["grade_mm"] - args.grade) > 0.005:
        diverge.append(f"grade {r['malha']['grade_mm']} != {args.grade}")
    if r["malha"]["furos"] != bool(args.furos):
        diverge.append(f"furos {r['malha']['furos']}")
    r["divergencias"] = diverge
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("revo")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--ponto", type=float, help="Point Distance da fusao, mm (padrao: o automatico do Revo)")
    ap.add_argument("--malha", choices=list(R.MALHA_MODO), default="rapida")
    ap.add_argument("--otimizar", choices=list(R.OTIMIZAR), default="baixo", help="so na malha rapida")
    ap.add_argument("--grade", type=float, help="Grid Size, mm, so na malha alta")
    ap.add_argument("--furos", action="store_true", help="Hole Filling")
    ap.add_argument("--marcadores", action="store_true", help="Remove marker points")
    ap.add_argument("--manter-isolados", action="store_true", help="desliga Remove Isolation")
    ap.add_argument("--entregaveis", choices=list(ENTREGAVEIS), default="nuvem+malha",
                    help="o que exportar por scan; so 'nuvem' nao gera a malha (mais rapido)")
    ap.add_argument("--scans", help="guids separados por virgula (padrao: todos do projeto)")
    ap.add_argument("--reusar", action="store_true",
                    help="nao importa: usa a copia mais recente de cada scan na biblioteca (retomar apos falha)")
    ap.add_argument("--so-exportar", action="store_true", help="nao funde nem gera malha: exporta o que ja existe no scan (com --reusar)")
    ap.add_argument("--fechar", action="store_true", help="fecha o Revo no fim")
    args = ap.parse_args()

    guids = args.scans.split(",") if args.scans else guids_do_projeto(args.revo)
    os.makedirs(args.saida, exist_ok=True)
    ui = R.RevoUI.abrir()
    if args.reusar:
        pastas = {g: R.pasta_mais_recente(g) for g in guids}
    else:
        pastas = ui.carregar_projeto(args.revo)
    faltam = [g for g in guids if g not in pastas]
    if faltam:
        sys.exit(f"scans que nao entraram na biblioteca: {faltam}")
    for g in guids:                       # antes de abrir qualquer um: o Revo so le o .revox na 1a abertura
        R.parametros(g, ponto_mm=args.ponto, malha=args.malha, otimizar=args.otimizar if args.malha == "rapida" else None,
                     grade_mm=args.grade if args.malha == "alta" else None, furos=args.furos,
                     marcadores=args.marcadores, isolados=not args.manter_isolados,
                     revox=os.path.join(pastas[g], g + ".revox"))
    relatorio = {"revo": os.path.abspath(args.revo), "parametros": {k: v for k, v in vars(args).items() if k not in ("revo", "saida")},
                 "scans": []}
    for g in guids:
        try:
            if not ui.vivo():             # Revo reiniciado a cada scan: sessao longa acumula memoria e ja travou
                ui = R.RevoUI.abrir()
            r = processar(ui, g, pastas[g], args)
        except Exception as e:            # um scan com problema nao derruba os outros
            r = {"guid": g, "erro": f"{type(e).__name__}: {e}"}
            ui._fechar_avisos()
        print(json.dumps(r, ensure_ascii=False), flush=True)
        relatorio["scans"].append(r)
        try:
            ui.fechar()
        except Exception:
            pass
    with open(os.path.join(args.saida, "revo_pipeline.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, indent=1, ensure_ascii=False)
    if args.fechar:
        ui.fechar()
    R.restaurar_janelas()
    ruins = [s["guid"] for s in relatorio["scans"] if "erro" in s or s.get("divergencias")]
    print("ok" if not ruins else f"atencao: {ruins}")
    sys.exit(1 if ruins else 0)


if __name__ == "__main__":
    main()
