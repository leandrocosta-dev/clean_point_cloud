"""Importa um .revo na biblioteca do Revo Scan 6 e abre um scan (automacao por UI; usa o mouse).

    python pipeline/abrir_no_revo.py PROJETO.revo [--scan GUID] [--so-importar]

O Revo Scan precisa estar fechado ou ser uma sessao em ingles aberta pela propria automacao; sessao do usuario
em outra lingua e recusada por revo_ui.abrir(). Nao mexa no mouse durante a execucao.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import revo_ui as R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("revo")
    ap.add_argument("--scan", help="guid do scan a abrir (padrao: o primeiro scan do projeto)")
    ap.add_argument("--so-importar", action="store_true", help="importa na biblioteca sem abrir nenhum scan")
    a = ap.parse_args()
    revo = os.path.abspath(a.revo)
    nos = json.load(open(revo, encoding="utf-8"))["nodes"]
    scans = [n["guid"] for n in nos if n.get("type") != 5]
    alvo = a.scan or scans[0]
    ui = R.RevoUI.abrir()
    pastas = ui.carregar_projeto(revo)
    print("importado na biblioteca:", ", ".join(pastas), flush=True)
    if not a.so_importar:
        print("scan aberto:", ui.abrir_scan(alvo, pastas[alvo]), flush=True)
    R.restaurar_janelas()


if __name__ == "__main__":
    main()
