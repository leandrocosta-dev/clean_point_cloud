"""Limpeza quadro a quadro de um projeto do Revo Scan.

Uso:
    python limpar_revo.py <pasta_do_projeto> --scans 20261017_172640 [--saida <pasta>]

Gera uma COPIA do projeto so com os scans pedidos, com prato e pecas de apoio
apagados de cada quadro. Poses inalteradas. O original nao e tocado.
"""
import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from moviai_scan.revo_projeto import RevoScan, segmentar_peca, limpar_scan, gravar_revo, render_comparacao

ERRO_MAX_CALIB = 0.6   # mm; acima disso o scan nao e limpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projeto")
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--saida")
    a = ap.parse_args()
    proj = os.path.abspath(a.projeto)
    revo = [f for f in os.listdir(proj) if f.endswith(".revo")][0]
    nome = os.path.splitext(revo)[0] + "_limpo"
    saida = os.path.abspath(a.saida or os.path.join(proj, "saida", nome))
    if os.path.exists(saida):
        sys.exit(f"destino ja existe: {saida}")
    os.makedirs(os.path.join(saida, "data"))
    rel = {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for s in a.scans:
            t0 = time.time()
            print(f"== {s}", flush=True)
            scan = RevoScan(os.path.join(proj, "data", s), tmp)
            try:
                modos = sorted(set(int(m) for m, p in zip(scan.modo, scan.tem_prof) if p))
                erros = {}
                for m in modos:
                    K, e = scan.calibrar(m)
                    erros[m] = e
                    print(f"   calibracao modo {m}: fx {K[0]:.1f} fy {K[1]:.1f} cx {K[2]:.1f} cy {K[3]:.1f} | erro {e:.3f} mm", flush=True)
                if max(erros.values()) > ERRO_MAX_CALIB:
                    print(f"   PULADO: calibracao com erro acima de {ERRO_MAX_CALIB} mm", flush=True)
                    rel[s] = dict(status="pulado", erros=erros)
                    continue
                seg = segmentar_peca(scan)
                r = limpar_scan(scan, seg, os.path.join(saida, "data", s))
                render_comparacao(scan, seg, os.path.join(saida, f"{s}_antes_depois.png"))
                rel[s] = dict(status="limpo", erros=erros, peca=seg["peca"], **r,
                              tempo_s=round(time.time() - t0))
            finally:
                scan.fechar()
    limpos = [s for s in rel if rel[s]["status"] == "limpo"]
    if limpos:
        gravar_revo(os.path.join(proj, revo), saida, nome, limpos)
    json.dump(rel, open(os.path.join(saida, "relatorio_limpeza.json"), "w", encoding="utf-8"), indent=2)
    print("saida:", saida)


if __name__ == "__main__":
    main()
