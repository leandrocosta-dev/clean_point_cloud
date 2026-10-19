"""Linha de comando.

Limpar uma nuvem:
    python run.py limpar peca.ply --saida resultado/
    python run.py limpar peca.ply --saida resultado/ --llm --config minha_config.json

Mesclar varias varreduras da mesma peca (captura dividida em partes):
    python run.py mesclar parte1.ply parte2.ply parte3.ply --saida resultado/
"""
import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

from moviai_scan import Config, process
from moviai_scan.align import merge_scans
from moviai_scan.geom import mean_spacing


def main():
    ap = argparse.ArgumentParser(description="MoviAI Scan")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("limpar")
    a.add_argument("entrada")
    a.add_argument("--saida", default="saida")
    a.add_argument("--config")
    a.add_argument("--llm", action="store_true", help="inspecao final por LLM (1 chamada)")
    a.add_argument("--base", choices=["auto", "on", "off"])

    b = sub.add_parser("mesclar")
    b.add_argument("entradas", nargs="+")
    b.add_argument("--saida", default="saida")
    b.add_argument("--limpar-depois", action="store_true")

    args = ap.parse_args()
    cfg = Config.load(args.config) if getattr(args, "config", None) else Config()

    if args.cmd == "limpar":
        if args.base:
            cfg.base_mode = args.base
        rel = process(args.entrada, args.saida, cfg, use_llm=args.llm)
        sys.exit(0 if rel["status"] == "aprovado" else 2)

    if args.cmd == "mesclar":
        os.makedirs(args.saida, exist_ok=True)
        scans = [np.asarray(o3d.io.read_point_cloud(p).points) for p in args.entradas]
        sp = float(np.median([mean_spacing(s) for s in scans]))
        merged, rel = merge_scans(scans, sp, cfg.seed)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged)).voxel_down_sample(sp * 0.8)
        out = os.path.join(args.saida, "mesclado.ply")
        o3d.io.write_point_cloud(out, pcd)
        with open(os.path.join(args.saida, "mesclado_relatorio.json"), "w", encoding="utf-8") as f:
            json.dump(rel, f, ensure_ascii=False, indent=2)
        for r in rel:
            print(r.get("scan"), r.get("status"), r.get("fitness", ""), r.get("rmse", ""))
        print(f"salvo em {out}")
        if args.limpar_depois:
            process(out, args.saida, cfg)


if __name__ == "__main__":
    main()
