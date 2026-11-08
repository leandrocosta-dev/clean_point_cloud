"""Formulario unico para o ciclo de precisao (CLI e catalogo GUI).

python pipeline/rodar_ciclo_precisao.py --acao iniciar --projeto PROJ --saida NOVA
python pipeline/rodar_ciclo_precisao.py --acao fundir --ciclo EXP
python pipeline/rodar_ciclo_precisao.py --acao consenso --ciclo EXP --poses poses.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--acao",
        required=True,
        choices=[
            "iniciar",
            "limpar",
            "triagem",
            "consenso",
            "qualidade",
            "fundir",
            "revo",
            "registrar-revo",
            "comparar",
            "inspecionar",
            "consultar",
            "corrigir",
            "aprovar",
            "exportar-merge",
            "status",
        ],
    )
    ap.add_argument("--projeto", help="pasta de origem, somente para iniciar")
    ap.add_argument("--ciclo", help="pasta do experimento nas demais acoes")
    ap.add_argument("--saida", help="pasta/HTML novo, conforme a acao")
    ap.add_argument("--scan", help="ID na revisao ativa; consultar em status/manifesto")
    ap.add_argument(
        "--motor", choices=["cuda"], default="cuda", help=argparse.SUPPRESS
    )  # compatibilidade: so existe GPU
    ap.add_argument("--voxel", type=float, default=0.08)
    ap.add_argument("--regioes", help="JSON de selecao 3D para correcao")
    ap.add_argument("--relatorio", help="revo_pipeline.json da entrada congelada")
    ap.add_argument("--parecer", help="JSON da revisao 3D para liberar merge")
    ap.add_argument("--configuracao", help="JSON de parametros da primeira limpeza, opcional")
    ap.add_argument("--poses", help="JSON das poses alinhadas, obrigatorio para consenso")
    ap.add_argument("--caixa", nargs=6, type=float, help="xmin ymin zmin xmax ymax zmax, em mm")
    a = ap.parse_args()
    required = {
        "iniciar": ["projeto", "saida"],
        "limpar": ["ciclo"],
        "triagem": ["ciclo"],
        "consenso": ["ciclo", "poses"],
        "qualidade": ["ciclo", "poses"],
        "fundir": ["ciclo"],
        "revo": ["ciclo"],
        "registrar-revo": ["ciclo", "relatorio"],
        "comparar": ["ciclo"],
        "inspecionar": ["ciclo", "scan", "saida"],
        "consultar": ["ciclo", "scan", "caixa", "saida"],
        "corrigir": ["ciclo", "regioes"],
        "aprovar": ["ciclo", "parecer"],
        "exportar-merge": ["ciclo", "saida"],
        "status": ["ciclo"],
    }
    for k in required[a.acao]:
        if getattr(a, k) is None:
            ap.error(f"--{k} obrigatorio para {a.acao}")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("ciclo_precisao.py")),
        a.acao,
        a.projeto if a.acao == "iniciar" else a.ciclo,
    ]
    for k in required[a.acao]:
        if k in ("projeto", "ciclo"):
            continue
        v = getattr(a, k)
        cmd += ["--" + k] + ([str(x) for x in v] if isinstance(v, list) else [v])
    if a.acao == "fundir":
        cmd += ["--voxel", str(a.voxel)]
    if a.acao == "limpar" and a.configuracao:
        cmd += ["--configuracao", a.configuracao]
    if a.acao == "triagem" and a.saida:
        cmd += ["--saida", a.saida]
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == "__main__":
    main()
