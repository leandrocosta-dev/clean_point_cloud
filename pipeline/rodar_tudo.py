"""Pipeline completa: limpa os quadros (limpar_raw) e funde/gera malha/exporta no Revo (revo_pipeline).

    python pipeline/rodar_tudo.py <pasta_do_projeto> --saida-ply PASTA [opcoes]

Etapa 1 grava <pasta_do_projeto>/../<nome>_limpo (ou --saida-limpo) SEM os fuse.ply/fuse_mesh.ply/mark.ply
originais; etapa 2 importa esse .revo no Revo Scan e exporta, por scan, a nuvem fundida e a malha em
PASTA/<scan>/. O original nunca e tocado. Opcoes da limpeza: --periferia-mm/--periferia-banda/
--periferia-alargue/--jobs. Opcoes do Revo: --ponto, --malha, --otimizar, --grade, --furos, --marcadores,
--manter-isolados, --scans. Nao mexa no mouse durante a etapa 2.
"""
import argparse
import os
import subprocess
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--saida-ply", required=True)
    ap.add_argument("--saida-limpo")
    ap.add_argument("--periferia-mm", type=float, default=0.0)
    ap.add_argument("--periferia-banda", type=float, default=2.5)
    ap.add_argument("--periferia-alargue", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--scans", nargs="+")
    ap.add_argument("--ponto", type=float)
    ap.add_argument("--malha", choices=("rapida", "alta"), default="rapida")
    ap.add_argument("--otimizar", choices=("nenhum", "baixo", "alto"), default="baixo")
    ap.add_argument("--grade", type=float)
    ap.add_argument("--furos", action="store_true")
    ap.add_argument("--marcadores", action="store_true")
    ap.add_argument("--manter-isolados", action="store_true")
    a = ap.parse_args()

    proj = os.path.abspath(a.projeto)
    revo = [f for f in os.listdir(proj) if f.endswith(".revo")][0]
    saida = os.path.abspath(a.saida_limpo or os.path.join(os.path.dirname(proj), os.path.splitext(revo)[0] + "_limpo"))

    c1 = [sys.executable, os.path.join(AQUI, "limpar_raw.py"), proj, "--saida", saida, "--sem-ply-antigos",
          "--periferia-mm", str(a.periferia_mm), "--periferia-banda", str(a.periferia_banda),
          "--periferia-alargue", str(a.periferia_alargue), "--jobs", str(a.jobs)]
    if a.scans:
        c1 += ["--scans", *a.scans]
    print("== etapa 1: limpeza dos quadros", flush=True)
    if subprocess.run(c1).returncode:
        sys.exit("limpeza falhou; o Revo nao foi acionado")

    novo = os.path.join(saida, os.path.splitext(revo)[0] + "_limpo.revo")
    if not os.path.exists(novo):
        novo = os.path.join(saida, [f for f in os.listdir(saida) if f.endswith(".revo")][0])
    c2 = [sys.executable, os.path.join(AQUI, "revo_pipeline.py"), novo, "--saida", os.path.abspath(a.saida_ply),
          "--malha", a.malha, "--otimizar", a.otimizar]
    for flag, v in (("--ponto", a.ponto), ("--grade", a.grade)):
        if v is not None:
            c2 += [flag, str(v)]
    for flag, on in (("--furos", a.furos), ("--marcadores", a.marcadores), ("--manter-isolados", a.manter_isolados)):
        if on:
            c2.append(flag)
    if a.scans:
        c2 += ["--scans", ",".join(a.scans)]
    print("== etapa 2: Revo Scan (fusao, malha, exportacao)", flush=True)
    sys.exit(subprocess.run(c2).returncode)


if __name__ == "__main__":
    main()
