"""Pipeline completa: limpa os quadros (limpar_raw) e funde/gera malha/exporta no Revo (revo_pipeline).

    python pipeline/rodar_tudo.py <pasta_do_projeto> --saida-ply PASTA [opcoes]

Com --unir A B (duas partes da mesma peca) ha uma etapa extra entre as duas: une os frames de B em A
(uniao_scans.py, --lados opostos = peca virada, mesmo = mesma face; so quem escaneou sabe) e o Revo funde
o scan unico <A>_uniao. Etapa 1 grava <pasta_do_projeto>/../<nome>_limpo (ou --saida-limpo) SEM os fuse.ply/fuse_mesh.ply/mark.ply
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
    ap.add_argument("--saida-ply", help="obrigatorio, exceto com --so-limpar")
    ap.add_argument("--saida-limpo")
    ap.add_argument("--periferia-mm", type=float, default=0.0)
    ap.add_argument("--periferia-banda", type=float, default=2.5)
    ap.add_argument("--periferia-alargue", type=float, default=20.0)
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--scans", nargs="+")
    ap.add_argument("--unir", nargs=2, metavar=("A", "B"), help="guids das duas partes da mesma peca")
    ap.add_argument("--lados", choices=("opostos", "mesmo", "auto"), default="opostos", help="so com --unir")
    ap.add_argument("--saida-uniao", help="so com --unir (padrao: <saida-limpo>_uniao)")
    ap.add_argument("--ponto", type=float)
    ap.add_argument("--malha", choices=("rapida", "alta"), default="rapida")
    ap.add_argument("--otimizar", choices=("nenhum", "baixo", "alto"), default="baixo")
    ap.add_argument("--grade", type=float)
    ap.add_argument("--so-limpar", action="store_true",
                    help="para depois da limpeza (e da uniao, se houver): entrega so o .revo limpo, sem abrir o Revo Scan")
    ap.add_argument("--entregaveis", choices=("nuvem+malha", "nuvem", "malha"), default="nuvem+malha")
    ap.add_argument("--furos", action="store_true")
    ap.add_argument("--marcadores", action="store_true")
    ap.add_argument("--manter-isolados", action="store_true")
    a = ap.parse_args()

    if not a.so_limpar and not a.saida_ply:
        ap.error("--saida-ply e obrigatorio (ou use --so-limpar)")
    proj = os.path.abspath(a.projeto)
    revo = [f for f in os.listdir(proj) if f.endswith(".revo")][0]
    saida = os.path.abspath(a.saida_limpo or os.path.join(os.path.dirname(proj), os.path.splitext(revo)[0] + "_limpo"))

    c1 = [sys.executable, os.path.join(AQUI, "limpar_raw.py"), proj, "--saida", saida, "--sem-ply-antigos",
          "--periferia-mm", str(a.periferia_mm), "--periferia-banda", str(a.periferia_banda),
          "--periferia-alargue", str(a.periferia_alargue), "--jobs", str(a.jobs)]
    if a.unir:
        c1 += ["--scans", *a.unir]
    elif a.scans:
        c1 += ["--scans", *a.scans]
    print("== etapa 1: limpeza dos quadros", flush=True)
    if subprocess.run(c1).returncode:
        sys.exit("limpeza falhou; o Revo nao foi acionado")

    if a.unir:
        nome = a.unir[0] + "_uniao"
        uniao = os.path.abspath(a.saida_uniao or saida + "_uniao")
        print("== etapa 1b: uniao das partes (frames de B levados para A)", flush=True)
        cu = [sys.executable, os.path.join(AQUI, "uniao_scans.py"), saida, "--scans", *a.unir, "--lados", a.lados,
              "--saida", uniao, "--nome", nome]
        if subprocess.run(cu).returncode:
            sys.exit("uniao falhou; o Revo nao foi acionado")
        novo = os.path.join(uniao, nome + ".revo")
    else:
        novo = os.path.join(saida, os.path.splitext(revo)[0] + "_limpo.revo")
        if not os.path.exists(novo):
            novo = os.path.join(saida, [f for f in os.listdir(saida) if f.endswith(".revo")][0])
    if a.so_limpar:
        print(f"== concluido sem Revo. Projeto .revo limpo: {novo}", flush=True)
        sys.exit(0)
    c2 = [sys.executable, os.path.join(AQUI, "revo_pipeline.py"), novo, "--saida", os.path.abspath(a.saida_ply),
          "--malha", a.malha, "--otimizar", a.otimizar,
          "--entregaveis", a.entregaveis]
    for flag, v in (("--ponto", a.ponto), ("--grade", a.grade)):
        if v is not None:
            c2 += [flag, str(v)]
    for flag, on in (("--furos", a.furos), ("--marcadores", a.marcadores), ("--manter-isolados", a.manter_isolados)):
        if on:
            c2.append(flag)
    if a.scans and not a.unir:
        c2 += ["--scans", ",".join(a.scans)]
    print("== etapa 2: Revo Scan (fusao, malha, exportacao)", flush=True)
    sys.exit(subprocess.run(c2).returncode)


if __name__ == "__main__":
    main()
