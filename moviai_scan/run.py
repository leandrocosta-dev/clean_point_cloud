"""MoviAI Scan: linha de comando unica.

Nuvem PLY exportada do Revo Scan (Open3D):
    python run.py limpar peca.ply --saida resultado/ [--llm] [--base auto|on|off] [--config cfg.json]
    python run.py mesclar parte1.ply parte2.ply --saida resultado/ [--limpar-depois]
    python run.py malha limpo.ply malha.stl [--depth 9]

CloudCompare em linha de comando:
    python run.py cc limpar a.ply b.ply --saida limpo.ply [--config cfg.json] [--dry-run]
    python run.py cc deriva limpo.ply --saida sem_deriva.ply [--suporte 500]

Projeto do Revo Scan quadro a quadro (o original nunca e alterado):
    python run.py revo info <projeto> [--calibrar]
    python run.py revo limpar <projeto> --scans 20261017_172640 [--config cfg.json] [--faixa-base 2.5]
    python run.py revo fundir <projeto> --scan S --saida nuvem.ply [--correcao correcao.json]
    python run.py revo giro <projeto> --scan S --ref 0-69 --alvo 148-247 --rampa 70-147 \\
        --centro X,Y,Z --raio 22.9 --altura 46 [--base-de raw_preview_relatorio.json] --saida pasta/
    python run.py revo aplicar <projeto> --correcao pasta/correcao.json   (de 'revo giro' ou de 'revo limpar')
"""
import argparse
import functools
import json
import os
import sys
import tempfile

from moviai_scan import Config

log = functools.partial(print, flush=True)


def intervalo(txt):
    a, b = (int(x) for x in txt.split("-"))
    return a, b


def vetor3(txt):
    v = [float(x) for x in txt.split(",")]
    if len(v) != 3:
        raise argparse.ArgumentTypeError("esperado X,Y,Z")
    return v


def cmd_limpar(a):
    from moviai_scan.pipeline import process
    cfg = Config.load(a.config) if a.config else Config()
    if a.base:
        cfg.base_mode = a.base
    rel = process(a.entrada, a.saida, cfg, use_llm=a.llm, log=log)
    return 0 if rel["status"] == "aprovado" else 2


def cmd_mesclar(a):
    import numpy as np
    import open3d as o3d
    from moviai_scan.align import merge_scans
    from moviai_scan.geom import mean_spacing
    from moviai_scan.pipeline import process
    cfg = Config()
    os.makedirs(a.saida, exist_ok=True)
    scans = [np.asarray(o3d.io.read_point_cloud(p).points) for p in a.entradas]
    sp = float(np.median([mean_spacing(s) for s in scans]))
    merged, rel = merge_scans(scans, sp, cfg.seed, max_rmse_mult=cfg.align_max_rmse_mult)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(merged)).voxel_down_sample(sp * 0.8)
    out = os.path.join(a.saida, "mesclado.ply")
    o3d.io.write_point_cloud(out, pcd)
    with open(os.path.join(a.saida, "mesclado_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, ensure_ascii=False, indent=2)
    for r in rel:
        log(r.get("scan"), r.get("status"), r.get("fitness", ""), r.get("rmse", ""),
            "; ".join(r.get("motivo_ambiguidade", [])))
    log(f"salvo em {out}")
    if a.limpar_depois:
        process(out, a.saida, cfg, log=log)
    return 0 if all(r["status"] in ("referencia", "mesclado") for r in rel) else 2


def cmd_malha(a):
    from moviai_scan.malha import gerar_malha
    gerar_malha(a.entrada, a.saida, a.depth, a.normal_radius, a.normal_max_nn,
                a.density_quantile, a.max_hole_size, log=log)


def cmd_cc_limpar(a):
    from moviai_scan import cloudcompare
    cloudcompare.limpar(a.scans, a.saida, a.cc, a.config, a.dry_run, a.manter_temp, a.componente, log=log)


def cmd_cc_deriva(a):
    from moviai_scan import cloudcompare
    cloudcompare.corrigir_deriva(a.entrada, a.saida, a.cc, a.suporte, a.epsilon, a.manter_temp, log=log)


def cmd_revo_info(a):
    from moviai_scan.revo.formato import RevoScan, scans_do_projeto, calibrar_modos
    from moviai_scan.revo.info import resumo
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for s in a.scans or scans_do_projeto(a.projeto):
            with RevoScan(os.path.join(a.projeto, "data", s), tmp) as scan:
                resumo(scan, a.saltos, log)
                if a.calibrar:
                    calibrar_modos(scan, log)


def cmd_revo_limpar(a):
    from moviai_scan.revo.limpeza import limpar_projeto
    cfg = Config.load(a.config) if a.config else Config()
    _, rel = limpar_projeto(a.projeto, a.scans, a.saida, cfg, a.faixa_base, log)
    return 0 if all(r["status"] == "aprovado" for r in rel.values()) else 2


def cmd_revo_fundir(a):
    from moviai_scan.revo.formato import RevoScan, exigir_calibracao, gravar_ply_xyz
    from moviai_scan.revo.giro import fundir
    cor = None
    if a.correcao:
        with open(a.correcao, encoding="utf-8") as f:
            cor = json.load(f)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        with RevoScan(os.path.join(a.projeto, "data", a.scan), tmp) as scan:
            exigir_calibracao(scan, log)
            gravar_ply_xyz(a.saida, fundir(scan, cor, a.voxel, log))
    log(f"salvo em {a.saida}")


def cmd_revo_giro(a):
    from moviai_scan.revo.giro import preparar_correcao
    base = None
    if a.base_de:
        with open(a.base_de, encoding="utf-8") as f:
            b = json.load(f)["base"]
        if not b:
            raise ValueError(f"{a.base_de} nao tem base detectada")
        base = (b["n"], b["d"])
    preparar_correcao(a.projeto, a.scan, a.ref, a.alvo, a.rampa, a.centro, a.raio, a.altura, a.saida,
                      a.ate, base, log)


def cmd_revo_aplicar(a):
    from moviai_scan.revo.giro import aplicar
    aplicar(a.projeto, a.correcao, a.saida, log)


def parser():
    ap = argparse.ArgumentParser(description="MoviAI Scan")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("limpar", help="limpa uma nuvem PLY (base, ruido, ilhas, trechos deslocados)")
    p.add_argument("entrada")
    p.add_argument("--saida", default="saida")
    p.add_argument("--config")
    p.add_argument("--llm", action="store_true", help="inspecao final por LLM (1 chamada)")
    p.add_argument("--base", choices=["auto", "on", "off"])
    p.set_defaults(func=cmd_limpar)

    p = sub.add_parser("mesclar", help="mescla varreduras separadas da mesma peca")
    p.add_argument("entradas", nargs="+")
    p.add_argument("--saida", default="saida")
    p.add_argument("--limpar-depois", action="store_true")
    p.set_defaults(func=cmd_mesclar)

    p = sub.add_parser("malha", help="malha Poisson a partir de nuvem limpa")
    p.add_argument("entrada")
    p.add_argument("saida", help=".ply, .obj ou .stl")
    p.add_argument("--depth", type=int, default=9, help="profundidade do octree do Poisson (8-10 tipico)")
    p.add_argument("--normal-radius", type=float, default=1.0, help="raio de busca para estimar normais")
    p.add_argument("--normal-max-nn", type=int, default=30, help="max vizinhos para estimar/orientar normais")
    p.add_argument("--density-quantile", type=float, default=0.02,
                   help="quantil de densidade abaixo do qual vertices sao removidos (0 desativa)")
    p.add_argument("--max-hole-size", type=float, default=0.0,
                   help="tamanho maximo de furo a fechar (0 desativa; em unidades do modelo)")
    p.set_defaults(func=cmd_malha)

    cc = sub.add_parser("cc", help="CloudCompare em linha de comando").add_subparsers(dest="cc_cmd", required=True)
    p = cc.add_parser("limpar", help="ICP + componente da peca + SOR + ruido + crop + reamostragem")
    p.add_argument("scans", nargs="*")
    p.add_argument("--saida", required=True)
    p.add_argument("--cc", help="caminho do CloudCompare.exe")
    p.add_argument("--config", help="JSON sobreposto a cloudcompare_padrao.json")
    p.add_argument("--dry-run", action="store_true", help="so mostra os comandos")
    p.add_argument("--manter-temp", action="store_true", help="nao apaga a pasta de staging")
    p.add_argument("--componente", help="pula a extracao e limpa direto este *_COMPONENT_*.ply")
    p.set_defaults(func=cmd_cc_limpar)
    p = cc.add_parser("deriva", help="remove passadas desalinhadas de peca cilindrica (RANSAC)")
    p.add_argument("entrada")
    p.add_argument("--saida", required=True)
    p.add_argument("--cc")
    p.add_argument("--suporte", type=int, default=500, help="minimo de pontos por cilindro no RANSAC")
    p.add_argument("--epsilon", type=float, default=0.001, help="distancia C2C para ponto duplicado")
    p.add_argument("--manter-temp", action="store_true")
    p.set_defaults(func=cmd_cc_deriva)

    rv = sub.add_parser("revo", help="projeto do Revo Scan quadro a quadro").add_subparsers(dest="revo_cmd",
                                                                                          required=True)
    p = rv.add_parser("info", help="quadros, modos e maiores saltos de pose")
    p.add_argument("projeto")
    p.add_argument("--scans", nargs="+")
    p.add_argument("--calibrar", action="store_true", help="ajusta a calibracao de cada modo")
    p.add_argument("--saltos", type=int, default=8)
    p.set_defaults(func=cmd_revo_info)
    p = rv.add_parser("limpar", help="copia do projeto limpa quadro a quadro (prato, apoio, ruido, ilhas, "
                                     "trechos com deriva) + correcao de pose proposta")
    p.add_argument("projeto")
    p.add_argument("--scans", nargs="+", required=True)
    p.add_argument("--saida")
    p.add_argument("--config", help="JSON do motor de limpeza (mesmo formato do 'limpar' de PLY)")
    p.add_argument("--faixa-base", type=float, default=2.5, help="altura sobre o prato tratada como prato (mm)")
    p.set_defaults(func=cmd_revo_limpar)
    p = rv.add_parser("fundir", help="funde os quadros de um scan numa nuvem PLY")
    p.add_argument("projeto")
    p.add_argument("--scan", required=True)
    p.add_argument("--saida", required=True)
    p.add_argument("--correcao", help="correcao.json de 'revo giro'")
    p.add_argument("--voxel", type=float, default=0.15)
    p.set_defaults(func=cmd_revo_fundir)
    p = rv.add_parser("giro", help="estima deriva de giro em torno do eixo (nao grava no projeto)")
    p.add_argument("projeto")
    p.add_argument("--scan", required=True)
    p.add_argument("--ref", type=intervalo, required=True, help="quadros de referencia, ex.: 0-69")
    p.add_argument("--alvo", type=intervalo, required=True, help="quadros depois da deriva, ex.: 148-247")
    p.add_argument("--rampa", type=intervalo, required=True, help="trecho cego onde o giro cresce, ex.: 70-147")
    p.add_argument("--ate", type=int, help="ultimo quadro com giro constante (padrao: ultimo do scan)")
    p.add_argument("--centro", type=vetor3, required=True, help="ponto do eixo da peca no mundo: X,Y,Z")
    p.add_argument("--raio", type=float, required=True, help="raio do cilindro (mm)")
    p.add_argument("--altura", type=float, required=True, help="detalhe so acima desta altura sobre a base (mm)")
    p.add_argument("--base-de", help="relatorio.json de 'run.py limpar' do raw_preview.ply, com o plano do prato")
    p.add_argument("--saida", required=True)
    p.set_defaults(func=cmd_revo_giro)
    p = rv.add_parser("aplicar", help="grava COPIA do projeto com a correcao de giro")
    p.add_argument("projeto")
    p.add_argument("--correcao", required=True)
    p.add_argument("--saida")
    p.set_defaults(func=cmd_revo_aplicar)
    return ap


def main():
    a = parser().parse_args()
    try:
        sys.exit(a.func(a) or 0)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        sys.exit(f"erro: {e}")


if __name__ == "__main__":
    main()
