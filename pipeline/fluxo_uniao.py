"""Fluxo completo validado no Projeto 9: do projeto bruto com varios scans ao scan unido limpo, pronto para fundir.

    python pipeline/fluxo_uniao.py <projeto_bruto> --poses poses.json --saida D:/Projects/X_fluxo [--nome X_unido]

`poses.json` = {guid: 4x4 scan->referencia} (Merge Alignment do Revo ou `alinhar_varios.py`). Etapas, cada uma numa
subpasta de --saida (nada e sobrescrito; o projeto de origem nunca e alterado):
  1. atlas/      `atlas_partes segmentar --poses`: regioes de cada scan, medidas e imagens com id.
  2. rotulos.json `atlas_partes sugerir --ilhas --corte-placa-mm`: peca = confirmada por >= 2 outros scans; placa de
                 apoio cortada pelo plano 0 quando ele passa de --placa-area-min; do que fica, so o maior componente.
  3. limpo/      `atlas_partes aplicar`: pixels que nao sao peca saem dos quadros (regioes em duvida FICAM).
  3a. triagem/   `ciclo_uniao --descartar-ruins --so-triagem`: scan inteiro ruim (pouca cobertura) e descartado antes
                 de qualquer consenso, para nao votar.
  3b. visibilidade/ `consenso_visibilidade`: sai o que >= 2 outros scans viram ATRAVES e nenhum confirma (pedaco de
                 suporte ou massinha presente num scan so, que na uniao vira ilha solta no meio da peca).
  4. ciclo/      `ciclo_uniao --unir`: nos scans aceitos sai o trecho fora do consenso que outros scans cobrem; `ciclo/unido` = um scan com todos os quadros alinhados.
Depois: abrir `ciclo/unido/<nome>.revo` no Revo Scan (grupo novo) e fundir. Resultado no P9: aletas nitidas, "90%
limpo" (usuario, 2026-11-15); sobraram migalhas soltas dentro da abertura e uma pelicula sob as aletas.
Onde a LLM/pessoa deve OLHAR antes de confiar: `atlas/<scan>/vistas/*.png` (o plano 0 e mesmo a placa? peca com face
plana grande pede --placa-area-min maior ou --corte-placa-mm 0) e `ciclo/imagens/*_antes_*.png` (vermelho = cortado).
Com --rotulos ARQ as etapas 1-2 nao decidem: vale o arquivo revisado.
  5. conferencia/ `conferir_uniao`: com --fusao-final PLY (a fusao que o Revo gravou para o scan unido; --malha-final
                 para a malha) mede apoio por numero de scans, soltos, zonas fracas e a malha. Com --revo-grupo NOME
                 [mouse] o proprio fluxo importa o scan unido num grupo NOVO do Revo, funde a --revo-ponto mm
                 (--revo-malha tambem gera a malha) e confere o resultado.
Pendencias (codigo 2; o scan unido e gravado mesmo assim e `fluxo.json` lista cada uma): pose suspeita (> 30% em
defeito), pose com vies (celula inteira deslocada mais que --vies-max), regiao em duvida que o consenso nao confirmou
acima de 2% da superficie do scan, conferencia final reprovada. As regioes em duvida da limpeza sao decididas pelo
consenso no ciclo (confirmada por >= 2 scans vira peca), entao a duvida sozinha ja nao torna o resultado parcial.
Placa de video ocupada (Revo ou outro programa 3D aberto) deixa o fluxo muitas vezes mais lento: o fluxo avisa
quando ha menos de --vram-min-gb livres. --tolerancias-auto tira --tol e --acordo do ruido medido entre os scans.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import atlas_partes
import ciclo_uniao
import conferir_uniao
import consenso_visibilidade
import gpu
from comum import configurar_log, salvar_json_atomico

_LOG = configurar_log(__name__)


def tolerancias_do_ruido(p90):
    """(tol, acordo) em mm a partir do p90 do desvio entre scans: no Projeto 9 (p90 ~0,08 mm) da os valores
    aprovados 0,15 e 0,10."""
    return round(min(max(1.9 * p90, 0.10), 0.50), 3), round(min(max(1.25 * p90, 0.06), 0.35), 3)


def fundir_no_revo(revo, nome, grupo, ponto, malha):
    """[mouse] Importa o scan unido num grupo novo do Revo e funde. Devolve (ply da fusao, ply da malha ou None)."""
    import revo_ui

    cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_cli.py")
    subprocess.run([sys.executable, cli, "importar", revo, "--grupo", grupo], check=True)
    cmd = [sys.executable, cli, "fundir", nome, "--grupo", grupo, "--ponto", str(ponto)]
    if malha:
        cmd += ["--malha", "alta", "--grade", str(ponto)]
    subprocess.run(cmd, check=True)
    pasta = os.path.dirname(revo_ui.revox_no_grupo(grupo, nome))

    def mais_novo(padrao):
        arqs = glob.glob(os.path.join(pasta, padrao))
        return max(arqs, key=os.path.getmtime) if arqs else None

    return mais_novo(os.path.join("pointclouds", "Fused*.ply")), mais_novo(
        os.path.join("meshs", "Mesh*.ply")
    ) if malha else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", help="projeto com varios scans da mesma peca (copia de trabalho ou original: so e lido)")
    ap.add_argument("--poses", required=True, help="JSON {guid: 4x4 scan->referencia}")
    ap.add_argument("--saida", required=True, help="pasta nova")
    ap.add_argument("--nome", default="uniao", help="nome do scan unido")
    ap.add_argument("--rotulos", help="rotulos.json ja revisado (pula a sugestao automatica)")
    ap.add_argument("--corte-placa-mm", type=float, default=1.0, help="0 = nao cortar pelo plano de apoio")
    ap.add_argument("--placa-area-min", type=float, default=3000.0, help="mm2 do plano 0 para ser placa de apoio")
    ap.add_argument("--descartar", nargs="+", help="scans que saem inteiros")
    ap.add_argument("--manter-ruins", action="store_true", help="nao descartar scan ruim sozinho")
    ap.add_argument("--sem-visibilidade", action="store_true", help="pula a limpeza do que outro scan viu atraves")
    ap.add_argument("--fusoes", help="pasta com <guid>.ply (fusoes do Revo dos scans LIMPOS) para o ciclo")
    ap.add_argument("--tolerancias-auto", action="store_true", help="--tol e --acordo do ciclo pelo ruido medido")
    ap.add_argument("--fusao-final", help="PLY da fusao do scan unido ja feita no Revo: roda a conferencia final")
    ap.add_argument("--malha-final", help="PLY da malha do scan unido (com --fusao-final)")
    ap.add_argument("--revo-grupo", help="[mouse] importa o scan unido neste grupo NOVO do Revo, funde e confere")
    ap.add_argument("--revo-ponto", type=float, default=0.1, help="mm da fusao no Revo")
    ap.add_argument("--revo-malha", action="store_true", help="com --revo-grupo: gera tambem a malha e confere")
    ap.add_argument("--vram-min-gb", type=float, default=6.0, help="avisa quando ha menos VRAM livre que isso")
    a = ap.parse_args(argv)
    saida = os.path.abspath(a.saida)
    if os.path.exists(saida) and os.listdir(saida):
        sys.exit(f"destino ja existe: {saida}")
    os.makedirs(saida, exist_ok=True)
    t0 = time.time()
    avisos = []
    livre, total = gpu.memoria_vram()
    if livre < a.vram_min_gb * 2**30:
        avisos.append(
            f"placa de video com {livre / 2**30:.1f} de {total / 2**30:.1f} GB livres: feche o Revo Scan e outros "
            "programas 3D; com a memoria cheia o fluxo fica muitas vezes mais lento"
        )
        _LOG.warning("AVISO: %s", avisos[-1])
    atlas, limpo, visto, ciclo = (os.path.join(saida, x) for x in ("atlas", "limpo", "visibilidade", "ciclo"))
    rot = os.path.join(saida, "rotulos.json")
    atlas_partes.main(["segmentar", a.projeto, "--saida", atlas, "--poses", a.poses])
    if a.rotulos:
        with open(a.rotulos, encoding="utf-8") as fh:
            salvar_json_atomico(rot, json.load(fh))
    else:
        cmd = ["sugerir", atlas, "--saida", rot, "--ilhas"]
        if a.corte_placa_mm > 0:
            cmd += ["--corte-placa-mm", str(a.corte_placa_mm), "--placa-area-min", str(a.placa_area_min)]
        atlas_partes.main(cmd)
    atlas_partes.main(["aplicar", a.projeto, atlas, "--rotulos", rot, "--saida", limpo])
    fora = list(a.descartar or [])
    motivos, tol_auto = None, None
    if not a.manter_ruins:  # scan ruim inteiro sai ANTES das etapas de consenso: ele nao pode votar
        tri = os.path.join(saida, "triagem")
        ciclo_uniao.main(
            [
                limpo,
                "--poses",
                a.poses,
                "--saida",
                tri,
                "--descartar-ruins",
                "--so-triagem",
                *(["--descartar", *fora] if fora else []),
            ]
        )
        motivos = os.path.join(tri, "descartados.json")
        with open(motivos, encoding="utf-8") as fh:
            fora = sorted(json.load(fh))
        if a.tolerancias_auto and os.path.exists(os.path.join(tri, "triagem.json")):
            with open(os.path.join(tri, "triagem.json"), encoding="utf-8") as fh:
                ruido = json.load(fh).get("ruido_entre_scans")
            if ruido:
                tol_auto = tolerancias_do_ruido(ruido["p90_mm"])
                _LOG.info("tolerancias pelo ruido (p90 %.3f mm): tol %.3f, acordo %.3f", ruido["p90_mm"], *tol_auto)
    if a.tolerancias_auto and tol_auto is None:
        avisos.append("--tolerancias-auto sem triagem (--manter-ruins): ficaram as tolerancias padrao")
    base = limpo
    if not a.sem_visibilidade:  # suporte/massinha que so um scan tem e os outros viram ATRAVES (ilha no meio da peca)
        with open(a.poses, encoding="utf-8") as fh:
            com_pose = [s for s in json.load(fh) if s not in fora]
        consenso_visibilidade.main([limpo, "--poses", a.poses, "--saida", visto, "--scans", *com_pose])
        base = visto
    antes = limpo if base != limpo else None
    cmd = [base, "--poses", a.poses, "--saida", ciclo, "--unir", a.nome, "--ilhas-uniao"]
    cmd += ["--duvida-atlas", atlas, "--duvida-rotulos", rot]
    if fora:
        cmd += ["--descartar", *fora]
    if motivos:
        cmd += ["--motivos", motivos]
    if tol_auto:
        cmd += ["--tol", str(tol_auto[0]), "--acordo", str(tol_auto[1])]
    if antes:
        cmd += ["--antes", antes]
    if a.fusoes:
        cmd += ["--fusoes", a.fusoes, "--voxel", "0.1"]
    cod_ciclo = ciclo_uniao.main(cmd)
    with open(os.path.join(ciclo, "relatorio.json"), encoding="utf-8") as fh:
        rel = json.load(fh)
    with open(os.path.join(limpo, "atlas_aplicacao.json"), encoding="utf-8") as fh:
        ap_rel = json.load(fh)
    revo = os.path.join(ciclo, "unido", a.nome + ".revo")
    fusao, malha, conf, cod_conf = a.fusao_final, a.malha_final, None, 0
    if a.revo_grupo:  # [mouse] importa em grupo novo, funde e devolve os PLY gravados na biblioteca do Revo
        gpu.liberar_memoria()
        fusao, malha = fundir_no_revo(revo, a.nome, a.revo_grupo, a.revo_ponto, a.revo_malha)
        if not fusao:
            avisos.append("o Revo nao gravou a fusao do scan unido: conferencia final nao feita")
    if fusao:
        cmd = [ciclo, "--fusao", fusao, "--saida", os.path.join(saida, "conferencia")]
        if malha:
            cmd += ["--malha", malha]
        cod_conf = conferir_uniao.main(cmd)
        with open(os.path.join(saida, "conferencia", "conferencia.json"), encoding="utf-8") as fh:
            conf = json.load(fh)
    resumo = dict(
        origem=os.path.abspath(a.projeto),
        unido=rel.get("unido"),
        abrir_no_revo=revo,
        avisos=avisos,
        tolerancias=dict(tol=rel["parametros"]["tol"], acordo=rel["parametros"]["acordo"], automaticas=bool(tol_auto)),
        ruido_entre_scans=rel.get("ruido_entre_scans"),
        descartados=rel.get("descartados"),
        ilhas_de_um_scan=rel.get("ilhas_de_um_scan"),
        pendencias=dict(
            **rel["pendencias"],
            conferencia_final=(conf or {}).get("falhas", []),
        ),
        conferencia_final=None
        if conf is None
        else dict(aprovado=conf["aprovado"], nuvem=conf["nuvem"], malha=conf["malha"], imagens=conf["imagens"]),
        por_scan={
            s: dict(
                limpeza_mantida=ap_rel["scans"].get(s, {}).get("fracao_mantida"),
                duvida=x.get("duvida"),
                vies_de_pose_mm=(x.get("vies_de_pose") or {}).get("vies_abs_max_mm"),
                fora_do_consenso_antes_pct=x["antes"]["acima_tol_pct"],
                fora_do_consenso_depois_pct=(x["depois"] or {}).get("acima_tol_pct"),
                encaixe_p90_mm=(x["depois"] or {}).get("desvio_p90_mm"),
            )
            for s, x in rel["scans"].items()
        },
        parcial=bool(cod_ciclo or cod_conf),
        tempo_s=round(time.time() - t0, 1),
    )
    salvar_json_atomico(os.path.join(saida, "fluxo.json"), resumo)
    pend = {k: v for k, v in resumo["pendencias"].items() if v}
    _LOG.info(
        "fluxo concluido em %.0f s: %s %s%s",
        resumo["tempo_s"],
        "conferido" if conf else "abrir no Revo e fundir:",
        os.path.join(saida, "conferencia") if conf else revo,
        f" | PARCIAL: {pend}" if resumo["parcial"] else "",
    )
    return 2 if resumo["parcial"] else 0


if __name__ == "__main__":
    sys.exit(main())
