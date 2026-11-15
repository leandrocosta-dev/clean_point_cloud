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
Com --rotulos ARQ as etapas 1-2 nao decidem: vale o arquivo revisado. Codigo 2 = ha pendencias (duvida ou pose
suspeita); o scan unido e gravado mesmo assim e o relatorio `fluxo.json` lista o que ficou pendente.
"""

import argparse
import json
import os
import sys
import time

import atlas_partes
import ciclo_uniao
import consenso_visibilidade
from comum import configurar_log, salvar_json_atomico

_LOG = configurar_log(__name__)


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
    a = ap.parse_args(argv)
    saida = os.path.abspath(a.saida)
    if os.path.exists(saida) and os.listdir(saida):
        sys.exit(f"destino ja existe: {saida}")
    os.makedirs(saida, exist_ok=True)
    t0 = time.time()
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
    cod_atlas = atlas_partes.main(["aplicar", a.projeto, atlas, "--rotulos", rot, "--saida", limpo])
    fora = list(a.descartar or [])
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
        with open(os.path.join(tri, "descartados.json"), encoding="utf-8") as fh:
            fora = sorted(json.load(fh))
    base = limpo
    if not a.sem_visibilidade:  # suporte/massinha que so um scan tem e os outros viram ATRAVES (ilha no meio da peca)
        with open(a.poses, encoding="utf-8") as fh:
            com_pose = [s for s in json.load(fh) if s not in fora]
        consenso_visibilidade.main([limpo, "--poses", a.poses, "--saida", visto, "--scans", *com_pose])
        base = visto
    antes = limpo if base != limpo else None
    cmd = [base, "--poses", a.poses, "--saida", ciclo, "--unir", a.nome, "--ilhas-uniao"]
    if fora:
        cmd += ["--descartar", *fora]
    if antes:
        cmd += ["--antes", antes]
    if a.fusoes:
        cmd += ["--fusoes", a.fusoes, "--voxel", "0.1"]
    cod_ciclo = ciclo_uniao.main(cmd)
    with open(os.path.join(ciclo, "relatorio.json"), encoding="utf-8") as fh:
        rel = json.load(fh)
    with open(os.path.join(limpo, "atlas_aplicacao.json"), encoding="utf-8") as fh:
        ap_rel = json.load(fh)
    duvida = {
        s: x["classes"]["duvida"]["voxels"] for s, x in ap_rel["scans"].items() if x["classes"]["duvida"]["voxels"]
    }
    resumo = dict(
        origem=os.path.abspath(a.projeto),
        unido=rel.get("unido"),
        abrir_no_revo=os.path.join(ciclo, "unido", a.nome + ".revo"),
        descartados=rel.get("descartados"),
        ilhas_de_um_scan=rel.get("ilhas_de_um_scan"),
        pendencias=dict(regioes_em_duvida_voxels=duvida, pose_suspeita=rel["pendencias"]["pose_suspeita"]),
        por_scan={
            s: dict(
                limpeza_mantida=ap_rel["scans"].get(s, {}).get("fracao_mantida"),
                fora_do_consenso_antes_pct=x["antes"]["acima_tol_pct"],
                fora_do_consenso_depois_pct=(x["depois"] or {}).get("acima_tol_pct"),
                encaixe_p90_mm=(x["depois"] or {}).get("desvio_p90_mm"),
            )
            for s, x in rel["scans"].items()
        },
        parcial=bool(cod_atlas or cod_ciclo),
        tempo_s=round(time.time() - t0, 1),
    )
    salvar_json_atomico(os.path.join(saida, "fluxo.json"), resumo)
    _LOG.info(
        "fluxo concluido em %.0f s: abrir %s%s",
        resumo["tempo_s"],
        resumo["abrir_no_revo"],
        " | PARCIAL" if resumo["parcial"] else "",
    )
    return 2 if resumo["parcial"] else 0


if __name__ == "__main__":
    sys.exit(main())
