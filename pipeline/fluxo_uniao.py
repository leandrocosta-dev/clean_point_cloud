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
--refinar-poses refina as poses depois da limpeza (so superficie de mesma face) e segue com as novas; --escala
multiplica as medidas em mm para peca maior que a do Projeto 9 (padrao: pelo voxel do atlas); com 2 scans nao ha
maioria: nada e cortado por consenso, ficam a medida do par, os soltos e a uniao; --atlas reusa um atlas ja rotulado.
Sem --poses o fluxo alinha sozinho: limpa (com 3 ou mais scans: placa cortada + maior componente, em `pre_*`),
roda `alinhar_varios.py` no projeto limpo, refina as poses e, com 3 ou mais scans, refaz o atlas ja com as poses.
Com menos de 3 scans as regras nao decidem a limpeza (quase nada e confirmado por outro scan): o fluxo grava o atlas,
um rascunho de rotulos (maior componente = peca) e as imagens de `previa` e PARA (codigo 3; --so-rotulos para so essa
parte, codigo 0). Olhar `atlas/<scan>/previa/*_previa.png` (cinza fica, vermelho sai), corrigir `rotulos.json` se
preciso (apoio encostado na peca vem junto no rascunho) e rodar de novo com --continuar na mesma --saida.
--revo-ponto 0 escolhe o ponto pela escala (0,1 mm no Projeto 9); com --revo-malha os componentes soltos da malha
saem (`malha/<nome>_malha_limpa.ply`, nenhum furo e fechado) e e essa malha que e conferida.
Se a etapa do Revo falhar (janela ocupada, nome de grupo recusado), o scan unido ja esta gravado: o fluxo registra a
pendencia `revo` e termina com codigo 2; `--so-revo --revo-grupo NOVO` na mesma --saida refaz so a fusao, a malha e a
conferencia. O Revo corta nomes de grupo em 30 caracteres: nome maior e recusado antes de comecar.
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
import refinar_poses
from comum import configurar_log, salvar_json_atomico

_LOG = configurar_log(__name__)


def tolerancias_do_ruido(p90, k=1.0):
    """(tol, acordo) em mm a partir do p90 do desvio entre scans: no Projeto 9 (p90 ~0,08 mm) da os valores
    aprovados 0,15 e 0,10."""
    return round(min(max(1.9 * p90, 0.10 * k), 0.50 * k), 3), round(min(max(1.25 * p90, 0.06 * k), 0.35 * k), 3)


def escala_do_atlas(atlas):
    """Fator de escala das medidas em mm: voxel mediano do atlas / 0,3 mm (o do Projeto 9), nunca abaixo de 1."""
    vox = []
    for s in os.listdir(atlas):
        arq = os.path.join(atlas, s, "regioes.json")
        if os.path.exists(arq):
            with open(arq, encoding="utf-8") as fh:
                vox.append(float(json.load(fh)["voxel_mm"]))
    return max(1.0, round(sorted(vox)[len(vox) // 2] / 0.3, 2)) if vox else 1.0


def fundir_no_revo(revo, nome, grupo, ponto, malha):
    """[mouse] Importa o scan unido num grupo novo do Revo e funde. Devolve (ply da fusao, ply da malha ou None)."""
    import revo_ui

    cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_cli.py")
    inicio = time.time() - 5.0
    subprocess.run([sys.executable, cli, "importar", revo, "--grupo", grupo], check=True)
    cmd = [sys.executable, cli, "fundir", nome, "--grupo", grupo, "--ponto", str(ponto)]
    if malha:
        cmd += ["--malha", "alta", "--grade", str(ponto)]
    subprocess.run(cmd, check=True)
    pasta = os.path.dirname(revo_ui.revox_no_grupo(grupo, nome))

    def mais_novo(padrao, espera=180.0):
        """O Revo devolve o controle antes de terminar de gravar a malha: espera o arquivo existir e parar de crescer."""
        fim, visto = time.time() + espera, None
        while True:
            arqs = [f for f in glob.glob(os.path.join(pasta, padrao)) if os.path.getmtime(f) >= inicio]
            arq = max(arqs, key=os.path.getmtime) if arqs else None
            tam = os.path.getsize(arq) if arq else None
            if arq and visto == (arq, tam):
                return arq
            if time.time() > fim:
                return None
            visto = (arq, tam)
            time.sleep(3.0)

    return mais_novo(os.path.join("pointclouds", "Fused*.ply")), mais_novo(
        os.path.join("meshs", "Mesh*.ply")
    ) if malha else None


def ponto_pela_escala(k):
    """Ponto da fusao no Revo (mm): 0,1 no Projeto 9; em peca maior acompanha a escala, em passos de 0,05."""
    return 0.1 if k <= 1.5 else round(max(0.1, round(0.075 * k / 0.05) * 0.05), 2)


def limpar_malha(origem, destino, fracao=0.01):
    """Grava em `destino` a malha sem os componentes soltos (menores que `fracao` do maior). Nao fecha furo."""
    from malha_meshlib import metricas, reparar
    from meshlib import mrmeshpy as mm

    m = mm.loadMesh(origem)
    antes = metricas(m)
    reparar(m, min_faces=max(1, int(fracao * antes["maior_componente"])))
    os.makedirs(os.path.dirname(destino), exist_ok=True)
    mm.saveMesh(m, destino)
    return dict(origem=origem, limpa=destino, antes=antes, depois=metricas(m))


def alinhar(projeto, pasta, scans):
    """Roda `alinhar_varios.py` e devolve (poses {guid: 4x4}, scans sem ligacao)."""
    cli = os.path.join(os.path.dirname(os.path.abspath(__file__)), "alinhar_varios.py")
    subprocess.run(
        [sys.executable, "-u", cli, projeto, "--saida", pasta, *(["--scans", *scans] if scans else [])], check=True
    )
    with open(os.path.join(pasta, "alinhamento.json"), encoding="utf-8") as fh:
        d = json.load(fh)
    return d["poses"], list(d.get("sem_ligacao") or [])


GRUPO_MAX = 30  # o Revo Scan corta o nome do grupo neste tamanho e a automacao deixa de acha-lo


def revo_e_conferencia(a, saida, ciclo, rel, k, avisos):
    """Etapa 5: (opcional, [mouse]) funde o scan unido no Revo; limpa a malha; confere fusao e malha.
    Devolve dict(conf, cod, malha, falha_revo)."""
    revo = os.path.join(ciclo, "unido", a.nome + ".revo")
    fusao, malha, conf, cod, falha = a.fusao_final, a.malha_final, None, 0, None
    if a.revo_grupo:  # importa em grupo novo, funde e devolve os PLY gravados na biblioteca do Revo
        gpu.liberar_memoria()
        ponto = a.revo_ponto or ponto_pela_escala(k)
        try:
            fusao, malha = fundir_no_revo(revo, a.nome, a.revo_grupo, ponto, a.revo_malha)
        except (subprocess.CalledProcessError, OSError, LookupError, RuntimeError) as e:
            falha = f"a etapa do Revo falhou ({type(e).__name__}); o scan unido esta em {revo}; refazer com --so-revo"
            _LOG.warning("AVISO: %s", falha)
            fusao = malha = None
        if not falha and not fusao:
            falha = "o Revo nao gravou a fusao do scan unido: conferencia final nao feita"
        if not falha and a.revo_malha and not malha:
            avisos.append("o Revo nao gravou a malha do scan unido")
    malha_rel = None
    if malha:  # componentes soltos da malha saem (copia nova; a do Revo fica como esta)
        malha_rel = limpar_malha(malha, os.path.join(saida, "malha", a.nome + "_malha_limpa.ply"))
        malha = malha_rel["limpa"]
        _LOG.info("malha: %s -> %s", malha_rel["antes"], malha_rel["depois"])
    if fusao:
        cmd = [ciclo, "--fusao", fusao, "--saida", os.path.join(saida, "conferencia"), "--raio", f"{0.3 * k:.3f}"]
        quadros = list(((rel.get("unido") or {}).get("por_scan") or {}).values())
        if quadros and min(quadros) < 150:  # scan esparso: a fusao propria com 4 observacoes tem buracos
            cmd += ["--min-obs", "1"]
        if malha:
            cmd += ["--malha", malha]
        cod = conferir_uniao.main(cmd)
        with open(os.path.join(saida, "conferencia", "conferencia.json"), encoding="utf-8") as fh:
            conf = json.load(fh)
    return dict(conf=conf, cod=cod, malha=malha_rel, falha_revo=falha)


def so_revo(a, saida):
    """Refaz so a etapa 5 numa --saida que ja tem o scan unido (o Revo falhou ou a fusao ficou para depois)."""
    ciclo = os.path.join(saida, "ciclo")
    if not os.path.exists(os.path.join(ciclo, "unido", a.nome + ".revo")):
        sys.exit(f"--so-revo: nao achei o scan unido {a.nome}.revo em {os.path.join(ciclo, 'unido')}")
    if os.path.exists(os.path.join(saida, "conferencia")):
        sys.exit(f"--so-revo: {saida} ja tem a conferencia final")
    if not (a.revo_grupo or a.fusao_final):
        sys.exit("--so-revo precisa de --revo-grupo (grupo NOVO) ou de --fusao-final")
    t0 = time.time()
    with open(os.path.join(ciclo, "relatorio.json"), encoding="utf-8") as fh:
        rel = json.load(fh)
    arq = os.path.join(saida, "fluxo.json")
    resumo = {}
    if os.path.exists(arq):
        with open(arq, encoding="utf-8") as fh:
            resumo = json.load(fh)
    k = a.escala or resumo.get("escala") or escala_do_atlas(os.path.join(saida, "atlas"))
    avisos = list(resumo.get("avisos") or [])
    r = revo_e_conferencia(a, saida, ciclo, rel, k, avisos)
    conf = r["conf"]
    pend = dict(resumo.get("pendencias") or rel["pendencias"])
    pend["conferencia_final"] = (conf or {}).get("falhas", [])
    pend["revo"] = [r["falha_revo"]] if r["falha_revo"] else []
    resumo.update(
        unido=rel.get("unido"),
        abrir_no_revo=os.path.join(ciclo, "unido", a.nome + ".revo"),
        escala=round(k, 3),
        avisos=avisos,
        malha=r["malha"],
        pendencias=pend,
        conferencia_final=None
        if conf is None
        else dict(aprovado=conf["aprovado"], nuvem=conf["nuvem"], malha=conf["malha"], imagens=conf["imagens"]),
        parcial=any(bool(v) for v in pend.values()),
        tempo_revo_s=round(time.time() - t0, 1),
    )
    salvar_json_atomico(arq, resumo)
    _LOG.info(
        "etapa do Revo concluida em %.0f s: %s%s",
        time.time() - t0,
        os.path.join(saida, "conferencia") if conf else "sem conferencia",
        f" | PARCIAL: { {n: v for n, v in pend.items() if v} }" if resumo["parcial"] else "",
    )
    return 2 if resumo["parcial"] else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", help="projeto com varios scans da mesma peca (copia de trabalho ou original: so e lido)")
    ap.add_argument("--poses", help="JSON {guid: 4x4 scan->referencia}; sem ele o fluxo alinha com alinhar_varios.py")
    ap.add_argument("--scans", nargs="+", help="padrao: todos os scans do projeto (com pose, quando --poses e dado)")
    ap.add_argument("--so-rotulos", action="store_true", help="so atlas, rascunho de rotulos e imagens de previa")
    ap.add_argument(
        "--continuar",
        action="store_true",
        help="retoma uma --saida que parou na revisao dos rotulos (usa o atlas e o rotulos.json que estao nela)",
    )
    ap.add_argument("--saida", required=True, help="pasta nova")
    ap.add_argument("--nome", default="uniao", help="nome do scan unido")
    ap.add_argument("--rotulos", help="rotulos.json ja revisado (pula a sugestao automatica)")
    ap.add_argument("--corte-placa-mm", type=float, default=1.0, help="0 = nao cortar pelo plano de apoio")
    ap.add_argument("--placa-area-min", type=float, default=3000.0, help="mm2 do plano 0 para ser placa de apoio")
    ap.add_argument("--descartar", nargs="+", help="scans que saem inteiros")
    ap.add_argument("--manter-ruins", action="store_true", help="nao descartar scan ruim sozinho")
    ap.add_argument("--sem-visibilidade", action="store_true", help="pula a limpeza do que outro scan viu atraves")
    ap.add_argument("--fusoes", help="pasta com <guid>.ply (fusoes do Revo dos scans LIMPOS) para o ciclo")
    ap.add_argument("--atlas", help="atlas ja segmentado (os ids das regioes de --rotulos sao os dele): pula a etapa 1")
    ap.add_argument(
        "--escala",
        type=float,
        default=0.0,
        help="multiplica as medidas em mm do ciclo, da visibilidade e da conferencia (1 = peca de ~45 mm do Projeto 9); "
        "0 = voxel do atlas / 0,3 mm",
    )
    ap.add_argument(
        "--refinar-poses",
        action="store_true",
        help="depois da limpeza, refina as poses por superficie de mesma face (`refinar_poses.py`) e usa as novas",
    )
    ap.add_argument("--tolerancias-auto", action="store_true", help="--tol e --acordo do ciclo pelo ruido medido")
    ap.add_argument("--fusao-final", help="PLY da fusao do scan unido ja feita no Revo: roda a conferencia final")
    ap.add_argument("--malha-final", help="PLY da malha do scan unido (com --fusao-final)")
    ap.add_argument("--revo-grupo", help="[mouse] importa o scan unido neste grupo NOVO do Revo, funde e confere")
    ap.add_argument("--revo-ponto", type=float, default=0.0, help="mm da fusao no Revo; 0 = pela escala (0,1 no P9)")
    ap.add_argument("--revo-malha", action="store_true", help="com --revo-grupo: gera tambem a malha e confere")
    ap.add_argument("--so-revo", action="store_true", help="so a etapa do Revo numa --saida que ja tem o scan unido")
    ap.add_argument("--vram-min-gb", type=float, default=6.0, help="avisa quando ha menos VRAM livre que isso")
    a = ap.parse_args(argv)
    saida = os.path.abspath(a.saida)
    if a.revo_grupo and len(a.revo_grupo) > GRUPO_MAX:
        sys.exit(f"--revo-grupo com {len(a.revo_grupo)} caracteres: o Revo corta em {GRUPO_MAX}; use um nome menor")
    if a.so_revo:
        return so_revo(a, saida)
    if a.continuar:
        if not (os.path.isdir(os.path.join(saida, "atlas")) and os.path.exists(os.path.join(saida, "rotulos.json"))):
            sys.exit(f"--continuar: {saida} nao tem atlas/ e rotulos.json de uma rodada que parou na revisao")
        if os.path.exists(os.path.join(saida, "limpo")):
            sys.exit(f"--continuar: {saida} ja passou da limpeza; use uma --saida nova")
    elif os.path.exists(saida) and os.listdir(saida):
        sys.exit(f"destino ja existe: {saida}")
    os.makedirs(saida, exist_ok=True)
    _, _, todos = atlas_partes.scans_do_projeto(os.path.abspath(a.projeto))
    scans = [s for s in todos if not a.scans or s in a.scans]
    if a.poses:
        with open(a.poses, encoding="utf-8") as fh:
            poses_dadas = json.load(fh)
        scans = [s for s in scans if s in poses_dadas]
        if a.scans:  # as etapas seguintes usam todos os scans do arquivo de poses: gravar so os escolhidos
            a.poses = os.path.join(saida, "poses_usadas.json")
            salvar_json_atomico(a.poses, {s: poses_dadas[s] for s in scans})
    if len(scans) < 2:
        sys.exit(f"preciso de pelo menos 2 scans (achei {len(scans)}); para 1 scan use a receita A")
    poucos = len(scans) < 3  # sem maioria: as regras nao decidem a limpeza, alguem tem de olhar
    sel = ["--scans", *scans]
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
    pre = None
    if a.atlas:
        atlas = os.path.abspath(a.atlas)
    elif a.continuar:
        _LOG.info("continuando: atlas e rotulos de %s", saida)
    elif not a.poses and not poucos and not a.rotulos:
        # 3 ou mais scans sem poses: limpeza previa (placa + maior componente) so para poder alinhar; a limpeza de
        # verdade vem depois, com a confirmacao entre scans que as poses permitem
        pre = (
            os.path.join(saida, "pre_atlas"),
            os.path.join(saida, "pre_rotulos.json"),
            os.path.join(saida, "pre_limpo"),
        )
        atlas_partes.main(["segmentar", a.projeto, "--saida", pre[0], *sel])
        cmd = ["sugerir", pre[0], "--saida", pre[1], "--ilhas"]
        if a.corte_placa_mm > 0:
            cmd += ["--corte-placa-mm", str(a.corte_placa_mm), "--placa-area-min", str(a.placa_area_min)]
        atlas_partes.main(cmd)
        atlas_partes.main(["aplicar", a.projeto, pre[0], "--rotulos", pre[1], "--saida", pre[2]])
    else:
        atlas_partes.main(["segmentar", a.projeto, "--saida", atlas, *sel, *(["--poses", a.poses] if a.poses else [])])
    sem_ligacao, alinhado = [], False
    if pre:
        k0 = a.escala or escala_do_atlas(pre[0])
        poses_novas, sem_ligacao = alinhar(pre[2], os.path.join(saida, "alinhar"), scans)
        bruta = os.path.join(saida, "poses_alinhar.json")
        salvar_json_atomico(bruta, {s: T for s, T in poses_novas.items() if s not in sem_ligacao})
        a.poses = os.path.join(saida, "poses_refinadas.json")
        refinar_poses.main([pre[2], "--poses", bruta, "--saida", a.poses, "--escala", str(k0)])
        with open(a.poses + ".relatorio.json", encoding="utf-8") as fh:
            poses_refino = json.load(fh)
        alinhado = True
        scans = [s for s in scans if s not in sem_ligacao]
        sel = ["--scans", *scans]
        atlas_partes.main(["segmentar", a.projeto, "--saida", atlas, *sel, "--poses", a.poses])
    else:
        poses_refino = None
    k = a.escala or escala_do_atlas(atlas)
    if k != 1.0:
        _LOG.info("escala %.2f: medidas do ciclo, da visibilidade e da conferencia multiplicadas", k)
    medidas = [
        "--voxel", f"{0.15 * k:.3f}", "--raio", f"{0.4 * k:.3f}", "--alcance-normal", f"{1.2 * k:.3f}",
        "--tol", f"{0.15 * k:.3f}", "--acordo", f"{0.10 * k:.3f}", "--area-min", f"{0.5 * k * k:.3f}",
        "--vies-max", f"{0.12 * k:.3f}",
    ]  # fmt: skip
    revisar = False
    if a.rotulos:
        with open(a.rotulos, encoding="utf-8") as fh:
            salvar_json_atomico(rot, json.load(fh))
    elif a.continuar:
        pass
    elif poucos:
        atlas_partes.main(["sugerir", atlas, "--saida", rot, "--por-componente"])
        revisar = True
    else:
        cmd = ["sugerir", atlas, "--saida", rot, "--ilhas"]
        if a.corte_placa_mm > 0:
            cmd += ["--corte-placa-mm", str(a.corte_placa_mm), "--placa-area-min", str(a.placa_area_min)]
        atlas_partes.main(cmd)
    if revisar or a.so_rotulos:
        atlas_partes.main(["previa", atlas, "--rotulos", rot])
        _LOG.info(
            "PARADA PARA REVISAO dos rotulos (%d scans: sem maioria as regras nao decidem). Olhar "
            "%s\\<scan>\\previa\\*_previa.png (cinza fica, vermelho sai) e *_resultado.png; corrigir %s se preciso "
            '(ex.: "regioes": {"23": "suporte"} para um apoio que veio junto; ids em vistas\\*_regioes.png); '
            "depois rodar de novo com --continuar e a mesma --saida.",
            len(scans),
            atlas,
            rot,
        )
        return 0 if a.so_rotulos else 3
    atlas_partes.main(["aplicar", a.projeto, atlas, "--rotulos", rot, "--saida", limpo])
    fora = list(a.descartar or [])
    motivos, tol_auto = None, None
    if not a.poses:  # menos de 3 scans (ou rotulos dados) sem poses: alinhar o projeto ja limpo
        poses_novas, sem_ligacao = alinhar(limpo, os.path.join(saida, "alinhar"), scans)
        a.poses = os.path.join(saida, "poses_alinhar.json")
        salvar_json_atomico(a.poses, {s: T for s, T in poses_novas.items() if s not in sem_ligacao})
        alinhado, a.refinar_poses = True, True
    if sem_ligacao:
        avisos.append(f"scans sem ligacao confiavel no alinhamento, fora da uniao: {sem_ligacao}")
        _LOG.warning("AVISO: %s", avisos[-1])
    if a.refinar_poses and not pre:  # o alinhador pode ter puxado frente contra verso em parede fina
        nova = os.path.join(saida, "poses_refinadas.json")
        refinar_poses.main([limpo, "--poses", a.poses, "--saida", nova, "--escala", str(k)])
        with open(nova + ".relatorio.json", encoding="utf-8") as fh:
            poses_refino = json.load(fh)
        a.poses = nova
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
                *medidas,
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
                tol_auto = tolerancias_do_ruido(ruido["p90_mm"], k)
                _LOG.info("tolerancias pelo ruido (p90 %.3f mm): tol %.3f, acordo %.3f", ruido["p90_mm"], *tol_auto)
    if a.tolerancias_auto and tol_auto is None:
        avisos.append("--tolerancias-auto sem triagem (--manter-ruins): ficaram as tolerancias padrao")
    base = limpo
    if not a.sem_visibilidade:  # suporte/massinha que so um scan tem e os outros viram ATRAVES (ilha no meio da peca)
        with open(a.poses, encoding="utf-8") as fh:
            com_pose = [s for s in json.load(fh) if s not in fora]
        consenso_visibilidade.main(
            [limpo, "--poses", a.poses, "--saida", visto, "--voxel", f"{0.5 * k:.3f}", "--tol", f"{0.5 * k:.3f}"]
            + ["--scans", *com_pose]
        )
        base = visto
    antes = limpo if base != limpo else None
    cmd = [base, "--poses", a.poses, "--saida", ciclo, "--unir", a.nome, "--ilhas-uniao"]
    cmd += ["--duvida-atlas", atlas, "--duvida-rotulos", rot, *medidas]
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
    final = revo_e_conferencia(a, saida, ciclo, rel, k, avisos)
    conf, cod_conf, malha_rel = final["conf"], final["cod"], final["malha"]
    resumo = dict(
        origem=os.path.abspath(a.projeto),
        unido=rel.get("unido"),
        abrir_no_revo=revo,
        avisos=avisos,
        escala=round(k, 3),
        poses=os.path.abspath(a.poses),
        refino_de_poses=poses_refino,
        alinhado_pelo_fluxo=alinhado,
        malha=malha_rel,
        tolerancias=dict(tol=rel["parametros"]["tol"], acordo=rel["parametros"]["acordo"], automaticas=bool(tol_auto)),
        ruido_entre_scans=rel.get("ruido_entre_scans"),
        descartados=rel.get("descartados"),
        ilhas_de_um_scan=rel.get("ilhas_de_um_scan"),
        pendencias=dict(
            **rel["pendencias"],
            conferencia_final=(conf or {}).get("falhas", []),
            revo=[final["falha_revo"]] if final["falha_revo"] else [],
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
        parcial=bool(cod_ciclo or cod_conf or final["falha_revo"]),
        tempo_s=round(time.time() - t0, 1),
    )
    salvar_json_atomico(os.path.join(saida, "fluxo.json"), resumo)
    pend = {nome: v for nome, v in resumo["pendencias"].items() if v}
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
