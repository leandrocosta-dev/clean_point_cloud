"""Modelos de fluxo (secao 2 do GUIA_PIPELINE.md) montados com o contexto atual."""

import os
import time

from gui.catalog import PIPE
from gui.core import substituir


def _job(app, nome, script, *args):
    ctx = app.estado.ctx
    argv = [app.estado.env["python"], "-u", os.path.join(PIPE, script)] + [substituir(str(a), ctx) for a in args]
    return {"nome": nome, "argv": argv}


def _triagem(app):
    """Job 0: mede o projeto e grava perfil.json com a receita recomendada (nao altera nada)."""
    return _job(app, "0 · Triagem do projeto", "triagem.py", "{PROJETO}", "--saida", "{TRABALHO}\\{NOME}_perfil.json")


def receita_a(app):
    """Um scan: limpeza + Revo (fusao, malha, exportacao)."""
    return [
        _triagem(app),
        _job(
            app,
            "A · Pipeline completo (1 scan)",
            "rodar_tudo.py",
            "{PROJETO}",
            "--saida-ply",
            "{SAIDA}\\{NOME}",
            "--periferia-mm",
            5,
            "--periferia-banda",
            12,
            "--periferia-alargue",
            10,
            "--ponto",
            0.1,
            "--malha",
            "alta",
            "--grade",
            0.1,
        ),
    ]


def receita_b(app):
    """Duas partes da mesma peca (virada de cima/baixo)."""
    sc = app.estado.ctx.get("scans", [])
    if len(sc) < 2:
        raise ValueError("Selecione exatamente os 2 scans (partes A e B) na página Projeto.")
    return [
        _triagem(app),
        _job(
            app,
            "B · Duas partes (uniao + Revo)",
            "rodar_tudo.py",
            "{PROJETO}",
            "--saida-ply",
            "{SAIDA}\\{NOME}",
            "--unir",
            sc[0],
            sc[1],
            "--lados",
            "opostos",
            "--ponto",
            0.1,
            "--malha",
            "alta",
            "--grade",
            0.1,
        ),
    ]


def receita_c(app):
    """Muitos scans da mesma peca: ate a otimizacao conjunta e a limpeza por consenso (passos 1-4 do guia)."""
    sc = app.estado.ctx.get("scans", [])
    esc = ["--scans", *sc] if sc else []
    limpo, alinh, seg = "{TRABALHO}\\{NOME}_limpo", "{TRABALHO}\\{NOME}_alinh", "{TRABALHO}\\{NOME}_seg"
    return [
        _triagem(app),
        _job(app, "C1 · Limpar frames", "limpar_raw.py", "{PROJETO}", "--saida", limpo, "--sem-ply-antigos", *esc),
        _job(app, "C2 · Alinhamento inicial (âncoras)", "alinhar_varios.py", limpo, "--saida", alinh, *esc),
        _job(
            app,
            "C3 · Segmentos e centróides",
            "alinhar_segmentos.py",
            limpo,
            "--saida",
            seg,
            "--poses",
            alinh + "\\poses_iniciais.json",
            "--quadros",
            15,
            "--rodadas",
            1,
            *esc,
        ),
        _job(
            app,
            "C4a · Otimização conjunta (6→1,2 mm)",
            "alinhar_direto.py",
            seg,
            "--poses",
            seg + "\\segmentos.json",
            "--saida",
            seg + "\\direto1.json",
            "--tol-ini",
            6,
            "--tol-fim",
            1.2,
            "--iteracoes",
            14,
        ),
        _job(
            app,
            "C4b · Otimização conjunta (3→1 mm)",
            "alinhar_direto.py",
            seg,
            "--poses",
            seg + "\\direto1.json",
            "--saida",
            seg + "\\direto2.json",
            "--tol-ini",
            3,
            "--tol-fim",
            1,
            "--iteracoes",
            24,
            "--sigma-odo",
            1.0,
            "--amostra",
            20000,
        ),
        _job(
            app,
            "C4c · Otimização conjunta (2→0,8 mm)",
            "alinhar_direto.py",
            seg,
            "--poses",
            seg + "\\direto2.json",
            "--saida",
            seg + "\\direto3.json",
            "--tol-ini",
            2,
            "--tol-fim",
            0.8,
            "--iteracoes",
            30,
            "--k-por-no",
            6,
        ),
        _job(
            app,
            "C5 · Limpeza por consenso",
            "limpar_por_consenso.py",
            limpo,
            seg,
            seg + "\\direto3.json",
            "--saida",
            "{TRABALHO}\\{NOME}_limpo2",
        ),
    ]


def receita_d(app):
    """Peca pequena (~50 mm) em varias poses sobre placa de marcadores (Projetos 8 e 9): limpeza dos frames.
    Depois: alinhar (Merge Alignment do Revo ou alinhar_varios --escala auto), consenso e fundir_nitido."""
    sc = app.estado.ctx.get("scans", [])
    esc = ["--scans", *sc] if sc else []
    limpo = "{TRABALHO}\\{NOME}_limpo"
    return [
        _triagem(app),
        _job(
            app,
            "D1 · Limpar frames (peça pequena)",
            "limpar_raw.py",
            "{PROJETO}",
            "--saida",
            limpo,
            "--ext-max-mm",
            100,
            "--so-peca",
            "--min-mantido",
            0.02,
            "--sem-ply-antigos",
            *esc,
        ),
        _job(
            app,
            "D2 · Cortar a placa a 1 mm",
            "limpar_por_placa.py",
            limpo,
            "--saida",
            "{TRABALHO}\\{NOME}_placa",
            "--altura-min",
            1,
            "--voxel",
            1,
            "--min-pts-voxel",
            3,
            "--plano",
            "extenso",
        ),
    ]


def _scans_do_projeto(projeto):
    """Scans (nao os quadros avulsos `Frame*`) com frames.dataset na pasta do projeto."""
    data = os.path.join(projeto, "data")
    if not os.path.isdir(data):
        return []
    return sorted(
        d
        for d in os.listdir(data)
        if not d.startswith("Frame") and os.path.exists(os.path.join(data, d, "cache", "frames.dataset"))
    )


def receita_e(app):
    r"""Varios scans da mesma peca ate a fusao e a malha conferidas (fluxo_uniao.py; aprovada no Projeto 9).
    Poses: usa {TRABALHO}\{NOME}_poses.json ({guid: 4x4}, Merge Alignment do Revo) se existir; sem ele o fluxo alinha
    sozinho (alinhar_varios + refino por mesma face). Com 3 ou mais scans roda direto. Com 2 scans as regras nao
    decidem a limpeza: o 1o clique gera o atlas, o rascunho de rotulos e as imagens de previa e para; depois de
    olhar {TRABALHO}\{NOME}_fluxotlas\<scan>\previa e corrigir rotulos.json se preciso, clicar de novo na receita
    continua ate o Revo. A etapa final usa o mouse: importa num grupo NOVO do Revo, funde, gera a malha e confere."""
    ctx = app.estado.ctx
    projeto = ctx.get("projeto") or ""
    if not projeto:
        raise ValueError("Escolha o projeto na página Projeto.")
    escolhidos = list(ctx.get("scans") or [])
    scans = escolhidos or _scans_do_projeto(projeto)
    if len(scans) < 2:
        raise ValueError("A receita E precisa de pelo menos 2 scans da mesma peça. Para 1 scan use a receita A.")
    saida = substituir(r"{TRABALHO}\{NOME}_fluxo", ctx)
    poses = substituir(r"{TRABALHO}\{NOME}_poses.json", ctx)
    base = ["{PROJETO}", "--saida", saida, "--nome", "{NOME}_unido"]
    if escolhidos:
        base += ["--scans", *escolhidos]
    if os.path.exists(poses):
        base += ["--poses", poses]
    nome = os.path.basename(projeto.rstrip("\\/"))
    # o Revo corta o nome do grupo em 30 caracteres: fim do nome do projeto + data e hora (grupo sempre novo)
    revo = ["--revo-grupo", nome[-15:] + "_" + time.strftime("%m%d_%H%M"), "--revo-malha"]
    parada = os.path.exists(os.path.join(saida, "rotulos.json")) and not os.path.exists(os.path.join(saida, "limpo"))
    if parada:
        return [
            _job(
                app,
                "E2 · Continuar: limpar, alinhar, unir, fundir no Revo e conferir",
                "fluxo_uniao.py",
                *base,
                "--continuar",
                *revo,
            )
        ]
    unido = os.path.exists(os.path.join(saida, "ciclo", "unido", nome + "_unido.revo"))
    if unido and not os.path.exists(os.path.join(saida, "conferencia")):
        return [
            _job(
                app,
                "E3 · Só a etapa do Revo: fundir o scan unido, gerar a malha e conferir",
                "fluxo_uniao.py",
                "{PROJETO}",
                "--saida",
                saida,
                "--nome",
                "{NOME}_unido",
                "--so-revo",
                *revo,
            )
        ]
    if os.path.isdir(saida) and os.listdir(saida):
        raise ValueError(
            f"A pasta {saida} já tem um resultado desta receita. Renomeie ou apague essa pasta para rodar de novo."
        )
    if len(scans) < 3:
        return [
            _job(
                app,
                "E1 · Atlas e rascunho de rótulos (PARA para revisão; depois clicar de novo na receita E)",
                "fluxo_uniao.py",
                *base,
                "--so-rotulos",
            )
        ]
    return [_job(app, "E · Fluxo completo: limpar, unir, fundir no Revo e conferir", "fluxo_uniao.py", *base, *revo)]


RECEITAS = [
    ("A · Um scan (limpeza + Revo)", receita_a),
    ("B · Duas partes da mesma peça", receita_b),
    ("C · Muitos scans (passos 1 a 5)", receita_c),
    ("D · Peça pequena em várias poses", receita_d),
    ("E · Vários scans: scan unido, fusão e malha conferidas", receita_e),
]
