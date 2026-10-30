"""Modelos de fluxo (secao 2 do GUIA_PIPELINE.md) montados com o contexto atual."""
import os

from catalog import PIPE
from core import substituir


def _job(app, nome, script, *args):
    ctx = app.estado.ctx
    argv = [app.estado.env["python"], "-u", os.path.join(PIPE, script)] + [substituir(str(a), ctx) for a in args]
    return {"nome": nome, "argv": argv}


def receita_a(app):
    """Um scan: limpeza + Revo (fusao, malha, exportacao)."""
    return [_job(app, "A · Pipeline completo (1 scan)", "rodar_tudo.py", "{PROJETO}", "--saida-ply", "{SAIDA}\\{NOME}",
                 "--periferia-mm", 5, "--periferia-banda", 12, "--periferia-alargue", 10, "--ponto", 0.1, "--malha", "alta",
                 "--grade", 0.1, "--furos")]


def receita_b(app):
    """Duas partes da mesma peca (virada de cima/baixo)."""
    sc = app.estado.ctx.get("scans", [])
    if len(sc) < 2:
        raise ValueError("Selecione exatamente os 2 scans (partes A e B) na página Projeto.")
    return [_job(app, "B · Duas partes (uniao + Revo)", "rodar_tudo.py", "{PROJETO}", "--saida-ply", "{SAIDA}\\{NOME}",
                 "--unir", sc[0], sc[1], "--lados", "opostos", "--ponto", 0.1, "--malha", "alta", "--grade", 0.1, "--furos")]


def receita_c(app):
    """Muitos scans da mesma peca: ate a otimizacao conjunta e a limpeza por consenso (passos 1-4 do guia)."""
    sc = app.estado.ctx.get("scans", [])
    esc = ["--scans", *sc] if sc else []
    limpo, alinh, seg = "{TRABALHO}\\{NOME}_limpo", "{TRABALHO}\\{NOME}_alinh", "{TRABALHO}\\{NOME}_seg"
    return [
        _job(app, "C1 · Limpar frames", "limpar_raw.py", "{PROJETO}", "--saida", limpo, "--sem-ply-antigos", *esc),
        _job(app, "C2 · Alinhamento inicial (âncoras)", "alinhar_varios.py", limpo, "--saida", alinh, *esc),
        _job(app, "C3 · Segmentos e centróides", "alinhar_segmentos.py", limpo, "--saida", seg, "--poses", alinh + "\\poses_iniciais.json",
             "--quadros", 15, "--rodadas", 1, *esc),
        _job(app, "C4a · Otimização conjunta (6→1,2 mm)", "alinhar_direto.py", seg, "--poses", seg + "\\segmentos.json",
             "--saida", seg + "\\direto1.json", "--tol-ini", 6, "--tol-fim", 1.2, "--iteracoes", 14),
        _job(app, "C4b · Otimização conjunta (3→1 mm)", "alinhar_direto.py", seg, "--poses", seg + "\\direto1.json",
             "--saida", seg + "\\direto2.json", "--tol-ini", 3, "--tol-fim", 1, "--iteracoes", 24, "--sigma-odo", 1.0, "--amostra", 20000),
        _job(app, "C4c · Otimização conjunta (2→0,8 mm)", "alinhar_direto.py", seg, "--poses", seg + "\\direto2.json",
             "--saida", seg + "\\direto3.json", "--tol-ini", 2, "--tol-fim", 0.8, "--iteracoes", 30, "--k-por-no", 6),
        _job(app, "C5 · Limpeza por consenso", "limpar_por_consenso.py", limpo, seg, seg + "\\direto3.json",
             "--saida", "{TRABALHO}\\{NOME}_limpo2"),
    ]


RECEITAS = [
    ("A · Um scan (limpeza + Revo)", receita_a),
    ("B · Duas partes da mesma peça", receita_b),
    ("C · Muitos scans (passos 1 a 5)", receita_c),
]
