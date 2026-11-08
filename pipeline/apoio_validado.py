"""Le planos de apoio medidos antes da limpeza, com IDs originais ou remapeados.

O JSON pode ser a saida de ``avaliar_corte_apoio.py`` (``scans/id/plano``),
ou um mapa ``{id: {n: [nx, ny, nz], d: deslocamento_mm}}``.
O lado positivo do plano deve conter a peca. Nao estima um plano novo.
"""

import json
from pathlib import Path

import numpy as np


def validar_perda(antes: int, depois: int, perda_max: float) -> float:
    """Recusa corte vazio ou acima do limite antes de liberar uma copia limpa."""
    if not np.isfinite(perda_max) or not 0 < perda_max < 1:
        raise ValueError("limite de perda deve estar entre 0 e 1")
    if antes <= 0 or depois <= 0 or depois > antes:
        raise ValueError("corte de apoio sem pontos validos")
    perda = 1 - depois / antes
    if perda > perda_max:
        raise ValueError(
            f"corte retiraria {perda:.2%} dos pontos (limite {perda_max:.2%}); "
            "confira o plano no scan original e a recuperacao em outras vistas"
        )
    return perda


def carregar_planos(caminho: str | Path, projeto: str | Path) -> dict[str, dict]:
    """Valida parametros e remapeia as chaves para os IDs do projeto de destino."""
    with Path(caminho).open(encoding="utf-8-sig") as arquivo:
        doc = json.load(arquivo)
    entradas = doc.get("scans", doc)
    if not isinstance(entradas, dict) or not entradas:
        raise ValueError("JSON sem planos de apoio")
    mapa_path = Path(projeto) / "identificadores.json"
    mapa = json.loads(mapa_path.read_text(encoding="utf-8-sig")) if mapa_path.is_file() else {}
    resultado = {}
    for identificador, entrada in entradas.items():
        plano = entrada.get("plano", entrada)
        normal = np.asarray(plano["n"], dtype=float)
        d = float(plano["d"])
        if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isfinite(d):
            raise ValueError(f"{identificador}: plano nao finito ou normal invalida")
        norma = float(np.linalg.norm(normal))
        if norma <= 1e-12:
            raise ValueError(f"{identificador}: normal nula")
        if plano.get("dispersao_mad_mm", 0) > 0.25:
            raise ValueError(f"{identificador}: dispersao do plano maior que 0,25 mm")
        if plano.get("selecao_abaixo_menos_05_pct", 0) > 5:
            raise ValueError(f"{identificador}: plano atravessa a selecao")
        novo = mapa.get(identificador, identificador)
        if novo in resultado:
            raise ValueError(f"plano duplicado para o scan {novo}")
        resultado[novo] = {**plano, "n": normal / norma, "d": d / norma}
    return resultado
