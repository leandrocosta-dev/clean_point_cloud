"""Utilitarios comuns do pipeline (antes copiados em varios scripts).

    from comum import sha256, ler_ply, gravar_ply, salvar_json_atomico, configurar_log, destino_novo

    log = configurar_log("limpar_raw", arquivo=saida / "limpar_raw.log")
    saida = destino_novo(args.saida)                    # recusa destino com conteudo (nunca sobrescreve)
    cab, dados = ler_ply("a.ply")                       # dados (n, nprops) float32; cab = texto do cabecalho
    gravar_ply("b.ply", cab, dados[mascara])            # reescreve "element vertex" com o novo numero de pontos
    salvar_json_atomico("resultado.json", {"ok": True})  # grava em .tmp e troca (nao deixa JSON pela metade)
    print(sha256("frames.dataset"))

Importavel de qualquer pasta depois de `pip install -e .` (ver pyproject.toml).
"""
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
import tempfile

import numpy as np


def sha256(path, bloco=1 << 22):
    """SHA-256 hexadecimal do arquivo, lido em blocos de 4 MiB."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def ler_ply(path):
    """PLY binario little-endian com propriedades float32 -> (cabecalho str, array (n, nprops) float32).

    Mesmo formato de `revo_frames/revo_projeto.py::ler_ply`."""
    with open(path, "rb") as f:
        b = f.read()
    marca = b"end_header\n"
    k = b.find(marca)
    if k < 0:
        raise ValueError(f"{path}: PLY sem end_header")
    k += len(marca)
    h = b[:k].decode("latin1")
    linhas = h.split("\n")
    n = int([ln for ln in linhas if ln.startswith("element vertex")][0].split()[-1])
    nprop = len([ln for ln in linhas if ln.startswith("property")])
    return h, np.frombuffer(b[k:k + n * 4 * nprop], "<f4").reshape(n, nprop).copy()


def gravar_ply(path, header, arr):
    """Grava PLY binario float32: `header` (texto de ler_ply) com `element vertex` ajustado a len(arr)."""
    linhas = header.split("\n")
    linhas = [f"element vertex {len(arr)}" if ln.startswith("element vertex") else ln for ln in linhas]
    with open(path, "wb") as f:
        f.write("\n".join(linhas).encode("latin1"))
        f.write(np.ascontiguousarray(arr).astype("<f4").tobytes())


def _padrao_json(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"nao serializavel: {type(o).__name__}")


def salvar_json_atomico(path, valor, indent=2):
    """Grava JSON UTF-8 num arquivo temporario da mesma pasta e troca com os.replace (atomico).

    Aceita tipos numpy e Path. Cria a pasta se preciso. Devolve o Path gravado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(valor, f, ensure_ascii=False, indent=indent, default=_padrao_json)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def configurar_log(nome, arquivo=None, nivel=logging.INFO):
    """Logger `nome` com formato `HH:MM:SS nivel mensagem` no stdout e, opcionalmente, em `arquivo` (anexa).

    Idempotente: chamar de novo com o mesmo nome nao duplica as saidas."""
    log = logging.getLogger(nome)
    log.setLevel(nivel)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    saida = logging.StreamHandler(sys.stdout)
    saida.setFormatter(fmt)
    log.addHandler(saida)
    if arquivo is not None:
        arquivo = Path(arquivo)
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(arquivo, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    return log


def destino_novo(path, criar=True):
    """Pasta de saida que nao pode ter conteudo: encerra com `destino ja existe: ...` (SystemExit) se `path` for
    arquivo ou pasta nao vazia. Pasta vazia e reaproveitada. Com criar=True cria a pasta; devolve o Path."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        sys.exit(f"destino ja existe: {path}")
    if criar:
        path.mkdir(parents=True, exist_ok=True)
    return path
