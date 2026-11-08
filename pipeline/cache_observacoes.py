"""Cache versionado das observacoes de um scan (xyz, visada, quadro, pixel), reaproveitado entre fusoes.

    python pipeline/cache_observacoes.py <pasta_do_scan> [--raiz pipeline/output/cache_obs] [--limpar-orfaos]

Chave = sha256 de (sha256 do frames.dataset, sha256 do raw_preview.ply, versao do preparo, calibracao fixa ou
"auto"). Mesmo dado + mesma versao = mesma pasta `<raiz>/<chave>/` com o formato de
`fusao_superficies.preparar_observacoes` (observacoes.npy + observacoes.json). Qualquer mudanca nos quadros
gera chave nova: nada antigo e usado por engano. A gravacao e atomica (pasta temporaria + rename) e uma
entrada incompleta ou com hash divergente e refeita. O scan de origem nunca e alterado.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
RAIZ_PADRAO = AQUI / "output" / "cache_obs"


def sha256(path, bloco=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bloco), b""):
            h.update(b)
    return h.hexdigest()


def chave(scan, calibracao_fixa=None, versao=None):
    """(chave hex, componentes). `versao` padrao = fusao_superficies.PREPARO_VERSAO."""
    if versao is None:
        from fusao_superficies import PREPARO_VERSAO as versao
    scan = Path(scan)
    comp = dict(dataset_sha256=sha256(scan / "cache" / "frames.dataset"),
                preview_sha256=sha256(scan / "raw_preview.ply"), preparo_versao=int(versao),
                calibracao=calibracao_fixa if calibracao_fixa else "auto")
    txt = json.dumps(comp, sort_keys=True, ensure_ascii=True, default=float)
    return hashlib.sha256(txt.encode()).hexdigest()[:32], comp


def _valida(pasta, comp):
    try:
        meta = json.loads((pasta / "observacoes.json").read_text(encoding="utf-8"))
        obs = np.load(pasta / "observacoes.npy", mmap_mode="r")
        return (meta.get("dataset_sha256") == comp["dataset_sha256"] and meta.get("preview_sha256") == comp["preview_sha256"]
                and len(obs) == meta.get("observacoes")), meta
    except (OSError, ValueError, KeyError):
        return False, None


def obter(scan, raiz=None, calibracao_fixa=None):
    """(pasta, meta, reutilizado). Prepara e grava se a entrada nao existe ou nao confere."""
    from fusao_superficies import preparar_observacoes
    raiz = Path(raiz or RAIZ_PADRAO)
    raiz.mkdir(parents=True, exist_ok=True)
    k, comp = chave(scan, calibracao_fixa)
    pasta = raiz / k
    if pasta.is_dir():
        ok, meta = _valida(pasta, comp)
        if ok:
            return pasta, meta, True
        shutil.rmtree(pasta, ignore_errors=True)       # incompleta/divergente: refazer
    tmp = raiz / f".{k}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        meta = preparar_observacoes(Path(scan), tmp, calibracao_fixa)
        (tmp / "chave.json").write_text(json.dumps(dict(chave=k, **comp), indent=1, default=float), encoding="utf-8")
        try:
            os.replace(tmp, pasta)
        except OSError:                                # outro processo gravou a mesma chave ao mesmo tempo
            shutil.rmtree(tmp, ignore_errors=True)
            ok, meta = _valida(pasta, comp)
            if not ok:
                raise
            return pasta, meta, True
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return pasta, meta, False


def vincular(pasta_cache, destino):
    """Cria `destino` (ex.: <fusao>/origem) com o mesmo conteudo: hardlink no mesmo volume, copia senao.
    Mantem compativel quem le <fusao>/origem/observacoes.* (inspecao_fusao)."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=False)
    for nome in ("observacoes.npy", "observacoes.json"):
        try:
            os.link(pasta_cache / nome, destino / nome)
        except OSError:
            shutil.copy2(pasta_cache / nome, destino / nome)
    (destino / "cache.json").write_text(json.dumps(dict(cache=str(pasta_cache)), indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("--raiz", default=str(RAIZ_PADRAO))
    a = ap.parse_args()
    pasta, meta, reuso = obter(a.scan, a.raiz)
    print(f"{'reutilizado' if reuso else 'gravado'}: {pasta} ({meta['observacoes']:,} observacoes)")


if __name__ == "__main__":
    main()
