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
import re
import shutil
import uuid
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
from comum import salvar_json_atomico, sha256  # noqa: E402

RAIZ_PADRAO = AQUI / "output" / "cache_obs"
VERSAO_NITIDO = 2  # pontos_scan/calibracao_scan: independente do formato de observacoes
VERSAO_CACHE_OBSERVACOES = 1
_DTYPE_OBSERVACOES = np.dtype([("p", "<f4", (3,)), ("v", "<f4", (3,)), ("quadro", "<i4"), ("pixel", "<i4")])


def _hashes_param(scan):
    """Hashes dos Pl.bin capazes de alterar a calibracao automatica; vazio quando nao existem."""
    out = {}
    for arquivo in sorted(Path(scan).glob("param*/Pl.bin")):
        if re.fullmatch(r"param\d*", arquivo.parent.name):
            out[arquivo.relative_to(scan).as_posix()] = sha256(arquivo)
    return out


def _rmtree_dentro(raiz, alvo):
    """Remove somente um filho direto da raiz do cache, validando caminhos resolvidos."""
    root = Path(raiz).resolve()
    target = Path(alvo).resolve()
    if target == root or target.parent != root:
        raise ValueError(f"recusa remover cache fora do filho direto da raiz: {target}")
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)


def chave(scan, calibracao_fixa=None, versao=None):
    """(chave hex, componentes). `versao` padrao = fusao_superficies.PREPARO_VERSAO."""
    if versao is None:
        from fusao_superficies import PREPARO_VERSAO as versao
    scan = Path(scan)
    comp = dict(
        dataset_sha256=sha256(scan / "cache" / "frames.dataset"),
        preview_sha256=sha256(scan / "raw_preview.ply"),
        preparo_versao=int(versao),
        calibracao=calibracao_fixa if calibracao_fixa else "auto",
        parametros_sha256=_hashes_param(scan) if not calibracao_fixa else {},
        versao_cache=VERSAO_CACHE_OBSERVACOES,
    )
    txt = json.dumps(comp, sort_keys=True, ensure_ascii=True, default=float)
    return hashlib.sha256(txt.encode()).hexdigest()[:32], comp


def _valida(pasta, comp):
    obs = None
    try:
        meta = json.loads((pasta / "observacoes.json").read_text(encoding="utf-8"))
        chave_meta = json.loads((pasta / "chave.json").read_text(encoding="utf-8"))
        arquivo = pasta / "observacoes.npy"
        if sha256(arquivo) != meta.get("arquivo_observacoes_sha256"):
            return False, None
        if (
            meta.get("versao_cache") != VERSAO_CACHE_OBSERVACOES
            or meta.get("componentes_cache") != comp
            or chave_meta.get("componentes") != comp
            or meta.get("dataset_sha256") != comp["dataset_sha256"]
            or meta.get("preview_sha256") != comp["preview_sha256"]
        ):
            return False, None
        obs = np.load(arquivo, mmap_mode="r", allow_pickle=False)
        quadros = meta.get("quadros")
        cursor = 0
        valido_quadros = isinstance(quadros, list)
        if valido_quadros:
            for quadro in quadros:
                if not isinstance(quadro, dict):
                    valido_quadros = False
                    break
                inicio, fim = int(quadro.get("inicio", -1)), int(quadro.get("fim", -1))
                if inicio != cursor or fim < inicio or fim > len(obs):
                    valido_quadros = False
                    break
                cursor = fim
            valido_quadros = valido_quadros and cursor == len(obs)
        valido = (
            obs.ndim == 1 and obs.dtype == _DTYPE_OBSERVACOES and len(obs) == meta.get("observacoes") and valido_quadros
        )
        return bool(valido), meta if valido else None
    except (OSError, ValueError, KeyError, TypeError):
        return False, None
    finally:
        if isinstance(obs, np.memmap):
            obs._mmap.close()


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
        _rmtree_dentro(raiz, pasta)  # incompleta/divergente: refazer
    tmp = raiz / f".{k}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        meta = preparar_observacoes(Path(scan), tmp, calibracao_fixa)
        meta = dict(meta)
        meta.update(
            versao_cache=VERSAO_CACHE_OBSERVACOES,
            componentes_cache=comp,
            arquivo_observacoes_sha256=sha256(tmp / "observacoes.npy"),
        )
        salvar_json_atomico(tmp / "observacoes.json", meta)
        salvar_json_atomico(tmp / "chave.json", dict(chave=k, **comp, componentes=comp))
        try:
            os.replace(tmp, pasta)
        except OSError:  # outro processo gravou a mesma chave ao mesmo tempo
            _rmtree_dentro(raiz, tmp)
            ok, meta = _valida(pasta, comp)
            if not ok:
                raise
            return pasta, meta, True
    except BaseException:
        _rmtree_dentro(raiz, tmp)
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


def pontos_gpu(scan, k, K=None, mundo=True):
    """Mesma formula de `RevoScan.pontos`, calculada na GPU (a leitura e a decodificacao da mascara ficam no SQLite/CPU).
    Devolve (P (n, 3) float64 no dispositivo, pix numpy int64, largura, altura) ou None se o quadro nao tem profundidade."""
    import gpu

    r = scan.profundidade(scan.ids[k])
    if r is None:
        return None
    w, h, mask, z, _ = r
    pix = np.flatnonzero(mask)
    fx, fy, cx, cy = K if K is not None else scan.K[int(scan.modo[k])]
    m = gpu.xp()
    v, u = m.divmod(gpu.para_gpu(pix), w)
    zz = gpu.para_gpu(z, m.float64)
    P = m.stack(((u - cx) * zz / fx, -(v - cy) * zz / fy, -zz), axis=1)
    if mundo:
        T = scan.T[k]
        P = P @ gpu.para_gpu(T[:3, :3].T.copy(), m.float64) + gpu.para_gpu(T[:3, 3].copy(), m.float64)
    return P, pix, w, h


def _escrever_pasta(raiz, nome, escrever):
    """Grava `<raiz>/<nome>/` de forma atomica: `escrever(pasta_temporaria)` e depois rename."""
    raiz.mkdir(parents=True, exist_ok=True)
    pasta = raiz / nome
    tmp = raiz / f".{nome}.{uuid.uuid4().hex[:8]}.tmp"
    tmp.mkdir()
    try:
        escrever(tmp)
        try:
            os.replace(tmp, pasta)
        except OSError:  # outro processo gravou a mesma chave ao mesmo tempo
            if not pasta.is_dir():
                raise
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return pasta


def calibracao_scan(scan, calcular, modo=0, raiz=None):
    """(K, erro_mm, reutilizado): calibracao do `modo` do scan, em cache. `calcular()` devolve (K, erro) e so roda
    se nao ha entrada para (sha256 do frames.dataset, do raw_preview.ply, do Pl.bin, modo, VERSAO_NITIDO)."""
    scan = Path(scan)
    pl = scan / ("param" if modo == 0 else f"param{modo}") / "Pl.bin"
    comp = dict(
        dataset_sha256=sha256(scan / "cache" / "frames.dataset"),
        preview_sha256=sha256(scan / "raw_preview.ply"),
        pl_sha256=sha256(pl) if pl.exists() else None,
        modo=int(modo),
        versao=VERSAO_NITIDO,
    )
    k = hashlib.sha256(json.dumps(comp, sort_keys=True).encode()).hexdigest()[:32]
    arq = Path(raiz or RAIZ_PADRAO) / "calibracao" / f"{k}.json"
    try:
        d = json.loads(arq.read_text(encoding="utf-8"))
        if d.get("componentes") == comp:
            return tuple(d["K"]), d["erro_mm"], True
    except (OSError, ValueError, KeyError):
        pass
    K, erro = calcular()
    salvar_json_atomico(arq, dict(componentes=comp, K=list(K), erro_mm=erro))
    return tuple(K), erro, False


def pontos_scan(scan, K, gerar, raiz=None):
    """(P, cam, cont, meta, reutilizado): pontos 3D (mundo do scan, mm) de todos os quadros com profundidade para a
    calibracao `K`, em cache. P = cupy float64 (n, 3); cam (q, 3) float32 = centro da camera de cada quadro; cont (q,) =
    pontos por quadro. `gerar()` devolve (P, cam, cont) e so roda se nao ha entrada para (sha256 do frames.dataset, K,
    VERSAO_NITIDO). Entrada incompleta ou com contagem divergente e refeita."""
    import gpu

    scan = Path(scan)
    comp = dict(
        dataset_sha256=sha256(scan / "cache" / "frames.dataset"), K=[float(x).hex() for x in K], versao=VERSAO_NITIDO
    )
    k = hashlib.sha256(json.dumps(comp, sort_keys=True).encode()).hexdigest()[:32]
    raiz = Path(raiz or RAIZ_PADRAO) / "pontos"
    pasta = raiz / k
    P_cache = None
    try:
        meta = json.loads((pasta / "meta.json").read_text(encoding="utf-8"))
        hashes = meta["artefatos_sha256"]
        if meta["componentes"] == comp and all(sha256(pasta / f) == hashes[f] for f in ("pontos.npy", "quadros.npz")):
            P_cache = np.load(pasta / "pontos.npy", mmap_mode="r", allow_pickle=False)
            with np.load(pasta / "quadros.npz", allow_pickle=False) as z:
                cam, cont = z["cam"], z["cont"]
            if _pontos_validos(P_cache, cam, cont, meta):
                P = gpu.para_gpu(np.asarray(P_cache))
                if gpu.em_gpu():
                    gpu.xp().cuda.get_current_stream().synchronize()
                return P, cam, cont, meta, True
    except (OSError, ValueError, KeyError, TypeError):
        pass
    finally:
        if isinstance(P_cache, np.memmap):
            P_cache._mmap.close()  # liberar o arquivo antes de substituir a entrada no Windows
    P, cam, cont = gerar()
    P = gpu.para_gpu(P, gpu.xp().float64)
    cam, cont = np.asarray(cam, np.float32), np.asarray(cont, np.int64)
    meta = dict(componentes=comp, pontos=int(len(P)), quadros=int(len(cont)), scan=str(scan.resolve()))
    if not _pontos_validos(P, cam, cont, meta):
        raise ValueError("pontos do scan inconsistentes: esperar P(n,3), cam(q,3) e cont(q) somando n")

    def escrever(tmp):
        np.save(tmp / "pontos.npy", gpu.para_cpu(P))
        np.savez(tmp / "quadros.npz", cam=cam, cont=cont)
        meta["artefatos_sha256"] = {f: sha256(tmp / f) for f in ("pontos.npy", "quadros.npz")}
        salvar_json_atomico(tmp / "meta.json", meta)

    if pasta.exists():
        shutil.rmtree(pasta)
    _escrever_pasta(raiz, k, escrever)
    return P, cam, cont, meta, False


def _pontos_validos(P, cam, cont, meta):
    """Contrato dos artefatos de pontos; contagens e formas sao metadados de I/O."""
    return (
        P.ndim == 2
        and P.shape[1] == 3
        and P.dtype == np.float64
        and cont.ndim == 1
        and cam.shape == (len(cont), 3)
        and cont.dtype.kind in "iu"
        and bool((cont >= 0).all())
        and len(P) == meta["pontos"]
        and len(cont) == meta["quadros"]
        and int(cont.sum()) == len(P)
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan")
    ap.add_argument("--raiz", default=str(RAIZ_PADRAO))
    a = ap.parse_args()
    pasta, meta, reuso = obter(a.scan, a.raiz)
    print(f"{'reutilizado' if reuso else 'gravado'}: {pasta} ({meta['observacoes']:,} observacoes)")


if __name__ == "__main__":
    main()
