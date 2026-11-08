"""Fusao orientada na GPU, com camadas e pontos inconclusivos separados.

    python pipeline/fusao_superficies.py SCAN --saida NOVA [--cache [PASTA]]

As poses e profundidades originais nao mudam. Normais/visadas orientadas
separam faces opostas antes da media. Resultados exigem inspecao antes do merge.
Todo o calculo (camadas, normais, refino bilateral, representantes) roda na GPU via `gpu.py` (sem CUDA o script para;
REVO_PERMITIR_CPU=1 so para depuracao). Na CPU ficam so o SQLite/decodificacao dos quadros, a calibracao
(`RevoScan.calibrar`, sem equivalente) e a gravacao de PLY/NPY. `--cache` reaproveita as observacoes (cache_observacoes).
"""

import argparse
import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from comum import configurar_log
from recursos import orcamento_threads  # noqa: E402

orcamento_threads(1)  # antes do numpy: define OMP/OPENBLAS/MKL/REVO_THREADS pelo modo de recursos
import gpu
import numpy as np
import open3d as o3d
import pca3_gpu
from comum import sha256
from revo_projeto import RevoScan

PREPARO_VERSAO = 1  # mudar quando preparar_observacoes mudar o conteudo: invalida o cache_observacoes


_LOGGER = configurar_log(__name__)


def salvar_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)


def salvar_nuvem(path, p, n=None):
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(p, float)))
    if n is not None:
        cloud.normals = o3d.utility.Vector3dVector(np.asarray(n, float))
    # Open3D nao grava nuvem vazia. Nao criar um PLY falso.
    if len(p) and not o3d.io.write_point_cloud(str(path), cloud):
        raise IOError(f"falha ao gravar {path}")


def preparar_observacoes(pasta, saida, calibracao_fixa=None):
    """NPY em disco: xyz, visada, indice de quadro e pixel, sem perder origem. Projecao e visada calculadas na GPU."""
    from cache_observacoes import pontos_gpu

    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=False)
    m = gpu.xp()
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(str(pasta), tmp)
        try:
            if scan.inconsistentes:
                raise ValueError("quadros inconsistentes: revisar antes de fundir")
            calibracao = {}
            for md in sorted(set(scan.modo[scan.tem_prof].tolist())):
                if calibracao_fixa and str(md) in calibracao_fixa:
                    k = tuple(calibracao_fixa[str(md)]["K"])
                    erro = calibracao_fixa[str(md)]["erro_preview_mm"]
                    scan.K[md] = k
                else:
                    k, erro = scan.calibrar(md)
                calibracao[str(md)] = dict(K=k, erro_preview_mm=erro)
            total = sum(len(scan.profundidade(i)[3]) for k, i in enumerate(scan.ids) if scan.tem_prof[k])
            if not total:
                raise ValueError("scan sem observacoes")
            dtype = np.dtype([("p", "<f4", 3), ("v", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")])
            obs = np.lib.format.open_memmap(saida / "observacoes.npy", mode="w+", dtype=dtype, shape=(total,))
            offset = 0
            frames = []
            for k, i in enumerate(scan.ids):
                if not scan.tem_prof[k]:
                    continue
                pg, pix, w, h = pontos_gpu(scan, k)
                v = gpu.para_gpu(scan.T[k][:3, 3], m.float64) - pg
                v /= m.maximum(m.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                if not (bool(m.isfinite(pg).all()) and bool(m.isfinite(v).all())):
                    raise ValueError("observacoes nao finitas")
                end = offset + len(pg)
                obs["p"][offset:end] = gpu.para_cpu(pg)
                obs["v"][offset:end] = gpu.para_cpu(v)
                obs["quadro"][offset:end] = k
                obs["pixel"][offset:end] = pix
                frames.append(dict(indice=k, id=int(i), inicio=offset, fim=end, largura=w, altura=h))
                offset = end
            obs.flush()
            del obs
            meta = dict(
                scan=str(Path(pasta).resolve()),
                dataset_sha256=sha256(Path(pasta) / "cache/frames.dataset"),
                preview_sha256=sha256(Path(pasta) / "raw_preview.ply"),
                observacoes=total,
                quadros=frames,
                calibracao=calibracao,
                referencial="mundo do scan; camera->mundo de FrameInfo, em mm",
            )
            salvar_json(saida / "observacoes.json", meta)
            return meta
        finally:
            scan.c.close()


def _chave_voxel_lado(p, voxel, lado, n_lados):
    """int64 por (voxel absoluto de `voxel` mm, lado): mesma ordem lexicografica (x, y, z, lado) de np.unique(axis=0).
    A origem e multipla do voxel, entao as celulas coincidem com floor(p / voxel) absoluto."""
    m = gpu.xp()
    o = m.floor(p.min(0) / voxel)
    k = (m.floor(p / voxel) - o).astype(m.int64)
    d = (k.max(0) + 1).astype(m.int64)
    if float(d[0]) * float(d[1]) * float(d[2]) * n_lados >= float(np.iinfo(np.int64).max):
        raise ValueError("grade muito grande para chave int64")
    return ((k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]) * n_lados + lado


def reduzir_orientado(obs, frames, voxel, motor=None, bloco=4_000_000):
    """Cada voxel tem ate oito camadas de visada. Cada quadro vota uma vez/camada.

    Divisoes de lote ocorrem apenas ENTRE quadros: contagem de quadros e exata.
    Guarda uma observacao representativa por surfel; todas permanecem no NPY.
    `motor` e legado e ignorado: sempre a GPU (gpu.xp()).
    """
    xp = gpu.xp()
    grupos, lote, tamanho = [], [], 0
    for frame in frames:
        lote.append(frame)
        tamanho += frame["fim"] - frame["inicio"]
        if tamanho >= bloco:
            grupos.append(lote)
            lote = []
            tamanho = 0
    if lote:
        grupos.append(lote)
    lo, hi = None, None  # caixa da nuvem, por lote, na GPU
    for batch in grupos:
        a, b = batch[0]["inicio"], batch[-1]["fim"]
        x = xp.asarray(np.asarray(obs["p"][a:b]), dtype=xp.float64)
        lo = x.min(axis=0) if lo is None else xp.minimum(lo, x.min(axis=0))
        hi = x.max(axis=0) if hi is None else xp.maximum(hi, x.max(axis=0))
    lo = gpu.para_cpu(lo) - voxel
    dims = np.ceil((gpu.para_cpu(hi) - lo) / voxel).astype(np.int64) + 2
    if int(dims[0]) * int(dims[1]) * int(dims[2]) * 8 >= np.iinfo(np.int64).max:
        raise ValueError("grade muito grande para chave int64")
    nf = max(f["indice"] for f in frames) + 1  # indices de quadro podem incluir quadros sem profundidade
    if int(dims[0]) * int(dims[1]) * int(dims[2]) * 8 * nf >= np.iinfo(np.int64).max:
        raise ValueError("chave quadro/voxel excede int64")
    result = None
    for batch in grupos:
        a, b = batch[0]["inicio"], batch[-1]["fim"]
        x = xp.asarray(np.asarray(obs["p"][a:b]), dtype=xp.float64)
        v = xp.asarray(np.asarray(obs["v"][a:b]), dtype=xp.float64)
        k = xp.floor((x - xp.asarray(lo)) / voxel).astype(xp.int64)
        lado = (v[:, 0] >= 0).astype(xp.int64) + 2 * (v[:, 1] >= 0) + 4 * (v[:, 2] >= 0)
        key = ((k[:, 0] * int(dims[1]) + k[:, 1]) * int(dims[2]) + k[:, 2]) * 8 + lado
        u, primeiro, inv = xp.unique(key, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        count = xp.bincount(inv, minlength=len(u)).astype(xp.float64)
        data = xp.concatenate((x, v), axis=1)
        sums = gpu.somar_por_grupo(data, inv, len(u))
        # unique de (surfel, quadro) para nao confundir pixels com visadas independentes.
        fk = xp.asarray(obs["quadro"][a:b], dtype=xp.int64)
        pares = xp.unique(key * nf + fk) // nf
        iu = xp.searchsorted(u, pares)
        votos = xp.bincount(iu, minlength=len(u)).astype(xp.float64)
        representante = primeiro.astype(xp.int64) + a
        if result is not None:
            keys = xp.concatenate((result[0], u))
            values = xp.concatenate((result[1], sums))
            counts = xp.concatenate((result[2], count))
            votes = xp.concatenate((result[3], votos))
            reps = xp.concatenate((result[4], representante))
            u, first, inv = xp.unique(keys, return_index=True, return_inverse=True)
            inv = inv.reshape(-1)
            sums = gpu.somar_por_grupo(values, inv, len(u))
            count = xp.bincount(inv, weights=counts, minlength=len(u))
            votos = xp.bincount(inv, weights=votes, minlength=len(u))
            representante = reps[first]
        result = u, sums, count, votos, representante
        _LOGGER.info("%s", f"camadas: {b:,}/{len(obs):,} observacoes, {len(u):,} surfels")
    _, s, c, frames_count, rep = result
    pos = s[:, :3] / c[:, None]
    vis = s[:, 3:]
    coerencia = xp.linalg.norm(vis, axis=1) / c
    vis = vis / xp.maximum(xp.linalg.norm(vis, axis=1, keepdims=True), 1e-12)
    cpu = gpu.para_cpu
    return (
        cpu(pos).astype(np.float32),
        cpu(vis).astype(np.float32),
        cpu(c).astype(np.uint32),
        cpu(frames_count).astype(np.uint32),
        cpu(rep),
        cpu(coerencia),
    )


def _normais_das_camadas(centro, vc, rn, k=40):
    """Normal de cada centroide: PCA gaussiano dos ate `k` vizinhos (exatos) em `rn` mm, ignorando camadas de visada
    oposta. Devolve (normais (n, 3), ambiguo (n,)) na GPU."""
    m = gpu.xp()
    nc = m.empty_like(centro)
    ac = m.zeros(len(centro), bool)
    g = gpu.Grade(centro, rn)
    for a, b in gpu.blocos(len(centro), bytes_por_item=k * (16 + 8 * 10)):
        q = centro[a:b]
        d, idx = g.knn(q, k, gpu=True)
        valid = idx >= 0
        idx = m.where(valid, idx, 0)
        # Camadas com visadas opostas nao participam do ajuste do plano.
        valid &= m.einsum("nkj,nj->nk", vc[idx], vc[a:b]) > 0.15
        w = m.where(valid, m.exp(-(m.minimum(d, rn) ** 2) / (2 * (rn / 2) ** 2)), 0)
        sw = m.maximum(w.sum(axis=1), 1e-12)
        delta = centro[idx] - q[:, None, :]
        local = (delta * w[..., None]).sum(axis=1) / sw[:, None]
        y = delta - local[:, None, :]
        cov = m.einsum("nk,nki,nkj->nij", w, y, y) / sw[:, None, None]
        # autovalores crescentes l0 <= l1 <= l2: o menor da a normal; l0/l1 mede a planaridade
        l0, n = pca3_gpu.menor_autopar(cov)
        tr = m.einsum("nii->n", cov)
        l2 = _maior_autovalor(cov, tr)
        l1 = tr - l0 - l2
        n = n * m.where(m.einsum("ij,ij->i", n, vc[a:b]) < 0, -1, 1)[:, None]
        nc[a:b] = n
        ac[a:b] = (valid.sum(axis=1) < 6) | (l0 / m.maximum(l1, 1e-12) > 0.35)
    return nc, ac


def _maior_autovalor(M, tr):
    """Maior autovalor (forma fechada) de matrizes simetricas 3x3 (n, 3, 3), com tr = traco."""
    m = gpu.xp()
    q = tr / 3.0
    b = M - q[:, None, None] * m.eye(3)
    p = m.sqrt(m.maximum((b * b).sum(axis=(1, 2)) / 6.0, 0))
    c = b / m.maximum(p, 1e-300)[:, None, None]
    det = (
        c[:, 0, 0] * (c[:, 1, 1] * c[:, 2, 2] - c[:, 1, 2] * c[:, 2, 1])
        - c[:, 0, 1] * (c[:, 1, 0] * c[:, 2, 2] - c[:, 1, 2] * c[:, 2, 0])
        + c[:, 0, 2] * (c[:, 1, 0] * c[:, 2, 1] - c[:, 1, 1] * c[:, 2, 0])
    )
    phi = m.arccos(m.clip(det / 2.0, -1.0, 1.0)) / 3.0
    return q + 2.0 * p * m.cos(phi)


def refinar_camadas(p, vis, motor=None, raio=0.3, sigma=0.08, max_desloc=0.08, passadas=2):
    """Bilateral na normal; nao mistura normais opostas e preserva cantos ambiguos. Tudo na GPU (`motor` e legado e
    ignorado). Devolve arrays numpy."""
    m = gpu.xp()
    p = np.asarray(p)
    pg = gpu.para_gpu(p, m.float64)
    visg = gpu.para_gpu(vis, m.float64)
    # Normais sobre centroides mais espacados: nos dados brutos o ruido pode
    # dominar os vizinhos imediatos e dar PCA isotropico em quase todo o scan.
    eixo = m.abs(visg).argmax(axis=1)
    lado = eixo * 2 + (visg[m.arange(len(visg)), eixo] < 0)
    _, inv, cnt = gpu.agrupar(_chave_voxel_lado(pg, max(0.5, 6 * sigma), lado, 6))
    centro = gpu.somar_por_grupo(pg, inv, len(cnt)) / cnt[:, None]
    vc = gpu.somar_por_grupo(visg, inv, len(cnt))
    vc /= m.maximum(m.linalg.norm(vc, axis=1, keepdims=True), 1e-12)
    nc, ac = _normais_das_camadas(centro, vc, max(1.2, 4 * raio))
    # inv liga cada observacao ao seu centroide de mesma camada de visada.
    normals = nc[inv]
    ambiguo = ac[inv]
    del centro, vc, nc, ac
    out = pg.copy()
    for _ in range(passadas):
        g = gpu.Grade(out, raio)
        destino = out.copy()
        for a, b in gpu.blocos(len(out), bytes_por_item=64 * (16 + 8 * 8)):
            q = out[a:b]
            d, idx = g.knn(q, 64, gpu=True)
            valid = idx >= 0
            idx = m.where(valid, idx, 0)
            delta = out[idx] - q[:, None, :]
            n = normals[a:b]
            h = m.einsum("nkj,nj->nk", delta, n)
            cos = m.einsum("nkj,nj->nk", normals[idx], n)
            # Salto de altura e de normal define uma fronteira que nao e promediada.
            valid = valid & (cos > 0.9) & (m.abs(h) < 2 * sigma)
            dd = m.minimum(d, raio)
            tang = m.maximum(dd * dd - h * h, 0)
            w = m.where(valid, m.exp(-tang / (2 * (raio / 2) ** 2) - h * h / (2 * sigma * sigma)), 0)
            step = (w * h).sum(axis=1) / m.maximum(w.sum(axis=1), 1e-12)
            step = m.clip(step, -max_desloc / passadas, max_desloc / passadas)
            step = m.where(ambiguo[a:b], 0, step)
            destino[a:b] = q + step[:, None] * n
        out = destino
    cpu = gpu.para_cpu
    return (
        cpu(out).astype(p.dtype),
        cpu(normals).astype(np.float32),
        cpu(ambiguo),
        cpu(m.linalg.norm(out - pg, axis=1)),
    )


def representantes(p, n, count, voxel):
    """Reamostra sem misturar normais opostas nem mover o representante escolhido (maior `count` por celula e lado;
    empate: o de menor indice). Devolve indices numpy, ordenados pela celula."""
    m = gpu.xp()
    pg = gpu.para_gpu(p, m.float64)
    ng = gpu.para_gpu(n, m.float64)
    eixo = m.abs(ng).argmax(axis=1)
    lado = eixo * 2 + (ng[m.arange(len(ng)), eixo] < 0)
    _, inv, _ = gpu.agrupar(_chave_voxel_lado(pg, voxel, lado, 6))
    cg = gpu.para_gpu(count).astype(m.int64)
    ordem = m.lexsort(m.stack([-cg, inv]))  # por celula, depois maior contagem; empate: menor indice
    ig = inv[ordem]
    primeiro = m.flatnonzero(m.concatenate([m.ones(1, bool), ig[1:] != ig[:-1]]))
    return gpu.para_cpu(ordem[primeiro])


def executar(scan, saida, motor=None, voxel=0.08, min_quadros=3, calibracao_fixa=None, cache=None):
    """`motor` e legado e ignorado (sempre a GPU). cache=None: prepara as observacoes em <saida>/origem (comportamento original).
    cache=True ou pasta: reutiliza cache_observacoes (chave = hashes do dataset/preview + versao + calibracao)
    e vincula <saida>/origem a ele (hardlink), entao quem le origem/ continua funcionando."""
    if voxel <= 0 or not np.isfinite(voxel) or min_quadros < 1:
        raise ValueError("voxel positivo e min_quadros >=1")
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=False)
    try:
        from recursos import aplicar_prioridade

        aplicar_prioridade()
    except (ImportError, OSError):
        pass
    m = gpu.xp()  # ErroGPU se nao houver CUDA
    t = time.perf_counter()
    info_cache = None
    if cache:
        from cache_observacoes import obter, vincular

        pasta, meta, reuso = obter(Path(scan), None if cache is True else cache, calibracao_fixa)
        vincular(pasta, saida / "origem")
        info_cache = dict(pasta=str(pasta), reutilizado=reuso, segundos=time.perf_counter() - t)
    else:
        meta = preparar_observacoes(Path(scan), saida / "origem", calibracao_fixa)
    obs = np.load(saida / "origem/observacoes.npy", mmap_mode="r")
    p, vis, count, nf, reps, coerencia = reduzir_orientado(obs, meta["quadros"], voxel, motor)
    p0 = p.copy()
    p, n, ambiguo, desl = refinar_camadas(p, vis, motor, raio=max(0.2, 3 * voxel), sigma=voxel)
    sel = representantes(p, n, count, voxel)
    p, p0, n, vis, count, nf, reps, coerencia, ambiguo, desl = [
        x[sel] for x in (p, p0, n, vis, count, nf, reps, coerencia, ambiguo, desl)
    ]
    certo = gpu.para_cpu((gpu.para_gpu(nf) >= min_quadros) & ~gpu.para_gpu(ambiguo) & (gpu.para_gpu(coerencia) > 0.5))
    dg = gpu.para_gpu(desl, m.float64)
    salvar_nuvem(saida / "nuvem.ply", p, n)
    salvar_nuvem(saida / "nuvem_confirmada.ply", p[certo], n[certo])
    salvar_nuvem(saida / "nuvem_incertos.ply", p[~certo], n[~certo])
    np.savez(
        saida / "superficies.npz",
        pontos=p,
        pontos_medidos=p0,
        normais=n,
        visadas=vis,
        observacoes=count,
        quadros=nf,
        representante=reps,
        confirmado=certo,
        deslocamento_mm=desl,
    )
    rel = dict(
        versao_motor=1,
        codigo_sha256=sha256(Path(__file__)),
        motor="cuda" if gpu.em_gpu() else "cpu (REVO_PERMITIR_CPU)",
        voxel_mm=voxel,
        min_quadros=min_quadros,
        entrada=meta,
        pontos=len(p),
        confirmados=int(certo.sum()),
        incertos=int((~certo).sum()),
        deslocamento_p95_mm=float(m.percentile(dg, 95)),
        deslocamento_max_mm=float(dg.max()),
        segundos=time.perf_counter() - t,
        normal_voxel_mm=max(0.5, 6 * voxel),
        normal_raio_mm=max(1.2, 4 * max(0.2, 3 * voxel)),
        nuvem_sha256=sha256(saida / "nuvem.ply"),
        cache_observacoes=info_cache,
        estado="revisao_pendente",
        limites="Camadas por octante de visada: protecao parcial, nao garantia universal de separacao. "
        "Quadros distintos nao equivalem a scans/poses independentes. Incertos preservados.",
    )
    salvar_json(saida / "fusao.json", rel)
    print(json.dumps({k: v for k, v in rel.items() if k != "entrada"}, indent=2), flush=True)
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan")
    ap.add_argument("--saida", required=True)
    ap.add_argument(
        "--motor", choices=["cuda"], default="cuda", help=argparse.SUPPRESS
    )  # compatibilidade: so existe GPU
    ap.add_argument("--voxel", type=float, default=0.08)
    ap.add_argument("--min-quadros", type=int, default=3)
    ap.add_argument(
        "--cache",
        nargs="?",
        const=True,
        default=None,
        help="reutiliza observacoes em cache (pasta opcional; padrao pipeline/output/cache_obs)",
    )
    a = ap.parse_args()
    executar(a.scan, a.saida, None, a.voxel, a.min_quadros, cache=a.cache)


if __name__ == "__main__":
    main()
