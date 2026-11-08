"""Baseline de fusao por voxel na GPU contra referencia Revo.

    python pipeline/comparar_fusao_cuda.py SCAN --referencia fuse.ply --saida NOVA

Baseline de centroides: nao substitui fusao que preserva camadas/arestas.
Le frames sem alterar poses ou profundidades. Cache compartilhado identifica
o dataset por SHA-256; referencia externa nao e usada na construcao da fusao.
Fusao, distancias e vies por celula rodam na GPU (`gpu.py`; sem CUDA o script para). Na CPU ficam o SQLite e a
decodificacao dos quadros, a calibracao (`RevoScan.calibrar`) e a leitura/gravacao de PLY (Open3D). `--motor cuda` e
aceito por compatibilidade; o motor CPU deixou de existir (REVO_PERMITIR_CPU=1 so para depuracao).
"""

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from recursos import baixa_prioridade

# Limites definidos antes de importar backends numericos.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "REVO_THREADS"):
    os.environ[_key] = "12"
import geom_gpu
import gpu
import numpy as np
import open3d as o3d
import psutil
from cache_observacoes import pontos_gpu
from comum import sha256  # reexportado: fusao_superficies e ciclo_precisao importam daqui
from revo_projeto import RevoScan


def validar_ply(path):
    """Rejeita nuvem binaria truncada antes que Open3D preencha vertices com zero."""
    tipos = {
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
        "uchar": 1,
        "uint8": 1,
        "char": 1,
        "int8": 1,
        "ushort": 2,
        "uint16": 2,
        "short": 2,
        "int16": 2,
        "int": 4,
        "int32": 4,
        "uint": 4,
        "uint32": 4,
    }
    n = None
    stride = 0
    vertex = False
    formato = None
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError("assinatura PLY invalida")
        for _ in range(300):
            line = f.readline().decode("ascii").strip()
            parts = line.split()
            if line == "end_header":
                break
            if parts[:1] == ["format"]:
                formato = parts[1]
            if parts[:1] == ["element"]:
                vertex = parts[1] == "vertex"
                if vertex:
                    n = int(parts[2])
                elif int(parts[2]) != 0:
                    raise ValueError("esperada nuvem sem faces")
            if vertex and parts[:1] == ["property"]:
                if parts[1] not in tipos:
                    raise ValueError("tipo PLY nao suportado")
                stride += tipos[parts[1]]
        else:
            raise ValueError("cabecalho PLY incompleto")
        size = f.tell()
    if formato != "binary_little_endian" or not n or not stride:
        raise ValueError("nuvem binaria vazia/invalida")
    if Path(path).stat().st_size != size + n * stride:
        raise ValueError("nuvem PLY incompleta ou tamanho inconsistente")
    return n


def preparar(scan_path, cache):
    """Cache de coordenadas em mm para comparar exatamente as mesmas entradas."""
    ident = sha256(scan_path / "cache" / "frames.dataset")
    preview_sha = sha256(scan_path / "raw_preview.ply")
    chave = hashlib.sha256(f"{ident}:{preview_sha}:calibrar-v1".encode()).hexdigest()
    arquivo = cache / f"{chave}.npy"
    meta = cache / f"{chave}.json"
    if arquivo.exists() and meta.exists():
        return np.load(arquivo, mmap_mode="r"), json.loads(meta.read_text())
    cache.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(str(scan_path), tmp)
        try:
            if scan.inconsistentes:
                raise ValueError("dataset contem quadros inconsistentes")
            calib = {}
            for m in sorted(set(scan.modo[scan.tem_prof].tolist())):
                k, erro = scan.calibrar(m)
                calib[str(m)] = dict(K=k, erro_preview_mm=erro)
                print("calibracao", m, calib[str(m)], flush=True)
            pontos = []
            for k in range(len(scan.ids)):
                if scan.tem_prof[k]:
                    p = pontos_gpu(scan, k)[0]
                    if not bool(gpu.xp().isfinite(p).all()):
                        raise ValueError("coordenadas invalidas")
                    pontos.append(gpu.para_cpu(p.astype(gpu.xp().float32)))
            p = np.vstack(pontos)
            np.save(arquivo, p)
            info = dict(
                dataset_sha256=ident,
                preview_sha256=preview_sha,
                scan=str(scan_path.resolve()),
                quadros=int(scan.tem_prof.sum()),
                observacoes=len(p),
                calibracao=calib,
                preparo_s=time.perf_counter() - t,
            )
            meta.write_text(json.dumps(info, indent=2), encoding="utf-8")
        finally:
            scan.c.close()
    return np.load(arquivo, mmap_mode="r"), info


def agregar(xp, key, soma, cont):
    u, inv = xp.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(u)
    s = gpu.somar_por_grupo(soma, inv, n)
    c = xp.bincount(inv, weights=cont, minlength=n)
    return u, s, c


def fundir(p, voxel, minimo, motor=None, bloco=2_000_000):
    """Centroides por voxel na GPU, em lotes de `bloco` pontos (`motor` e legado e ignorado)."""
    xp = gpu.xp()
    # Warm-up separado do tempo de fusao; inclui compilacao inicial.
    t = time.perf_counter()
    agregar(xp, xp.asarray([0, 0, 1]), xp.ones((3, 3)), xp.ones(3))
    if gpu.em_gpu():
        xp.cuda.Stream.null.synchronize()
    warmup = time.perf_counter() - t
    lo = np.min(p, axis=0).astype(np.float64) - voxel
    dims = np.floor((np.max(p, axis=0) - lo) / voxel).astype(np.int64) + 2
    if int(dims[0]) * int(dims[1]) * int(dims[2]) >= np.iinfo(np.int64).max:
        raise ValueError("grade excede chaves int64")
    result = None
    rss = 0
    vram_pool = 0
    t = time.perf_counter()
    for a in range(0, len(p), bloco):
        x = xp.asarray(p[a : a + bloco], dtype=xp.float64)
        k = xp.floor((x - xp.asarray(lo)) / voxel).astype(xp.int64)
        key = (k[:, 0] * int(dims[1]) + k[:, 1]) * int(dims[2]) + k[:, 2]
        r = agregar(xp, key, x, xp.ones(len(x)))
        if result is None:
            result = r
        else:
            result = agregar(
                xp,
                xp.concatenate((result[0], r[0])),
                xp.concatenate((result[1], r[1])),
                xp.concatenate((result[2], r[2])),
            )
        rss = max(rss, psutil.Process().memory_info().rss)
        if gpu.em_gpu():
            vram_pool = max(vram_pool, xp.get_default_memory_pool().total_bytes())
        print(
            "cuda" if gpu.em_gpu() else "cpu", min(a + bloco, len(p)), "/", len(p), "voxels", len(result[0]), flush=True
        )
    _, s, c = result
    ok = c >= minimo
    out = gpu.para_cpu((s[ok] / c[ok, None]).astype(xp.float32))
    return out, dict(
        fusao_incluindo_transferencias_s=time.perf_counter() - t,
        aquecimento_s=warmup,
        rss_amostrado_max_bytes=rss,
        pool_cuda_amostrado_max_bytes=vram_pool,
    )


def estat(d):
    xp = gpu.xp()
    d = gpu.para_gpu(d, xp.float64)
    return dict(
        n=len(d),
        mediana_mm=float(xp.median(d)),
        p90_mm=float(xp.percentile(d, 90)),
        p95_mm=float(xp.percentile(d, 95)),
        acima_03mm_pct=float(100 * xp.mean(d > 0.3)),
    )


def avaliar(p, referencia):
    validar_ply(referencia)
    ref = o3d.io.read_point_cloud(str(referencia))
    xp = gpu.xp()
    if not len(ref.points) or not len(p):
        raise ValueError("nuvem vazia")
    if not ref.has_normals():
        # equivalente do hibrido do Open3D (raio 0,5, ate 40 vizinhos), calculado na GPU
        ref.normals = o3d.utility.Vector3dVector(
            np.nan_to_num(geom_gpu.normais_hibrido(np.asarray(ref.points), 0.5, max_nn=40), nan=0.0)
        )
    r = gpu.para_gpu(np.asarray(ref.points), xp.float64)
    p = gpu.para_gpu(p, xp.float64)
    d, j = gpu.vizinho_mais_proximo(r, p)
    dr = gpu.vizinho_mais_proximo(p, r)[0]
    n = gpu.para_gpu(np.asarray(ref.normals), xp.float64)[j]
    res = ((p - r[j]) * n).sum(1)
    # Orientacao das normais e a da referencia; pares distantes nao medem vies.
    m = d < 0.3
    cells = []
    if bool(m.any()):
        u, inv, cnt = gpu.agrupar(xp.floor(p[m] / 30).astype(xp.int64))
        rm = res[m]
        ordem = xp.lexsort(xp.stack([rm, inv.astype(rm.dtype)]))  # por celula, depois por residuo
        sr = rm[ordem]
        ini = xp.cumsum(cnt) - cnt
        mediana = 0.5 * (sr[ini + (cnt - 1) // 2] + sr[ini + cnt // 2])  # como np.median
        for i in range(len(cnt)):
            if int(cnt[i]) >= 30:
                cells.append(
                    dict(celula=gpu.para_cpu(u[i]).tolist(), n=int(cnt[i]), mediana_assinada_mm=float(mediana[i]))
                )
    return dict(
        nativo_para_revo=estat(d),
        revo_para_nativo=estat(dr),
        residuo_normal_absoluto=estat(xp.abs(res[m])) if bool(m.any()) else None,
        vies_celulas_30mm=cells,
        aviso="Consistencia com referencia, nao exatidao fisica; sem ICP. Nao mede ruido verdadeiro.",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan", type=Path)
    ap.add_argument("--referencia", type=Path, required=True)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("pipeline/output/benchmark_cache"))
    ap.add_argument(
        "--motor", choices=("cuda",), default="cuda", help=argparse.SUPPRESS
    )  # compatibilidade: so existe GPU
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--minimo", type=int, default=3)
    ap.add_argument("--bloco", type=int, default=2_000_000)
    a = ap.parse_args()
    if not np.isfinite(a.voxel) or a.voxel <= 0 or a.minimo < 1 or a.bloco < 1:
        ap.error("voxel positivo, minimo e bloco >= 1")
    if a.saida.exists():
        ap.error("saida deve ser nova")
    baixa_prioridade()
    a.saida.mkdir(parents=True)
    inicio = time.perf_counter()
    p, info = preparar(a.scan, a.cache)
    q, tempos = fundir(p, a.voxel, a.minimo, None, a.bloco)
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(q.astype(float)))
    if not o3d.io.write_point_cloud(str(a.saida / "nuvem.ply"), cloud):
        raise RuntimeError("falha gravando PLY")
    rel = dict(
        entrada=info,
        motor="cuda" if gpu.em_gpu() else "cpu (REVO_PERMITIR_CPU)",
        voxel_mm=a.voxel,
        minimo=a.minimo,
        pontos_saida=len(q),
        tempos=tempos,
        referencia=str(a.referencia.resolve()),
        referencia_sha256=sha256(a.referencia),
        metricas=avaliar(q, a.referencia),
        total_execucao_s=time.perf_counter() - inicio,
        limitacao="Baseline de media por voxel; referencia Revo historica pode usar frames de outra revisao.",
    )
    (a.saida / "relatorio.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(json.dumps(rel, indent=2), flush=True)


if __name__ == "__main__":
    main()
