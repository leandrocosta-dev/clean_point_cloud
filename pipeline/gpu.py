"""Camada GPU obrigatoria e unica do projeto (CuPy + kernels CUDA proprios).

Todo calculo numerico pesado do pipeline passa por aqui. Se nao ha GPU CUDA, `xp()` levanta `ErroGPU`; o fallback em
numpy/scipy so existe com `REVO_PERMITIR_CPU=1` (depuracao e testes de referencia; registra aviso e e MUITO mais lento).

Convencao de entrada/saida: as funcoes aceitam numpy OU cupy. Se algum argumento de dados for cupy o resultado sai em cupy
(encadeia na GPU sem voltar para a CPU); caso contrario sai em numpy. Resultados pequenos (normal e distancia do plano,
matriz 4x4, dicionarios de informacao) sao sempre numpy/float. Coordenadas em float64 por fora; os kernels de vizinhanca
trabalham em float32 relativo a origem da grade (precisao ~0,01 um em pecas de algumas centenas de mm).

    import gpu
    xp = gpu.xp()                                        # modulo cupy (ErroGPU se nao houver CUDA)
    d, j = gpu.vizinho_mais_proximo(alvo, consultas, dist_max=0.5)    # j = -1 fora do limite
    dist, idx = gpu.knn(pts, pts, k=16)                  # exato; raio_max=None acha o raio sozinho
    N = gpu.normais_pca(pts, k=24, orientar_para=[0, 0, 500])         # normais voltadas para a camera
    C, cont = gpu.reduzir_voxel(pts, 0.2, min_obs=2)     # centroides por voxel de 0,2 mm
    n, d, frac = gpu.ransac_plano(pts, limiar=0.3, iteracoes=3000)    # plano da mesa
    T, info = gpu.icp_ponto_plano(origem, alvo, normais_alvo, T0, dist_max=2.0)   # T leva origem -> alvo
    rot, tam = gpu.componentes_conexos_voxel(pts, 0.5)   # rotulo 0 = maior componente

    for a, b in gpu.blocos(len(pts), bytes_por_item=200):   # processa em fatias que cabem na VRAM
        ...

Execute este arquivo para o benchmark CPU x GPU com 2 M pontos:  python pipeline/gpu.py
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

log = logging.getLogger("gpu")


class ErroGPU(RuntimeError):
    """CUDA/CuPy indisponivel (ou falha ao iniciar a GPU) e REVO_PERMITIR_CPU nao esta ligado."""


# ---------------------------------------------------------------- CUDA_PATH (antes de importar o cupy)
def _pasta_cuda_runtime():
    """Pasta `nvidia/cuda_runtime` dos pacotes pip (nvidia-cuda-runtime-cu12) da venv, ou None."""
    candidatas = []
    try:
        spec = importlib.util.find_spec("nvidia")
        for p in (spec.submodule_search_locations or []) if spec else []:
            candidatas.append(Path(p) / "cuda_runtime")
    except (ImportError, ValueError):
        pass
    candidatas.append(Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_runtime")
    candidatas.append(Path(sys.prefix) / "lib" / "site-packages" / "nvidia" / "cuda_runtime")
    for c in candidatas:
        if c.is_dir():
            return c
    return None


def _configurar_cuda_path():
    atual = os.environ.get("CUDA_PATH")
    if atual and Path(atual).exists():
        return
    pasta = _pasta_cuda_runtime()
    if pasta is not None:
        os.environ["CUDA_PATH"] = str(pasta)


_configurar_cuda_path()

_ESTADO = {}


def _permitir_cpu():
    return os.environ.get("REVO_PERMITIR_CPU", "").strip() == "1"


def xp() -> ModuleType:
    """Modulo de arrays: `cupy` (GPU). Sem CUDA levanta `ErroGPU`, exceto com REVO_PERMITIR_CPU=1 (devolve numpy e avisa).

    xp = gpu.xp(); a = xp.arange(10) ** 2
    """
    if "xp" in _ESTADO:
        return _ESTADO["xp"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            import cupy as cp
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise ErroGPU("nenhuma GPU CUDA encontrada")
        cp.zeros(1, dtype=cp.float32).sum().item()  # forca contexto, driver e NVRTC/cuBLAS carregados
        _ESTADO["xp"] = cp
    except Exception as e:  # ImportError, CUDARuntimeError, ErroGPU...
        if not _permitir_cpu():
            raise ErroGPU(
                f"GPU CUDA indisponivel ({type(e).__name__}: {e}). O projeto exige GPU; instale requirements-cuda.txt e "
                f"confira o driver com `python pipeline/verificar_cuda.py`. Para depurar na CPU: REVO_PERMITIR_CPU=1."
            ) from e
        warnings.warn(
            "REVO_PERMITIR_CPU=1: usando numpy/scipy na CPU (so depuracao; resultados iguais, muito mais lento)",
            RuntimeWarning,
            stacklevel=2,
        )
        log.warning("gpu: fallback para CPU (REVO_PERMITIR_CPU=1)")
        _ESTADO["xp"] = np
    return _ESTADO["xp"]


def em_gpu() -> bool:
    """True se `xp()` e cupy (False so no modo de depuracao em CPU)."""
    return xp() is not np


def _eh_gpu(a):
    return type(a).__module__.split(".")[0] == "cupy"


def _saida_gpu(*dados):
    return em_gpu() and any(_eh_gpu(a) for a in dados)


def _devolver(x, gpu):
    return x if gpu or not em_gpu() else para_cpu(x)


def para_gpu(a: Any, dtype: Any = None) -> Any:
    """numpy/lista/cupy -> array no dispositivo (no modo CPU: numpy). `dtype` opcional."""
    m = xp()
    return m.asarray(a, dtype=dtype) if dtype is not None else m.asarray(a)


def para_cpu(a: Any) -> np.ndarray:
    """cupy/numpy -> numpy."""
    if _eh_gpu(a):
        import cupy as cp

        return cp.asnumpy(a)
    return np.asarray(a)


# ---------------------------------------------------------------- memoria
def memoria_vram() -> tuple[int, int]:
    """(livre, total) em bytes; o livre inclui blocos que o pool do CuPy guarda e pode reutilizar."""
    if not em_gpu():
        return 4 << 30, 4 << 30
    cp = xp()
    livre, total = cp.cuda.runtime.memGetInfo()
    return int(livre + cp.get_default_memory_pool().free_bytes()), int(total)


def liberar_memoria() -> None:
    """Devolve ao driver os blocos ociosos do pool (chamar entre etapas grandes)."""
    if em_gpu():
        cp = xp()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()


def tamanho_bloco(bytes_por_item: Any, fracao_vram: float = 0.6) -> int:
    """Quantos itens cabem em `fracao_vram` da VRAM livre gastando `bytes_por_item` cada (>= 1)."""
    livre, _ = memoria_vram()
    return max(1, int(livre * fracao_vram) // max(1, int(bytes_por_item)))


def blocos(n: int, bytes_por_item: Any, fracao_vram: float = 0.6) -> Iterator[tuple[int, int]]:
    """Gera (a, b) cobrindo range(n) em fatias que cabem na VRAM livre.

    for a, b in gpu.blocos(len(Q), bytes_por_item=k * 40):
        r = f(Q[a:b]); saida[a:b] = para_cpu(r)
    """
    passo = tamanho_bloco(bytes_por_item, fracao_vram)
    for a in range(0, n, passo):
        yield a, min(a + passo, n)


# ---------------------------------------------------------------- agrupamento e voxels
def agrupar(chaves: Any) -> tuple[Any, Any, Any]:
    """(unicas, inverso, contagens): `unicas` ordenadas; `inverso[i]` = posicao da chave i em `unicas`.
    `chaves` 1-D (int64 de chaves_voxel, por exemplo) ou 2-D (linhas distintas).

        unicas, inv, cont = gpu.agrupar(gpu.chaves_voxel(P, 0.2))
    """
    gpu = _saida_gpu(chaves)
    m = xp()
    k = para_gpu(chaves)
    if k.ndim == 1:
        u, inv, cnt = m.unique(k, return_inverse=True, return_counts=True)
    elif k.ndim == 2:
        # CuPy 14.2 unique(axis=0) ordena linhas com comparacoes em Python;
        # muitas linhas iguais tornam esse caminho quadratico. Ordenar colunas
        # na GPU e marcar mudancas preserva a ordem lexicografica sem esse laco.
        n, colunas = k.shape
        if n == 0:
            u, inv, cnt = k.copy(), m.empty(0, m.int64), m.empty(0, m.int64)
        elif colunas == 0:
            u, inv, cnt = k[:1].copy(), m.zeros(n, m.int64), m.asarray([n], m.int64)
        else:
            ordem = m.lexsort(k.T[::-1])
            ordenadas = k[ordem]
            inicio = m.ones(n, dtype=bool)
            inicio[1:] = m.any(ordenadas[1:] != ordenadas[:-1], axis=1)
            u = ordenadas[inicio]
            limites = m.concatenate([m.flatnonzero(inicio), m.asarray([n], m.int64)])
            cnt = m.diff(limites)
            inv = m.empty(n, m.int64)
            inv[ordem] = m.cumsum(inicio, dtype=m.int64) - 1
    else:
        raise ValueError("chaves devem ter uma ou duas dimensoes")
    inv = inv.reshape(-1)
    return _devolver(u, gpu), _devolver(inv, gpu), _devolver(cnt, gpu)


def somar_por_grupo(valores: Any, inverso: Any, n_grupos: int | None = None) -> Any:
    """Soma de `valores` ((n,) ou (n, d)) por grupo de `inverso`; devolve (n_grupos,) ou (n_grupos, d) float64.
    Soma em float64 com atomicos: a ordem varia, diferenca ~1e-16 relativa entre execucoes.

        S = gpu.somar_por_grupo(P, inv)           # soma das coordenadas de cada voxel
    """
    gpu = _saida_gpu(valores, inverso)
    m = xp()
    v = para_gpu(valores, m.float64)
    inv = para_gpu(inverso).astype(m.int64)
    ng = int(n_grupos) if n_grupos is not None else (int(inv.max()) + 1 if len(inv) else 0)
    if v.ndim == 1:
        out = m.bincount(inv, weights=v, minlength=ng)
    else:
        out = m.stack([m.bincount(inv, weights=v[:, c], minlength=ng) for c in range(v.shape[1])], axis=1)
    return _devolver(out, gpu)


_BITS = 21
_MASC = (1 << _BITS) - 1


def chaves_voxel(P: Any, voxel: float, origem: Any = None) -> Any:
    """Chave int64 do voxel de cada ponto: indice (floor((P - origem)/voxel)) empacotado em 3 x 21 bits.
    `origem` padrao = P.min(0); passe a MESMA origem para obter chaves comparaveis entre nuvens/blocos
    (indices devem ficar em [0, 2^21)). Para voltar ao centro: `decodificar_voxel`.

        o = P.min(0); chave = gpu.chaves_voxel(P, 0.2, o)
    """
    gpu = _saida_gpu(P)
    m = xp()
    Pg = para_gpu(P, m.float64)
    o = Pg.min(0) if origem is None else para_gpu(origem, m.float64)
    idx = m.floor((Pg - o) / float(voxel)).astype(m.int64)
    if len(idx) and (int(idx.min()) < 0 or int(idx.max()) > _MASC):
        raise ValueError("indice de voxel fora de [0, 2^21): aumente o voxel ou ajuste a origem")
    chave = (idx[:, 0] << (2 * _BITS)) | (idx[:, 1] << _BITS) | idx[:, 2]
    return _devolver(chave, gpu)


def decodificar_voxel(chaves: Any, voxel: float, origem: Any) -> Any:
    """Centros (m, 3) float64 dos voxels de `chaves` (inverso de chaves_voxel com a mesma origem/voxel)."""
    gpu = _saida_gpu(chaves)
    m = xp()
    k = para_gpu(chaves).astype(m.int64)
    idx = m.stack([(k >> (2 * _BITS)) & _MASC, (k >> _BITS) & _MASC, k & _MASC], axis=1)
    c = (idx.astype(m.float64) + 0.5) * float(voxel) + para_gpu(origem, m.float64)
    return _devolver(c, gpu)


def reduzir_voxel(P: Any, voxel: float, pesos: Any = None, min_obs: int = 1, origem: Any = None) -> tuple[Any, Any]:
    """Centroide (ponderado por `pesos`) dos pontos de cada voxel; descarta voxels com menos de `min_obs` pontos.
    Devolve (centroides (m, 3) float64, contagens (m,) int64), ordenados pela chave do voxel.

        C, n = gpu.reduzir_voxel(P, 0.2)                      # decimacao
        C, n = gpu.reduzir_voxel(P, 0.2, pesos=W, min_obs=3)  # so voxels com >= 3 observacoes
    """
    gpu = _saida_gpu(P, pesos)
    m = xp()
    Pg = para_gpu(P, m.float64)
    o = Pg.min(0) if origem is None else para_gpu(origem, m.float64)
    _, inv, cnt = agrupar(chaves_voxel(Pg, voxel, o))
    rel = Pg - o  # soma relativa a origem: menos erro de arredondamento
    if pesos is None:
        soma = somar_por_grupo(rel, inv, len(cnt))
        C = soma / cnt[:, None] + o
    else:
        w = para_gpu(pesos, m.float64)
        soma = somar_por_grupo(rel * w[:, None], inv, len(cnt))
        sw = somar_por_grupo(w, inv, len(cnt))
        C = soma / m.maximum(sw, 1e-300)[:, None] + o
    manter = cnt >= int(min_obs)
    if not bool(manter.all()):
        C, cnt = C[manter], cnt[manter]
    return _devolver(C, gpu), _devolver(cnt.astype(m.int64), gpu)


# ---------------------------------------------------------------- vizinhos
from vizinhos_gpu import Grade  # noqa: E402,F401  (reexportada: raio fixo, mean-shift, PCA)


class Indice:
    """Indice de vizinhanca reutilizavel de `pts` para consultas repetidas (ICP, varias passadas).
    `raio` e o limite de busca (celula da grade); fora dele: j = -1 e d = inf.

        ind = gpu.Indice(alvo, raio=2.0)
        d, j = ind.mais_proximo(Q)           # Q numpy ou cupy
        n = ind.contar(Q)                    # vizinhos a <= raio (inclui o proprio ponto se estiver em pts)
        d, j = ind.knn(Q, 8)                 # ate 8 vizinhos dentro do raio, em ordem de distancia
    """

    def __init__(self, pts: Any, raio: float) -> Any:
        if raio <= 0 or not np.isfinite(raio):
            raise ValueError("raio deve ser finito e > 0")
        self.raio = float(raio)
        self.n = len(pts)
        if em_gpu():
            self._g = Grade(para_gpu(pts, xp().float64), raio)
        else:
            from scipy.spatial import cKDTree

            self._arv = cKDTree(np.asarray(pts, np.float64))

    def mais_proximo(self, Q: Any) -> Any:
        gpu = _saida_gpu(Q)
        if em_gpu():
            d, j = self._g.mais_proximo(para_gpu(Q, xp().float64), gpu=True)
            return _devolver(d, gpu), _devolver(j, gpu)
        d, j = self._arv.query(np.asarray(Q, np.float64), 1, distance_upper_bound=self.raio, workers=-1)
        j = np.where(np.isfinite(d), j, -1).astype(np.int64)
        return d, j

    def contar(self, Q: Any) -> Any:
        gpu = _saida_gpu(Q)
        if em_gpu():
            return _devolver(self._g.contar(para_gpu(Q, xp().float64), gpu=True), gpu)
        return np.asarray(
            self._arv.query_ball_point(np.asarray(Q, np.float64), self.raio, return_length=True, workers=-1), np.int64
        )

    def knn(self, Q: Any, k: int) -> tuple[Any, Any]:
        gpu = _saida_gpu(Q)
        if em_gpu():
            d, j = self._g.knn(para_gpu(Q, xp().float64), k, gpu=True)
            return _devolver(d, gpu), _devolver(j, gpu)
        d, j = self._arv.query(np.asarray(Q, np.float64), k, distance_upper_bound=self.raio, workers=-1)
        d, j = d.reshape(len(Q), k), j.reshape(len(Q), k)
        return d, np.where(np.isfinite(d), j, -1).astype(np.int64)


def vizinho_mais_proximo(arvore_pts: Any, consultas: Any, dist_max: Any = np.inf) -> tuple[Any, Any]:
    """(dist, idx): vizinho mais proximo em `arvore_pts` de cada consulta; idx = -1 (dist = inf) se alem de `dist_max`.
    Com `dist_max` finito a busca e a mais rapida (grade com celula = dist_max). idx na ordem original de arvore_pts.

        dist, idx = gpu.vizinho_mais_proximo(alvo, origem_transformada, dist_max=1.0)
        casou = idx >= 0
    """
    if np.isfinite(dist_max):
        return Indice(arvore_pts, dist_max).mais_proximo(consultas)
    d, j = knn(arvore_pts, consultas, 1, None)
    return d[:, 0], j[:, 0]


def contar_no_raio(pts: Any, consultas: Any, raio: float | None) -> Any:
    """Numero de pontos de `pts` a <= `raio` de cada consulta (inclui o proprio ponto se a consulta estiver em pts).

    n = gpu.contar_no_raio(P, P, 1.0)        # densidade local
    """
    return Indice(pts, raio).contar(consultas)


def _caixa_diagonal(P, Q):
    m = xp()
    lo = m.minimum(P.min(0), Q.min(0))
    hi = m.maximum(P.max(0), Q.max(0))
    return float(m.linalg.norm(hi - lo)), float(m.prod(m.maximum(hi - lo, 1e-9)))


def _raio_inicial(n, k, diag, vol):
    r = (3.0 * k * vol / (4.0 * math.pi * max(n, 1))) ** (1.0 / 3.0)
    return min(max(r, diag * 1e-4, 1e-9), max(diag, 1e-9))


def _knn_dobrando(P, Q, k, r0, diag):
    """kNN exato sem raio conhecido: raio dobra so para as consultas que ainda nao acharam k vizinhos."""
    m = xp()
    nq = len(Q)
    D = m.full((nq, k), m.inf, dtype=m.float64)
    J = m.full((nq, k), -1, dtype=m.int64)
    pend = m.arange(nq)
    r = r0
    while len(pend):
        d, j = Grade(P, r).knn(Q[pend], k, gpu=True)
        ok = (j[:, k - 1] >= 0) if r < diag else m.ones(len(pend), dtype=bool)
        sel = pend[ok]
        D[sel], J[sel] = d[ok], j[ok]
        pend = pend[~ok]
        r *= 2.0
    return D, J


def _knn_em_blocos(P, Q, k, raio_max=None):
    """Gera (a, b, d, j) por bloco de consultas (cupy), sempre exato; reaproveita a grade em todos os blocos."""
    m = xp()
    nq = len(Q)
    diag, vol = _caixa_diagonal(P, Q)
    if raio_max is not None:
        r = float(raio_max)
    else:
        r0 = _raio_inicial(len(P), k, diag, vol)
        amostra = Q[m.linspace(0, nq - 1, min(nq, 20000)).astype(m.int64)]
        d, _ = _knn_dobrando(P, amostra, k, r0, diag)
        dk = d[:, k - 1]
        dk = dk[m.isfinite(dk)]
        r = float(1.3 * m.quantile(dk, 0.9)) if len(dk) else diag  # o resto dobra o raio so para si
        r = min(max(r, diag * 1e-6, 1e-9), max(diag, 1e-9))
    g = Grade(P, r)
    for a, b in blocos(nq, bytes_por_item=k * 16 * 3):
        d, j = g.knn(Q[a:b], k, gpu=True)
        if raio_max is None and r < diag:
            ruim = j[:, k - 1] < 0
            if bool(ruim.any()):
                d2, j2 = _knn_dobrando(P, Q[a:b][ruim], k, r * 2.0, diag)
                d[ruim], j[ruim] = d2, j2
        yield a, b, d, j


def _knn_iter(P, Q, k):
    """Blocos (a, b, d, j) de kNN exato sem raio; no modo CPU de depuracao, um bloco unico via cKDTree."""
    if em_gpu():
        yield from _knn_em_blocos(P, Q, k)
    else:
        d, j = knn(P, Q, k)
        yield 0, len(Q), d, j


def knn(pts: Any, consultas: Any, k: int, raio_max: float | None = None) -> tuple[Any, Any]:
    """(dist (n, k) float64, idx (n, k) int64): k vizinhos mais proximos em ordem crescente de distancia. Exato, k <= 64.
    `raio_max` dado: busca so ate ele (celula da grade = raio_max; use ~2x a distancia esperada ao k-esimo) e onde faltam
    vizinhos idx = -1, dist = inf. `raio_max=None`: acha o raio sozinho e devolve sempre os k mais proximos
    (idx = -1 so se len(pts) < k). Um ponto de `pts` usado como consulta aparece como seu proprio vizinho (dist 0).

        dist, idx = gpu.knn(P, P, 16, raio_max=1.5)
        vizinhos = P[idx[:, 1:]]          # sem o proprio ponto (se idx[:, 0] == i)
    """
    if not 1 <= k <= 64:
        raise ValueError("k deve estar entre 1 e 64")
    gpu = _saida_gpu(pts, consultas)
    m = xp()
    if not em_gpu():
        from scipy.spatial import cKDTree

        kw = {} if raio_max is None else dict(distance_upper_bound=float(raio_max))
        d, j = cKDTree(np.asarray(pts, np.float64)).query(np.asarray(consultas, np.float64), k, workers=-1, **kw)
        d, j = d.reshape(len(consultas), k), j.reshape(len(consultas), k)
        return d, np.where(np.isfinite(d), j, -1).astype(np.int64)
    P = para_gpu(pts, m.float64)
    Q = para_gpu(consultas, m.float64)
    D = m.empty((len(Q), k), m.float64)
    J = m.empty((len(Q), k), m.int64)
    for a, b, d, j in _knn_em_blocos(P, Q, k, raio_max):
        D[a:b], J[a:b] = d, j
    return _devolver(D, gpu), _devolver(J, gpu)


def _autovetores_menor(M):
    """Autovalores crescentes e autovetor do menor autovalor de matrizes simetricas (n, 3, 3)."""
    m = xp()
    w, v = m.linalg.eigh(M)
    return w, v[:, :, 0]


def normais_pca(
    pts: Any,
    k: int | None = None,
    raio: float | None = None,
    orientar_para: Any = None,
    consultas: Any = None,
    com_curvatura: bool = False,
) -> Any:
    """Normais (n, 3) float64 unitarias = autovetor de menor autovalor da covariancia da vizinhanca (PCA).
    Vizinhanca: `k` vizinhos mais proximos (exato; padrao k=16) OU todos os pontos a <= `raio`. `consultas` (padrao: pts).
    Orientacao: `orientar_para` (3,) = ponto de vista (a normal aponta para ele); (n, 3) = direcao desejada por ponto.
    Sem isso o sinal e arbitrario. Vizinhanca com < 3 pontos -> normal NaN. com_curvatura=True devolve tambem
    l0 / (l0 + l1 + l2) (0 = plano).

        N = gpu.normais_pca(P, k=24, orientar_para=posicao_da_camera)
        N, curv = gpu.normais_pca(P, raio=0.8, com_curvatura=True)
    """
    if k is not None and raio is not None:
        raise ValueError("passe k OU raio")
    gpu = _saida_gpu(pts, consultas, orientar_para)
    m = xp()
    P = para_gpu(pts, m.float64)
    Q = P if consultas is None else para_gpu(consultas, m.float64)
    nq = len(Q)
    N = m.full((nq, 3), m.nan, dtype=m.float64)
    curv = m.full(nq, m.nan, dtype=m.float64)
    if raio is not None:
        if em_gpu():
            g = Grade(P, raio)
            for a, b in blocos(nq, bytes_por_item=10 * 4 + 9 * 8 * 4):
                S = g.somas(Q[a:b], gpu=True).astype(m.float64)
                s0 = S[:, 0]
                med = S[:, 1:4] / m.maximum(s0, 1e-30)[:, None]
                xx, yy, zz, xy, xz, yz = (S[:, c] for c in range(4, 10))
                M = m.empty((b - a, 3, 3), m.float64)
                M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = xx, yy, zz
                M[:, 0, 1] = M[:, 1, 0] = xy
                M[:, 0, 2] = M[:, 2, 0] = xz
                M[:, 1, 2] = M[:, 2, 1] = yz
                M -= s0[:, None, None] * (med[:, :, None] * med[:, None, :])
                _preencher_normais(N, curv, a, b, M, s0 >= 3)
        else:
            from scipy.spatial import cKDTree

            arv = cKDTree(np.asarray(P))
            listas = arv.query_ball_point(np.asarray(Q), raio, workers=-1)
            for i, lst in enumerate(listas):
                if len(lst) >= 3:
                    y = P[lst] - P[lst].mean(0)
                    w, v = np.linalg.eigh(y.T @ y)
                    N[i], curv[i] = v[:, 0], w[0] / max(w.sum(), 1e-300)
    else:
        k = 16 if k is None else int(k)
        for a, b, d, j in _knn_iter(P, Q, k):
            valido = j >= 0
            Y = P[m.maximum(j, 0)]  # (b-a, k, 3)
            cont = valido.sum(1)
            wv = valido[:, :, None].astype(m.float64)
            med = (Y * wv).sum(1) / m.maximum(cont, 1)[:, None]
            Y = (Y - med[:, None, :]) * wv
            M = m.einsum("nki,nkj->nij", Y, Y)
            _preencher_normais(N, curv, a, b, M, cont >= 3)
    if orientar_para is not None:
        o = para_gpu(orientar_para, m.float64)
        direcao = (o - Q) if o.ndim == 1 else o
        N = m.where(((N * direcao).sum(1) < 0)[:, None], -N, N)
    return (_devolver(N, gpu), _devolver(curv, gpu)) if com_curvatura else _devolver(N, gpu)


def _preencher_normais(N, curv, a, b, M, ok):
    m = xp()
    w, nv = _autovetores_menor(m.where(ok[:, None, None], M, m.eye(3)))
    soma = m.maximum(w.sum(1), 1e-300)
    N[a:b] = m.where(ok[:, None], nv, m.nan)
    curv[a:b] = m.where(ok, w[:, 0] / soma, m.nan)


def remover_soltos(
    pts: Any, k: int = 16, desvios: float = 2.0, raio: float | None = None, min_vizinhos: int | None = None
) -> Any:
    """Mascara bool (True = manter) dos pontos que NAO sao soltos.
    Estatistico (padrao, como o Open3D): media das distancias aos `k` vizinhos; solto se > media_global + desvios * std.
    Por raio (`raio` e `min_vizinhos`): solto se tem menos de `min_vizinhos` outros pontos a <= raio.

        manter = gpu.remover_soltos(P, k=16, desvios=2.0)
        P2 = P[manter]
        manter = gpu.remover_soltos(P, raio=1.0, min_vizinhos=5)
    """
    gpu = _saida_gpu(pts)
    m = xp()
    P = para_gpu(pts, m.float64)
    if raio is not None:
        if min_vizinhos is None:
            raise ValueError("com `raio` informe `min_vizinhos`")
        manter = (contar_no_raio(P, P, raio) - 1) >= int(min_vizinhos)
        return _devolver(manter, gpu)
    k = int(k)
    if len(P) <= k:
        return _devolver(m.ones(len(P), dtype=bool), gpu)
    media = m.empty(len(P), m.float64)
    for a, b, d, j in _knn_iter(P, P, k + 1):
        media[a:b] = d[:, 1:].mean(1)  # coluna 0 = o proprio ponto
    limite = float(media.mean() + desvios * media.std())
    return _devolver(media <= limite, gpu)


# ---------------------------------------------------------------- geometria
def ransac_plano(
    P: Any, limiar: float = 1.0, iteracoes: int = 3000, semente: int = 0
) -> tuple[np.ndarray, float, float]:
    """Plano dominante por RANSAC com todas as hipoteses avaliadas em paralelo na GPU.
    Mesmo contrato de `revo_projeto.plano_ransac`: (n (3,) float64, d float, fracao float) com n.x + d = 0; n unitaria,
    refinada por PCA nos inliers e orientada para o lado onde ha mais pontos afastados (o lado dos objetos);
    fracao = proporcao de pontos a < `limiar` do melhor plano candidato (antes do refino).

        n, d, frac = gpu.ransac_plano(P, limiar=0.3, iteracoes=3000, semente=0)
        altura = P @ n + d
    """
    m = xp()
    Pg = para_gpu(P, m.float64)
    n = len(Pg)
    if n < 3:
        raise ValueError("RANSAC precisa de pelo menos 3 pontos")
    c0 = Pg.mean(0)
    Pc = Pg - c0
    P32 = Pc.astype(m.float32)
    gerador = m.random.default_rng(int(semente))
    idx = gerador.integers(0, n, size=(int(iteracoes), 3))
    a, b, c = Pc[idx[:, 0]], Pc[idx[:, 1]], Pc[idx[:, 2]]
    nrm = m.cross(b - a, c - a)
    nn = m.linalg.norm(nrm, axis=1)
    valido = nn > 1e-9
    nrm = nrm / m.maximum(nn, 1e-300)[:, None]
    dd = -(nrm * a).sum(1)
    cont = m.zeros(len(idx), dtype=m.int64)
    N32, d32 = nrm.astype(m.float32), dd.astype(m.float32)
    thr = np.float32(limiar)
    for h0, h1 in blocos(len(idx), bytes_por_item=n * 12):
        dist = P32 @ N32[h0:h1].T + d32[h0:h1]
        cont[h0:h1] = (m.abs(dist) < thr).sum(0)
    cont = m.where(valido, cont, -1)
    melhor = int(cont.argmax())
    if int(cont[melhor]) < 0:
        raise ValueError("RANSAC: todas as hipoteses degeneradas (pontos colineares?)")
    nb, db = nrm[melhor], dd[melhor]
    dist = Pc @ nb + db
    inl = m.abs(dist) < limiar
    fracao = float(inl.mean())
    Pi = Pc[inl]
    ci = Pi.mean(0)
    cov = (Pi - ci).T @ (Pi - ci)
    _, v = m.linalg.eigh(cov)
    nr = v[:, 0]
    dr = -(nr @ ci)
    h = Pc @ nr + dr
    if float((h > 3 * limiar).mean()) < float((h < -3 * limiar).mean()):
        nr, dr = -nr, -dr
    nr = para_cpu(nr).astype(np.float64)
    dr = float(dr) - float(nr @ para_cpu(c0))  # volta para as coordenadas originais
    return nr, dr, fracao


def _rotacao(w):
    """Rodrigues exato de um vetor de rotacao (3,) numpy."""
    ang = float(np.linalg.norm(w))
    if ang < 1e-15:
        return np.eye(3) + _antissim(w)
    k = w / ang
    K = _antissim(k)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _antissim(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float64)


def _pesos_perda(r, perda, c):
    m = xp()
    a = m.abs(r)
    if perda == "tukey":
        return m.where(a < c, (1 - (a / c) ** 2) ** 2, 0.0)
    if perda == "huber":
        return m.where(a <= c, 1.0, c / m.maximum(a, 1e-300))
    if perda in ("l2", "nenhuma"):
        return m.ones_like(r)
    raise ValueError("perda deve ser 'tukey', 'huber' ou 'l2'")


def icp_ponto_plano(
    origem: Any,
    alvo: Any,
    normais_alvo: Any,
    T0: Any = None,
    dist_max: float = 1.0,
    iteracoes: int = 30,
    perda: str = "tukey",
    c: Any = None,
    tol: float = 1e-08,
) -> tuple[np.ndarray, dict[str, Any]]:
    """ICP ponto-plano robusto na GPU. Devolve (T (4, 4) float64 numpy, info) com T levando `origem` para `alvo`
    (alvo ~ T @ origem). Cada iteracao: vizinho mais proximo (<= dist_max) de T*origem em `alvo`, residuo
    r = (p - q) . n_q, pesos robustos (`perda` 'tukey' | 'huber' | 'l2' com escala `c`, padrao dist_max / 3) e
    sistema normal 6x6 resolvido na CPU. Para quando passo < `tol` (rad e unidades). `normais_alvo` unitarias (n, 3).
    info = dict(iteracoes, convergiu, rmse (pares com peso > 0), pares, fitness = pares / len(origem)).

        T, info = gpu.icp_ponto_plano(scan_b, scan_a, normais_a, T0=np.eye(4), dist_max=2.0, iteracoes=40)
        scan_b_alinhado = scan_b @ T[:3, :3].T + T[:3, 3]
    """
    m = xp()
    S = para_gpu(origem, m.float64)
    A = para_gpu(alvo, m.float64)
    NA = para_gpu(normais_alvo, m.float64)
    if len(NA) != len(A):
        raise ValueError("normais_alvo precisa ter uma linha por ponto de alvo")
    T = np.eye(4) if T0 is None else np.array(T0, np.float64)
    c = float(dist_max) / 3.0 if c is None else float(c)
    ind = Indice(A, dist_max)

    def emparelhar(T):
        R, t = T[:3, :3], T[:3, 3]
        p = S @ para_gpu(R.T) + para_gpu(t)
        d, j = ind.mais_proximo(p)
        ok = j >= 0
        p, q, nq = p[ok], A[j[ok]], NA[j[ok]]
        return p, q, nq, ((p - q) * nq).sum(1)

    convergiu, it = False, 0
    for it in range(1, int(iteracoes) + 1):
        p, q, nq, r = emparelhar(T)
        if len(r) < 6:
            break
        w = _pesos_perda(r, perda, c)
        J = m.concatenate([m.cross(p, nq), nq], axis=1)
        Jw = J * w[:, None]
        H = para_cpu(Jw.T @ J)
        g = para_cpu(Jw.T @ r)
        try:
            x = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            x = np.linalg.lstsq(H, -g, rcond=None)[0]
        D = np.eye(4)
        D[:3, :3] = _rotacao(x[:3])
        D[:3, 3] = x[3:]
        T = D @ T
        if float(np.abs(x).max()) < tol:
            convergiu = True
            break
    p, q, nq, r = emparelhar(T)
    w = _pesos_perda(r, perda, c) if len(r) else r
    pares = int((w > 0).sum()) if len(r) else 0
    rmse = float(m.sqrt((r[w > 0] ** 2).mean())) if pares else float("nan")
    info = dict(iteracoes=it, convergiu=convergiu, rmse=rmse, pares=pares, fitness=pares / max(len(S), 1))
    return T, info


# ---------------------------------------------------------------- componentes conexos
_FONTE_CCL = r"""
#define BITS 21
#define MASC ((1LL << BITS) - 1)

__device__ __forceinline__ int busca(const long long* u, int n, long long k){
    int lo = 0, hi = n - 1;
    while (lo <= hi){
        int m = (lo + hi) >> 1;
        long long v = u[m];
        if (v == k) return m;
        if (v < k) lo = m + 1; else hi = m - 1;
    }
    return -1;
}

__device__ __forceinline__ int acha(volatile int* p, int x){
    while (true){
        int px = p[x];
        if (px == x) return x;
        int ppx = p[px];
        p[x] = ppx;
        x = ppx;
    }
}

extern "C" __global__ void unir(const long long* u, int m, const int* off, int no, volatile int* p){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= m) return;
    long long k = u[i];
    long long x = (k >> (2 * BITS)) & MASC, y = (k >> BITS) & MASC, z = k & MASC;
    for (int o = 0; o < no; ++o){
        long long nx = x + off[3*o], ny = y + off[3*o+1], nz = z + off[3*o+2];
        if (nx < 0 || ny < 0 || nz < 0 || nx > MASC || ny > MASC || nz > MASC) continue;
        int j = busca(u, m, (nx << (2 * BITS)) | (ny << BITS) | nz);
        if (j < 0) continue;
        int a = acha(p, i), b = acha(p, j);
        while (a != b){
            int hi = a > b ? a : b, lo = a > b ? b : a;
            int velho = atomicCAS((int*)&p[hi], hi, lo);
            if (velho == hi) break;
            a = acha(p, a); b = acha(p, b);
        }
    }
}

extern "C" __global__ void comprimir(volatile int* p, int m){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= m) return;
    p[i] = acha(p, i);
}
"""

_MOD_CCL = {}


def _deslocamentos(conectividade):
    """Metade (lexicograficamente positiva) dos deslocamentos de vizinhanca 6, 18 ou 26: (3|9|13, 3) int32."""
    maximo = {6: 1, 18: 2, 26: 3}.get(conectividade)
    if maximo is None:
        raise ValueError("conectividade deve ser 6, 18 ou 26")
    out = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) > (0, 0, 0) and abs(dx) + abs(dy) + abs(dz) <= maximo
    ]
    return np.array(out, np.int32)


def componentes_conexos_voxel(P: Any, voxel: float, conectividade: int = 26) -> tuple[Any, Any]:
    """Componentes conexos da ocupacao de voxels (aresta/face/vertice conforme `conectividade` 6, 18 ou 26).
    Devolve (rotulos (n,) int32 por ponto, tamanhos (c,) int64 pontos por componente); rotulo 0 = o maior componente.
    Rotulagem esparsa na GPU (uniao-busca com atomicCAS), sem grade densa: serve para extensoes grandes.

        rot, tam = gpu.componentes_conexos_voxel(P, 0.5)
        P_maior = P[rot == 0]
    """
    gpu = _saida_gpu(P)
    m = xp()
    Pg = para_gpu(P, m.float64)
    chaves = chaves_voxel(Pg, voxel, Pg.min(0))
    u, inv, cont = agrupar(chaves)
    nv = len(u)
    off = _deslocamentos(conectividade)
    if em_gpu():
        cp = m
        if "m" not in _MOD_CCL:
            _MOD_CCL["m"] = cp.RawModule(code=_FONTE_CCL)
        mod = _MOD_CCL["m"]
        pai = cp.arange(nv, dtype=cp.int32)
        offg = cp.asarray(off.reshape(-1))
        th = 128
        grade = ((nv + th - 1) // th,)
        u64 = u.astype(cp.int64)
        for _ in range(32):  # a 1a passada pode perder uniao por corrida; repete ate estabilizar
            antes = pai.copy()
            mod.get_function("unir")(grade, (th,), (u64, np.int32(nv), offg, np.int32(len(off)), pai))
            mod.get_function("comprimir")(grade, (th,), (pai, np.int32(nv)))
            if bool((pai == antes).all()):
                break
        raiz = pai
    else:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        ii, jj = [], []
        for dx, dy, dz in off:
            delta = (int(dx) << (2 * _BITS)) + (int(dy) << _BITS) + int(dz)
            kx = (u >> (2 * _BITS)) & _MASC
            ky = (u >> _BITS) & _MASC
            kz = u & _MASC
            ok = (
                ((kx + dx) >= 0)
                & ((ky + dy) >= 0)
                & ((kz + dz) >= 0)
                & ((kx + dx) <= _MASC)
                & ((ky + dy) <= _MASC)
                & ((kz + dz) <= _MASC)
            )
            alvo = u + delta
            pos = np.searchsorted(u, alvo)
            pos = np.minimum(pos, nv - 1)
            ok &= u[pos] == alvo
            ii.append(np.nonzero(ok)[0])
            jj.append(pos[ok])
        ii, jj = np.concatenate(ii), np.concatenate(jj)
        _, raiz = connected_components(coo_matrix((np.ones(len(ii)), (ii, jj)), shape=(nv, nv)), directed=False)
    _, rinv = m.unique(raiz, return_inverse=True)
    rinv = rinv.reshape(-1)
    rcont = m.bincount(rinv, weights=cont).astype(m.int64)  # pontos (nao voxels) por componente
    ordem = m.argsort(-rcont, kind="stable")
    posto = m.empty(len(ordem), dtype=m.int64)
    posto[ordem] = m.arange(len(ordem))
    rot_voxel = posto[rinv].astype(m.int32)
    rotulos = rot_voxel[inv]
    tam = rcont[ordem].astype(m.int64)
    return _devolver(rotulos, gpu), _devolver(tam, gpu)


# ---------------------------------------------------------------- benchmark
def _bench():
    import time

    from scipy.spatial import cKDTree

    def tempo(f, *a, **kw):
        if em_gpu():
            xp().cuda.Stream.null.synchronize()
        t = time.perf_counter()
        r = f(*a, **kw)
        if em_gpu():
            xp().cuda.Stream.null.synchronize()
        return time.perf_counter() - t, r

    n = 2_000_000
    rng = np.random.default_rng(0)
    # superficie ondulada ~ 200 x 200 mm com ruido (aprox. 0,14 mm entre pontos) + 5% de pontos soltos
    xy = rng.uniform(0, 200, (n, 2))
    z = 5 * np.sin(xy[:, 0] / 17) * np.cos(xy[:, 1] / 23) + 0.02 * rng.standard_normal(n)
    P = np.column_stack([xy, z])
    P[: n // 20] = rng.uniform([0, 0, -20], [200, 200, 40], (n // 20, 3))
    Q = P[rng.choice(n, n, replace=False)] + 0.01 * rng.standard_normal((n, 3))
    print(f"GPU: {xp().__name__}  pontos: {n:,}  (CPU = scipy/numpy com todos os threads)")
    tab = []

    def linha(nome, tc, tg):
        tab.append((nome, tc, tg))
        print(f"  {nome:<34s} CPU {tc:8.2f} s   GPU {tg:8.2f} s   x{tc / max(tg, 1e-9):6.1f}")

    gpu_warm = P[:1000]
    knn(gpu_warm, gpu_warm, 8, 1.0)  # compila kernels antes de medir
    vizinho_mais_proximo(gpu_warm, gpu_warm, 1.0)
    normais_pca(gpu_warm, k=8)
    componentes_conexos_voxel(gpu_warm, 1.0)
    ransac_plano(gpu_warm, 1.0, 10)

    tc, arv = tempo(cKDTree, P)
    tc2, rc = tempo(arv.query, Q, 1, distance_upper_bound=0.5, workers=-1)
    tg, rg = tempo(vizinho_mais_proximo, P, Q, 0.5)
    linha("vizinho mais proximo (dist_max 0,5)", tc + tc2, tg)
    tc2, _ = tempo(arv.query, Q, 16, workers=-1)
    tg, _ = tempo(knn, P, Q, 16, None)
    linha("knn k=16 (indice + consulta)", tc + tc2, tg)
    tc2, _ = tempo(arv.query_ball_point, Q[:200_000], 0.5, return_length=True, workers=-1)
    tg, _ = tempo(contar_no_raio, P, Q[:200_000], 0.5)
    linha("contar no raio 0,5 (200 mil consultas)", tc + tc2, tg)

    def normais_cpu():
        _, j = arv.query(P, 16, workers=-1)
        Y = P[j]
        Y = Y - Y.mean(1, keepdims=True)
        return np.linalg.eigh(np.einsum("nki,nkj->nij", Y, Y))[1][:, :, 0]

    tc2, _ = tempo(normais_cpu)
    tg, _ = tempo(normais_pca, P, k=16)
    linha("normais PCA k=16", tc + tc2, tg)

    def voxel_cpu():
        o = P.min(0)
        idx = np.floor((P - o) / 0.2).astype(np.int64)
        ch = (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]
        u, inv, cnt = np.unique(ch, return_inverse=True, return_counts=True)
        return np.stack([np.bincount(inv, weights=(P - o)[:, c]) for c in range(3)], 1) / cnt[:, None] + o

    tc, rc = tempo(voxel_cpu)
    tg, rg = tempo(reduzir_voxel, P, 0.2)
    linha("reduzir_voxel 0,2 mm", tc, tg)

    def ransac_cpu():
        from revo_projeto import plano_ransac

        return plano_ransac(P[::4], 0.3, 300)  # CPU em 1/4 dos pontos e 300 hipoteses (150x menos)

    tc, _ = tempo(ransac_cpu)
    tc *= 4 * 10  # extrapola para 2 M pontos e 3000 hipoteses
    tg, _ = tempo(ransac_plano, P, 0.3, 3000)
    linha("ransac_plano 3000 hip. (CPU extrap.)", tc, tg)

    def comp_cpu():
        from scipy import ndimage

        idx = np.floor((P - P.min(0)) / 0.5).astype(np.int64)
        g = np.zeros(idx.max(0) + 1, bool)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        return ndimage.label(g, structure=np.ones((3, 3, 3)))

    tc, _ = tempo(comp_cpu)
    tg, _ = tempo(componentes_conexos_voxel, P, 0.5)
    linha("componentes conexos voxel 0,5 mm", tc, tg)

    S = P[::10] + 0.0
    ang = 0.02
    Rt = np.array([[math.cos(ang), -math.sin(ang), 0], [math.sin(ang), math.cos(ang), 0], [0, 0, 1]])
    S = S @ Rt.T + [0.3, -0.2, 0.1]
    Nn = normais_pca(P[::2], k=16, orientar_para=[100, 100, 500])
    tg, (T, info) = tempo(icp_ponto_plano, S, P[::2], Nn, None, 1.5, 20)
    print(f"  icp_ponto_plano ({len(S):,} x {len(P[::2]):,}, 20 it.)  GPU {tg:6.2f} s  {info}")
    return tab


if __name__ == "__main__":
    _bench()
