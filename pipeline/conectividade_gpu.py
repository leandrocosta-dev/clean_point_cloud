"""Componentes exatas do grafo euclidiano de raio, sem truncar a vizinhanca.

    from conectividade_gpu import componentes_por_raio
    rotulos, tamanhos = componentes_por_raio(pontos, 0.2)

Saidas ficam na GPU. Cada celula tem lado 0,55 * raio: sua diagonal e menor
que o raio e seus pontos formam uma clique. Basta encontrar uma aresta entre
duas celulas para uni-las. Um kernel visita todos os pares de celulas candidatas
e procura uma aresta real em float64, sem matriz NxN ou lista de todas as arestas.
Memoria O(n + celulas); o pior tempo continua quadratico para celulas densas
proximas mas desconectadas. Nenhum limite de k altera a conectividade.
"""

from functools import lru_cache

import gpu
import numpy as np

_FONTE = r"""
#define BITS 21
#define MASC ((1LL << BITS) - 1)

__device__ int procurar(const long long* u, int n, long long chave) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        int i = lo + ((hi - lo) >> 1);
        long long k = u[i];
        if (k == chave) return i;
        if (k < chave) lo = i + 1; else hi = i - 1;
    }
    return -1;
}

__device__ int raiz(volatile int* pai, int i) {
    while (true) {
        int px = atomicAdd((int*) &pai[i], 0);
        if (px == i) return i;
        int ppx = atomicAdd((int*) &pai[px], 0);
        atomicMin((int*) &pai[i], ppx);
        i = ppx;
    }
}

__device__ void unir(volatile int* pai, int a, int b) {
    while (true) {
        a = raiz(pai, a); b = raiz(pai, b);
        if (a == b) return;
        int hi = a > b ? a : b, lo = a > b ? b : a;
        if (atomicCAS((int*) &pai[hi], hi, lo) == hi) return;
    }
}

extern "C" __global__ void ligar_celulas(
    const double* P, const long long* u, const long long* ini,
    const long long* cont, int nc, const int* off, int no,
    double raio2, volatile int* pai
) {
    int c = blockDim.x * blockIdx.x + threadIdx.x;
    if (c >= nc) return;
    long long k = u[c];
    long long x = (k >> (2 * BITS)) & MASC;
    long long y = (k >> BITS) & MASC, z = k & MASC;
    int a0 = (int) ini[c], a1 = a0 + (int) cont[c];
    for (int o = 0; o < no; ++o) {
        long long xx = x + off[3*o], yy = y + off[3*o+1], zz = z + off[3*o+2];
        if (xx < 0 || yy < 0 || zz < 0 || xx > MASC || yy > MASC || zz > MASC) continue;
        int viz = procurar(u, nc, (xx << (2 * BITS)) | (yy << BITS) | zz);
        if (viz < 0) continue;
        int b0 = (int) ini[viz], b1 = b0 + (int) cont[viz];
        if (raiz(pai, a0) == raiz(pai, b0)) continue;
        bool ligado = false;
        for (int a = a0; a < a1 && !ligado; ++a) {
            double px = P[3*a], py = P[3*a+1], pz = P[3*a+2];
            for (int b = b0; b < b1; ++b) {
                double dx = px - P[3*b], dy = py - P[3*b+1], dz = pz - P[3*b+2];
                if (dx*dx + dy*dy + dz*dz <= raio2) {
                    unir(pai, a0, b0);
                    ligado = true;
                    break;
                }
            }
        }
    }
}

extern "C" __global__ void comprimir(volatile int* pai, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) pai[i] = raiz(pai, i);
}
"""


@lru_cache(maxsize=None)
def _modulo(dispositivo):
    return gpu.xp().RawModule(code=_FONTE, options=("--fmad=false",))


def componentes_por_raio(pontos, raio):
    """(rotulos, tamanhos) em cupy; liga pares distintos a distancia <= raio.

    Os rotulos seguem o menor indice ORIGINAL de cada componente, como o csgraph
    da referencia CPU. Duplicatas conectam; singletons e entrada vazia sao validos.
    Coordenadas e raio precisam ser finitos. Usa a grade de 3 x 21 bits da gpu.py.
    """
    raio = float(raio)
    if not np.isfinite(raio) or raio <= 0 or not np.isfinite(raio * raio):
        raise ValueError("raio deve ser finito e positivo")
    m = gpu.xp()
    P = gpu.para_gpu(pontos, m.float64)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("pontos precisa ter forma (n,3)")
    if not bool(m.isfinite(P).all()):
        raise ValueError("pontos precisa conter somente coordenadas finitas")
    n = len(P)
    if n == 0:
        return m.empty(0, m.int64), m.empty(0, m.int64)
    if not gpu.em_gpu():  # referencia de depuracao, so com REVO_PERMITIR_CPU=1
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        pares = cKDTree(P).query_pairs(raio, output_type="ndarray")
        _, rot = connected_components(coo_matrix((np.ones(len(pares)), pares.T), shape=(n, n)), directed=False)
        return rot, np.bincount(rot)
    if n > np.iinfo(np.int32).max:
        raise ValueError("grafo excede o limite de indices int32 do kernel")
    chaves = gpu.chaves_voxel(P, float(raio) * 0.55)
    celulas, _, cont = gpu.agrupar(chaves)
    ordem = m.argsort(chaves)
    P = m.ascontiguousarray(P[ordem])
    ini = m.cumsum(cont) - cont
    # A diagonal de cada celula e sqrt(3)*0,55*raio < raio; ligar ao primeiro e exato.
    pai = m.repeat(ini, cont).astype(m.int32)
    # Celulas que diferem em >= 3 indices estao separadas por >= 1,1*raio.
    off = m.asarray(
        [(x, y, z) for x in range(-2, 3) for y in range(-2, 3) for z in range(-2, 3) if (x, y, z) > (0, 0, 0)],
        dtype=m.int32,
    )
    mod = _modulo(m.cuda.runtime.getDevice())
    nc, threads = len(celulas), 128
    mod.get_function("ligar_celulas")(
        ((nc + threads - 1) // threads,),
        (threads,),
        (P, celulas, ini, cont, np.int32(nc), off, np.int32(len(off)), np.float64(raio * raio), pai),
    )
    mod.get_function("comprimir")(((n + threads - 1) // threads,), (threads,), (pai, np.int32(n)))
    raiz_original = m.empty_like(pai)
    raiz_original[ordem] = pai
    _, rot, tam = gpu.agrupar(raiz_original)
    menores = m.full(len(tam), n, m.int64)
    m.minimum.at(menores, rot, m.arange(n, dtype=m.int64))
    ordem_rotulos = m.argsort(menores)
    posto = m.empty_like(ordem_rotulos)
    posto[ordem_rotulos] = m.arange(len(tam), dtype=m.int64)
    return posto[rot], tam[ordem_rotulos]
