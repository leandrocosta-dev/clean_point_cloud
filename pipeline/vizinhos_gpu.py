"""Vizinhos por RAIO FIXO na GPU (CuPy): grade hash com chaves ordenadas, 27 celulas, kernels que acumulam direto.

    from vizinhos_gpu import Grade, disponivel
    g = Grade(F, raio)                      # F (m,3) em mm; celula = raio
    D = g.mean_shift(Q, NQ, W, VF, sigma_t, sigma_h, max_passo)
    S0, M, media = g.pca(Q, sigma=0.18)      # somas ponderadas; M = sum w (y-media)(y-media)^T
    n = g.contar(Q)                          # vizinhos a <= raio (inclui o proprio ponto se estiver em F)
    d, j = g.mais_proximo(Q)                 # vizinho mais proximo a <= raio (j=-1 se nenhum)
    d, j = g.knn(Q, k)                       # k vizinhos mais proximos (exato, k <= 64) a <= raio, ordenados; j=-1 se faltar

F e Q podem ser numpy ou cupy. Os metodos devolvem numpy; com `gpu=True` (mais_proximo, contar, somas, knn) devolvem cupy
sem voltar para a CPU. A camada de uso geral esta em `gpu.py` (vizinho_mais_proximo, knn, normais_pca...).

Diferenca conhecida para o caminho CPU do fundir_nitido: la o cKDTree pega os k MAIS PROXIMOS dentro do raio
(k=200 no mean-shift, 48 nas normais); aqui entram TODOS os pontos do raio. Em grade densa isso usa mais vizinhos
(raio efetivo maior). A diferenca e medida no benchmark, nao escondida.
Execute este arquivo para um autoteste contra a referencia exata em CPU (scipy):  python pipeline/vizinhos_gpu.py
"""

import numpy as np

_FONTE = r"""
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

#define CABECALHO \
    const float* P, const long long* u, const int* st, const int* en, int nu, \
    float ox, float oy, float oz, float cel, long long gx, long long gy, long long gz, float raio

#define LACO_VIZINHOS(CORPO) \
    long long cx = (long long)floorf((qx - ox) / cel), cy = (long long)floorf((qy - oy) / cel), \
              cz = (long long)floorf((qz - oz) / cel); \
    float r2 = raio * raio; \
    for (int a = -1; a <= 1; ++a){ long long X = cx + a; if (X < 0 || X >= gx) continue; \
    for (int b = -1; b <= 1; ++b){ long long Y = cy + b; if (Y < 0 || Y >= gy) continue; \
    for (int c = -1; c <= 1; ++c){ long long Z = cz + c; if (Z < 0 || Z >= gz) continue; \
        int j = busca(u, nu, (X * gy + Y) * gz + Z); if (j < 0) continue; \
        for (int p = st[j]; p < en[j]; ++p){ \
            float ex = P[3*p] - qx, ey = P[3*p+1] - qy, ez = P[3*p+2] - qz; \
            float d2 = ex*ex + ey*ey + ez*ez; if (d2 > r2) continue; \
            CORPO \
        } }}}

extern "C" __global__ void mean_shift(const float* Q, const float* NQ, int nq, const float* W, const float* V,
                                      float sig_t, float sig_h, float vmin, float max_passo, float* D, CABECALHO){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= nq) return;
    float qx = Q[3*i], qy = Q[3*i+1], qz = Q[3*i+2];
    float nx = NQ[3*i], ny = NQ[3*i+1], nz = NQ[3*i+2];
    float at = 1.0f / (2.0f * sig_t * sig_t), ah = 1.0f / (2.0f * sig_h * sig_h);
    float sw = 0.f, swh = 0.f;
    LACO_VIZINHOS(
        if (V[3*p]*nx + V[3*p+1]*ny + V[3*p+2]*nz <= vmin) continue;
        float h = ex*nx + ey*ny + ez*nz;
        float t2 = fmaxf(d2 - h*h, 0.f);
        float w = W[p] * expf(-t2 * at - h * h * ah);
        sw += w; swh += w * h;
    )
    float s = swh / fmaxf(sw, 1e-9f);
    D[i] = fminf(fmaxf(s, -max_passo), max_passo);
}

extern "C" __global__ void pca_somas(const float* Q, int nq, float sig, int gauss, float* S, CABECALHO){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= nq) return;
    float qx = Q[3*i], qy = Q[3*i+1], qz = Q[3*i+2];
    float a2 = 1.0f / (2.0f * sig * sig);
    float s0=0, sx=0, sy=0, sz=0, xx=0, yy=0, zz=0, xy=0, xz=0, yz=0;
    LACO_VIZINHOS(
        float w = gauss ? expf(-d2 * a2) : 1.0f;
        s0 += w; sx += w*ex; sy += w*ey; sz += w*ez;
        xx += w*ex*ex; yy += w*ey*ey; zz += w*ez*ez; xy += w*ex*ey; xz += w*ex*ez; yz += w*ey*ez;
    )
    float* o = S + 10 * (long long)i;
    o[0]=s0; o[1]=sx; o[2]=sy; o[3]=sz; o[4]=xx; o[5]=yy; o[6]=zz; o[7]=xy; o[8]=xz; o[9]=yz;
}

extern "C" __global__ void contar(const float* Q, int nq, int* N, CABECALHO){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= nq) return;
    float qx = Q[3*i], qy = Q[3*i+1], qz = Q[3*i+2];
    int n = 0;
    LACO_VIZINHOS( n += 1; )
    N[i] = n;
}

extern "C" __global__ void mais_proximo(const float* Q, int nq, float* Dm, int* J, CABECALHO){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= nq) return;
    float qx = Q[3*i], qy = Q[3*i+1], qz = Q[3*i+2];
    float melhor = 3.4e38f; int jm = -1;
    LACO_VIZINHOS( if (d2 < melhor){ melhor = d2; jm = p; } )
    J[i] = jm; Dm[i] = jm >= 0 ? sqrtf(melhor) : 3.4e38f;
}

#ifdef KMAX
extern "C" __global__ void knn(const float* Q, int nq, int k, float* DD, int* JJ, CABECALHO){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= nq) return;
    float qx = Q[3*i], qy = Q[3*i+1], qz = Q[3*i+2];
    float bd[KMAX]; int bj[KMAX];
    for (int t = 0; t < k; ++t){ bd[t] = 3.4e38f; bj[t] = -1; }
    LACO_VIZINHOS(
        if (d2 < bd[k-1]){
            int t = k - 1;
            while (t > 0 && bd[t-1] > d2){ bd[t] = bd[t-1]; bj[t] = bj[t-1]; --t; }
            bd[t] = d2; bj[t] = p;
        }
    )
    for (int t = 0; t < k; ++t){
        JJ[(long long)i * k + t] = bj[t];
        DD[(long long)i * k + t] = bj[t] >= 0 ? sqrtf(bd[t]) : 3.4e38f;
    }
}
#endif
"""

_MOD = {}


def disponivel():
    """True se CuPy e uma GPU CUDA estao acessiveis."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:  # Isola falhas de backend/automacao ou de um item do lote.
        return False


def _modulo(kmax=0):
    """Modulo compilado; kmax > 0 habilita o kernel knn com vetores locais de kmax posicoes (16, 32 ou 64)."""
    if kmax not in _MOD:
        import cupy as cp

        opcoes = ("--use_fast_math",) + ((f"-DKMAX={kmax}",) if kmax else ())
        _MOD[kmax] = cp.RawModule(code=_FONTE, options=opcoes)
    return _MOD[kmax]


class Grade:
    """Pontos F ordenados por celula (celula = raio) na GPU. Coordenadas em float32 relativas a origem da grade
    (precisao ~0,01 um para pecas de algumas centenas de mm)."""

    BLOCO = 1_000_000  # consultas por lancamento (limita memoria de saida)

    def __init__(self, F, raio):
        import cupy as cp

        self.cp = cp
        F = cp.asarray(F, dtype=cp.float64)  # numpy ou cupy
        if raio <= 0 or not len(F):
            raise ValueError("raio > 0 e F nao vazio")
        self.raio = float(raio)
        self.origem = cp.asnumpy(F.min(0)) - raio
        self.dims = np.floor((cp.asnumpy(F.max(0)) - self.origem) / raio).astype(np.int64) + 2
        if float(np.prod(self.dims.astype(float))) >= 2**62:
            raise ValueError("grade grande demais para chave int64")
        Fl = (F - cp.asarray(self.origem)).astype(cp.float32)
        self.n = len(Fl)
        k = cp.floor(Fl / np.float32(raio)).astype(cp.int64)
        gx, gy, gz = (int(x) for x in self.dims)
        chave = (k[:, 0] * gy + k[:, 1]) * gz + k[:, 2]
        self.ordem = cp.argsort(chave)
        chave = chave[self.ordem]
        self.P = cp.ascontiguousarray(Fl[self.ordem])
        self.u, ini, cnt = cp.unique(chave, return_index=True, return_counts=True)
        self.st = ini.astype(cp.int32)
        self.en = (ini + cnt).astype(cp.int32)

    def _args(self):
        gx, gy, gz = (int(x) for x in self.dims)
        return (
            self.P,
            self.u,
            self.st,
            self.en,
            np.int32(len(self.u)),
            np.float32(0),
            np.float32(0),
            np.float32(0),
            np.float32(self.raio),
            np.int64(gx),
            np.int64(gy),
            np.int64(gz),
            np.float32(self.raio),
        )

    def _q(self, Q):
        cp = self.cp
        return cp.ascontiguousarray((cp.asarray(Q, dtype=cp.float64) - cp.asarray(self.origem)).astype(cp.float32))

    def _ordenado(self, X):
        """Atributo por ponto de F (numpy) na ordem interna da grade, em float32 na GPU."""
        return self.cp.ascontiguousarray(self.cp.asarray(X, dtype=self.cp.float32)[self.ordem])

    def _lancar(self, nome, n, *args, kmax=0):
        th = 128
        _modulo(kmax).get_function(nome)(((n + th - 1) // th,), (th,), args)

    def mean_shift(self, Q, NQ, W, VF, sigma_t, sigma_h, max_passo=0.1, vmin=0.05):
        """Passo ao longo da normal = media de h ponderada (mesma formula de fundir_nitido.mean_shift)."""
        cp = self.cp
        Wd, Vd = self._ordenado(W), self._ordenado(VF)
        out = np.empty(len(Q), np.float64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a : a + self.BLOCO])
            nq = cp.ascontiguousarray(cp.asarray(np.asarray(NQ[a : a + self.BLOCO], np.float32)))
            D = cp.empty(len(q), cp.float32)
            self._lancar(
                "mean_shift",
                len(q),
                q,
                nq,
                np.int32(len(q)),
                Wd,
                Vd,
                np.float32(sigma_t),
                np.float32(sigma_h),
                np.float32(vmin),
                np.float32(max_passo),
                D,
                *self._args(),
            )
            out[a : a + len(q)] = cp.asnumpy(D)
        return out

    def somas(self, Q, sigma=None, gpu=False):
        """(n,10): s0, sum w y (3), sum w y y^T (xx,yy,zz,xy,xz,yz), y = F - q. sigma=None -> pesos 1.
        gpu=True devolve cupy float32 (sem passar pela CPU)."""
        cp = self.cp
        out = cp.empty((len(Q), 10), cp.float32) if gpu else np.empty((len(Q), 10), np.float64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a : a + self.BLOCO])
            S = cp.empty((len(q), 10), cp.float32)
            self._lancar(
                "pca_somas",
                len(q),
                q,
                np.int32(len(q)),
                np.float32(sigma or 1.0),
                np.int32(sigma is not None),
                S,
                *self._args(),
            )
            out[a : a + len(q)] = S if gpu else cp.asnumpy(S)
        return out

    def pca(self, Q, sigma=None):
        """(s0, M, media): M = sum w (y - media)(y - media)^T (a matriz que fundir_nitido decompoe)."""
        S = self.somas(Q, sigma)
        s0 = S[:, 0]
        m = S[:, 1:4] / np.maximum(s0, 1e-30)[:, None]
        xx, yy, zz, xy, xz, yz = (S[:, k] for k in range(4, 10))
        M = np.empty((len(Q), 3, 3))
        M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = xx, yy, zz
        M[:, 0, 1] = M[:, 1, 0] = xy
        M[:, 0, 2] = M[:, 2, 0] = xz
        M[:, 1, 2] = M[:, 2, 1] = yz
        M -= s0[:, None, None] * np.einsum("ni,nj->nij", m, m)
        return s0, M, m

    def contar(self, Q, gpu=False):
        cp = self.cp
        out = cp.empty(len(Q), cp.int64) if gpu else np.empty(len(Q), np.int64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a : a + self.BLOCO])
            N = cp.empty(len(q), cp.int32)
            self._lancar("contar", len(q), q, np.int32(len(q)), N, *self._args())
            out[a : a + len(q)] = N if gpu else cp.asnumpy(N)
        return out

    def mais_proximo(self, Q, gpu=False):
        """(d, j): j = indice em F (ordem ORIGINAL) do mais proximo a <= raio; -1 e d=inf se nenhum."""
        cp = self.cp
        xp = cp if gpu else np
        d = xp.empty(len(Q), xp.float64)
        j = xp.empty(len(Q), xp.int64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a : a + self.BLOCO])
            Dm = cp.empty(len(q), cp.float32)
            J = cp.empty(len(q), cp.int32)
            self._lancar("mais_proximo", len(q), q, np.int32(len(q)), Dm, J, *self._args())
            ok = J >= 0
            jj = cp.where(ok, self.ordem[cp.maximum(J, 0)], -1)
            dd = cp.where(ok, Dm.astype(cp.float64), cp.inf)
            j[a : a + len(q)] = jj if gpu else cp.asnumpy(jj)
            d[a : a + len(q)] = dd if gpu else cp.asnumpy(dd)
        return d, j

    def knn(self, Q, k, gpu=False):
        """(d, j) de forma (n, k): k vizinhos mais proximos DENTRO do raio da grade, em ordem crescente de distancia.
        Exato (k <= 64). Onde faltam vizinhos no raio: j=-1 e d=inf. j em ordem ORIGINAL de F."""
        cp = self.cp
        if not 1 <= k <= 64:
            raise ValueError("k deve estar entre 1 e 64")
        kmax = 16 if k <= 16 else 32 if k <= 32 else 64
        xp = cp if gpu else np
        d = xp.empty((len(Q), k), xp.float64)
        j = xp.empty((len(Q), k), xp.int64)
        bloco = max(1, min(self.BLOCO, 150_000_000 // (12 * k)))
        for a in range(0, len(Q), bloco):
            q = self._q(Q[a : a + bloco])
            Dm = cp.empty((len(q), k), cp.float32)
            J = cp.empty((len(q), k), cp.int32)
            self._lancar("knn", len(q), q, np.int32(len(q)), np.int32(k), Dm, J, *self._args(), kmax=kmax)
            ok = J >= 0
            jj = cp.where(ok, self.ordem[cp.maximum(J, 0)], -1)
            dd = cp.where(ok, Dm.astype(cp.float64), cp.inf)
            j[a : a + len(q)] = jj if gpu else cp.asnumpy(jj)
            d[a : a + len(q)] = dd if gpu else cp.asnumpy(dd)
        return d, j


# ---------------------------------------------------------------- referencia CPU exata (mesma semantica de raio)
def mean_shift_cpu(F, Q, NQ, W, VF, raio, sigma_t, sigma_h, max_passo=0.1, vmin=0.05):
    from scipy.spatial import cKDTree

    arv = cKDTree(F)
    D = np.zeros(len(Q))
    for i, lista in enumerate(arv.query_ball_point(Q, raio)):
        if not lista:
            continue
        e = F[lista] - Q[i]
        h = e @ NQ[i]
        t2 = np.maximum((e * e).sum(1) - h * h, 0)
        w = W[lista] * np.exp(-t2 / (2 * sigma_t**2) - h * h / (2 * sigma_h**2))
        w *= VF[lista] @ NQ[i] > vmin
        D[i] = np.clip((w * h).sum() / max(w.sum(), 1e-9), -max_passo, max_passo)
    return D


def contar_cpu(F, Q, raio):
    from scipy.spatial import cKDTree

    return np.asarray(cKDTree(F).query_ball_point(Q, raio, return_length=True))


if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    F = rng.uniform(0, 20, (200_000, 3))
    F[:, 2] = 0.01 * rng.standard_normal(len(F))  # superficie fina
    Q = F[:2000] + [0, 0, 0.03]
    NQ = np.tile([0, 0, 1.0], (len(Q), 1))
    W = np.ones(len(F))
    VF = np.tile([0, 0, 1.0], (len(F), 1))
    if not disponivel():
        print("CUDA indisponivel")
    else:
        t = time.time()
        g = Grade(F, 0.35)
        D = g.mean_shift(Q, NQ, W, VF, 0.12, 0.08)
        print("gpu", time.time() - t, "s")
        R = mean_shift_cpu(F, Q, NQ, W, VF, 0.35, 0.12, 0.08)
        print("max |gpu-cpu| mean-shift (um):", float(np.abs(D - R).max() * 1000))
        print("contagem igual:", bool(np.array_equal(g.contar(Q), contar_cpu(F, Q, 0.35))))
