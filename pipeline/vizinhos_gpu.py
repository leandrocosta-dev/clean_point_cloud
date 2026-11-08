"""Vizinhos por RAIO FIXO na GPU (CuPy): grade hash com chaves ordenadas, 27 celulas, kernels que acumulam direto.

    from vizinhos_gpu import Grade, disponivel
    g = Grade(F, raio)                      # F (m,3) em mm; celula = raio
    D = g.mean_shift(Q, NQ, W, VF, sigma_t, sigma_h, max_passo)
    S0, M, media = g.pca(Q, sigma=0.18)      # somas ponderadas; M = sum w (y-media)(y-media)^T
    n = g.contar(Q)                          # vizinhos a <= raio (inclui o proprio ponto se estiver em F)
    d, j = g.mais_proximo(Q)                 # vizinho mais proximo a <= raio (j=-1 se nenhum)

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
"""

_MOD = None


def disponivel():
    """True se CuPy e uma GPU CUDA estao acessiveis."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _modulo():
    global _MOD
    if _MOD is None:
        import cupy as cp
        _MOD = cp.RawModule(code=_FONTE, options=("--use_fast_math",))
    return _MOD


class Grade:
    """Pontos F ordenados por celula (celula = raio) na GPU. Coordenadas em float32 relativas a origem da grade
    (precisao ~0,01 um para pecas de algumas centenas de mm)."""

    BLOCO = 1_000_000          # consultas por lancamento (limita memoria de saida)

    def __init__(self, F, raio):
        import cupy as cp
        self.cp = cp
        F = np.asarray(F, np.float64)
        if raio <= 0 or not len(F):
            raise ValueError("raio > 0 e F nao vazio")
        self.raio = float(raio)
        self.origem = F.min(0) - raio
        self.dims = (np.floor((F.max(0) - self.origem) / raio).astype(np.int64) + 2)
        if float(np.prod(self.dims.astype(float))) >= 2 ** 62:
            raise ValueError("grade grande demais para chave int64")
        Fl = cp.asarray((F - self.origem).astype(np.float32))
        k = cp.floor(Fl / np.float32(raio)).astype(cp.int64)
        gx, gy, gz = (int(x) for x in self.dims)
        chave = (k[:, 0] * gy + k[:, 1]) * gz + k[:, 2]
        self.ordem = cp.argsort(chave)
        chave = chave[self.ordem]
        self.P = cp.ascontiguousarray(Fl[self.ordem])
        self.u, ini, cnt = cp.unique(chave, return_index=True, return_counts=True)
        self.st = ini.astype(cp.int32)
        self.en = (ini + cnt).astype(cp.int32)
        self.n = len(F)

    def _args(self):
        cp = self.cp
        gx, gy, gz = (int(x) for x in self.dims)
        return (self.P, self.u, self.st, self.en, np.int32(len(self.u)), np.float32(0), np.float32(0), np.float32(0),
                np.float32(self.raio), np.int64(gx), np.int64(gy), np.int64(gz), np.float32(self.raio))

    def _q(self, Q):
        return self.cp.ascontiguousarray(self.cp.asarray((np.asarray(Q, np.float64) - self.origem).astype(np.float32)))

    def _ordenado(self, X):
        """Atributo por ponto de F (numpy) na ordem interna da grade, em float32 na GPU."""
        return self.cp.ascontiguousarray(self.cp.asarray(np.asarray(X, np.float32))[self.ordem])

    def _lancar(self, nome, n, *args):
        th = 128
        _modulo().get_function(nome)(((n + th - 1) // th,), (th,), args)

    def mean_shift(self, Q, NQ, W, VF, sigma_t, sigma_h, max_passo=0.1, vmin=0.05):
        """Passo ao longo da normal = media de h ponderada (mesma formula de fundir_nitido.mean_shift)."""
        cp = self.cp
        Wd, Vd = self._ordenado(W), self._ordenado(VF)
        out = np.empty(len(Q), np.float64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a:a + self.BLOCO])
            nq = cp.ascontiguousarray(cp.asarray(np.asarray(NQ[a:a + self.BLOCO], np.float32)))
            D = cp.empty(len(q), cp.float32)
            self._lancar("mean_shift", len(q), q, nq, np.int32(len(q)), Wd, Vd, np.float32(sigma_t), np.float32(sigma_h),
                         np.float32(vmin), np.float32(max_passo), D, *self._args())
            out[a:a + len(q)] = cp.asnumpy(D)
        return out

    def somas(self, Q, sigma=None):
        """(n,10): s0, sum w y (3), sum w y y^T (xx,yy,zz,xy,xz,yz), y = F - q. sigma=None -> pesos 1."""
        cp = self.cp
        out = np.empty((len(Q), 10), np.float64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a:a + self.BLOCO])
            S = cp.empty((len(q), 10), cp.float32)
            self._lancar("pca_somas", len(q), q, np.int32(len(q)), np.float32(sigma or 1.0), np.int32(sigma is not None),
                         S, *self._args())
            out[a:a + len(q)] = cp.asnumpy(S)
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

    def contar(self, Q):
        cp = self.cp
        out = np.empty(len(Q), np.int64)
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a:a + self.BLOCO])
            N = cp.empty(len(q), cp.int32)
            self._lancar("contar", len(q), q, np.int32(len(q)), N, *self._args())
            out[a:a + len(q)] = cp.asnumpy(N)
        return out

    def mais_proximo(self, Q):
        """(d, j): j = indice em F (ordem ORIGINAL) do mais proximo a <= raio; -1 e d=inf se nenhum."""
        cp = self.cp
        d = np.empty(len(Q), np.float64)
        j = np.empty(len(Q), np.int64)
        ordem = self.ordem
        for a in range(0, len(Q), self.BLOCO):
            q = self._q(Q[a:a + self.BLOCO])
            Dm = cp.empty(len(q), cp.float32)
            J = cp.empty(len(q), cp.int32)
            self._lancar("mais_proximo", len(q), q, np.int32(len(q)), Dm, J, *self._args())
            jj = cp.where(J >= 0, ordem[cp.maximum(J, 0)], -1)
            j[a:a + len(q)] = cp.asnumpy(jj)
            dd = cp.asnumpy(Dm).astype(np.float64)
            dd[cp.asnumpy(J) < 0] = np.inf
            d[a:a + len(q)] = dd
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
        w = W[lista] * np.exp(-t2 / (2 * sigma_t ** 2) - h * h / (2 * sigma_h ** 2))
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
    F[:, 2] = 0.01 * rng.standard_normal(len(F))       # superficie fina
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
