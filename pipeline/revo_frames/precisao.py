"""Evidencia entre quadros por reprojecao, sem mover pontos ou poses.

Um ponto so e contradito quando outra camera observa espaco livre na sua
posicao. Oclusao, falta de cobertura e pares com vies alto sao inconclusivos.
A reprojecao e a validacao rodam na GPU (cupy); so o grafico `render_cortes` (matplotlib) usa a CPU.
"""

from collections import OrderedDict

import gpu
import numpy as np


def _eh_cupy(a):
    return type(a).__module__.split(".")[0] == "cupy"


_FONTE_REPROJ = r"""
extern "C" __global__ void reproj(const double* P, int n, const double* R, const double* T, const float* D, int w, int h,
                                  double fx, double fy, double cx, double cy, double tol, const unsigned char* usar,
                                  unsigned char* apoio, unsigned char* livre, double* residuo){
    int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= n) return;
    apoio[i] = 0; livre[i] = 0; residuo[i] = nan("");
    if (usar && !usar[i]) return;
    double d0 = P[3*i] - T[0], d1 = P[3*i+1] - T[1], d2 = P[3*i+2] - T[2];
    // camera = (P - t) @ R
    double c0 = d0*R[0] + d1*R[3] + d2*R[6];
    double c1 = d0*R[1] + d1*R[4] + d2*R[7];
    double c2 = d0*R[2] + d1*R[5] + d2*R[8];
    double z = -c2;
    if (!(isfinite(c0) && isfinite(c1) && isfinite(c2)) || !(z > 0)) return;
    double uf = rint(c0 * fx / z + cx), vf = rint(-c1 * fy / z + cy);
    if (!(uf >= 1 && vf >= 1 && uf < w - 1 && vf < h - 1)) return;
    int u = (int)uf, v = (int)vf;
    bool algum_apoio = false, todos_livres = true;
    for (int dy = -1; dy <= 1; ++dy) for (int dx = -1; dx <= 1; ++dx){
        double val = D[(v + dy) * w + (u + dx)];
        bool valido = isfinite(val) && val > 0;
        double delta = val - z;
        if (valido && fabs(delta) <= tol) algum_apoio = true;
        if (!(valido && delta > tol)) todos_livres = false;
        if (dy == 0 && dx == 0 && valido) residuo[i] = delta;
    }
    apoio[i] = algum_apoio; livre[i] = todos_livres;
}
"""
_MOD = {}


def reprojetar(P, T, K, D, tol, usar=None):
    """Retorna apoio, contradicao e residuo central (medido - previsto), em mm. Um unico kernel CUDA por chamada
    (janela 3x3 do mapa de profundidade `D`); P numpy ou cupy (as saidas seguem o tipo de P). `usar` (bool) limita o
    calculo aos pontos marcados: os demais saem False/NaN."""
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tolerancia deve ser positiva")
    m = gpu.xp()
    na_gpu = _eh_cupy(P)
    P = m.ascontiguousarray(gpu.para_gpu(P, m.float64))
    D = m.ascontiguousarray(gpu.para_gpu(D, m.float32))
    T = np.asarray(T, np.float64)
    h, w = D.shape
    fx, fy, cx, cy = (float(x) for x in K)
    n = len(P)
    apoio, livre = m.zeros(n, m.uint8), m.zeros(n, m.uint8)
    residuo = m.empty(n, m.float64)
    if n:
        if "k" not in _MOD:
            _MOD["k"] = m.RawModule(code=_FONTE_REPROJ).get_function("reproj")
        u = None if usar is None else m.ascontiguousarray(gpu.para_gpu(usar, bool).astype(m.uint8))
        Rg, Tg = gpu.para_gpu(np.ascontiguousarray(T[:3, :3])), gpu.para_gpu(np.ascontiguousarray(T[:3, 3]))
        th = 128
        _MOD["k"](
            ((n + th - 1) // th,),
            (th,),
            (
                P,
                np.int32(n),
                Rg,
                Tg,
                D,
                np.int32(w),
                np.int32(h),
                np.float64(fx),
                np.float64(fy),
                np.float64(cx),
                np.float64(cy),
                np.float64(tol),
                u if u is not None else np.intp(0),
                apoio,
                livre,
                residuo,
            ),
        )
    apoio, livre = apoio.astype(bool), livre.astype(bool)
    if na_gpu:
        return apoio, livre, residuo
    return gpu.para_cpu(apoio), gpu.para_cpu(livre), gpu.para_cpu(residuo)


def decidir(candidato, apoios, contradicoes, minimo=2):
    """Apoio prevalece. Sem duas contradicoes de quadros distintos, conservar."""
    return candidato & (apoios == 0) & (contradicoes >= minimo)


def render_cortes(base, local, final, caminho):
    """Tres cortes de 1 mm na mesma amostra; nao interpola superficie."""
    if len(base) < 3:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centro = np.median(base, axis=0)
    _, _, eixos = np.linalg.svd(base - centro, full_matrices=False)
    b, l, f = [(p - centro) @ eixos.T for p in (base, local, final)]
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    for linha, pos in enumerate(np.quantile(b[:, 0], [0.25, 0.5, 0.75])):
        for col, (p, nome) in enumerate(((l, "Filtro local"), (f, "Validacao temporal"))):
            ax = axs[linha, col]
            bg = b[np.abs(b[:, 0] - pos) <= 0.5]
            q = p[np.abs(p[:, 0] - pos) <= 0.5]
            ax.scatter(bg[:, 1], bg[:, 2], s=3, c="#c8c8c8", label="peca selecionada")
            ax.scatter(q[:, 1], q[:, 2], s=2, c="#126f98", label="mantidos")
            ax.set_title(f"{nome}: corte em {pos:.1f} mm")
            ax.set_xlabel("mm")
            ax.set_ylabel("mm")
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Mesma amostra (1/8 dos pixels); cortes com espessura de 1 mm")
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


class ValidadorTemporal:
    def __init__(self, scan, tol=1.0, passo=8, minimo=2):
        if tol <= 0 or not np.isfinite(tol) or passo < 1 or minimo not in (2, 3, 4):
            raise ValueError("tol > 0, passo >= 1 e minimo entre 2 e 4")
        self.scan, self.tol, self.passo, self.minimo = scan, tol, passo, minimo
        self.ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
        self.pos = {k: j for j, k in enumerate(self.ks)}
        self.cache = OrderedDict()
        self.diagnosticos = []

    def mapa(self, k):
        if k not in self.cache:
            w, h, mask, z, _ = self.scan.profundidade(self.scan.ids[k])
            m = gpu.xp()
            d = m.full(w * h, m.nan, m.float32)  # mapa de profundidade na GPU (cache de 12 quadros)
            d[gpu.para_gpu(mask)] = gpu.para_gpu(z)
            self.cache[k] = d.reshape(h, w)
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
        self.cache.move_to_end(k)
        return self.cache[k]

    def validar(self, k, P, candidato, referencia=None):
        """Evidencia temporal do quadro k (GPU). P/candidato/referencia numpy ou cupy: as mascaras de saida
        (`rejeitar` e `ultima_evidencia`) seguem o tipo de P. Um kernel por par de quadros e uma unica ida a CPU
        para o diagnostico (mediana e MAD dos residuos de cada vizinho)."""
        m = gpu.xp()
        na_gpu = _eh_cupy(P)
        P = gpu.para_gpu(P, m.float64)
        candidato = gpu.para_gpu(candidato, bool)
        referencia = None if referencia is None else gpu.para_gpu(referencia, bool)
        j = self.pos[k]
        # Diagnostico limitado a 1500 pontos, sem ICP e sem alterar poses.
        ref = P if referencia is None else P[referencia]
        amostra = ref[:: max(1, len(ref) // 1500)]
        vizinhos = [
            self.ks[j + d]
            for d in (-2 * self.passo, -self.passo, self.passo, 2 * self.passo)
            if 0 <= j + d < len(self.ks)
        ]
        mapas = [self.mapa(v) for v in vizinhos]
        Ks = [self.scan.K[int(self.scan.modo[v])] for v in vizinhos]
        # Nao truncar residuos grandes: isso esconderia justamente deriva
        # de rastreio. Oclusoes podem gerar alertas conservadores de revisao.
        diag = []
        if vizinhos:
            res = m.stack(
                [reprojetar(amostra, self.scan.T[v], K, d, self.tol)[2] for v, K, d in zip(vizinhos, Ks, mapas)]
            )
            n_ok = m.isfinite(res).sum(axis=1)
            med = m.nanmedian(res, axis=1)
            mad = 1.4826 * m.nanmedian(m.abs(res - med[:, None]), axis=1)
            diag = gpu.para_cpu(m.stack([n_ok.astype(m.float64), med, mad]))  # unica transferencia
        suporte = m.zeros(len(P), m.uint8)
        livre = suporte.copy()
        diagnostico = []
        for i, (viz, K, d) in enumerate(zip(vizinhos, Ks, mapas)):
            n_i = int(diag[0][i])
            med = float(diag[1][i]) if n_i >= 100 else None
            mad = float(diag[2][i]) if med is not None else None
            suspeito = med is not None and abs(med) > 1.5 * self.tol
            # Dispersao entre quadros so pode elevar a tolerancia ate 2x.
            tol = max(self.tol, min(2 * self.tol, 2.5 * mad)) if mad is not None else self.tol
            diagnostico.append(
                dict(
                    vizinho=int(self.scan.ids[viz]),
                    amostras=n_i,
                    vies_profundidade_mm=med,
                    dispersao_mm=mad,
                    suspeita_desalinhamento=suspeito,
                    tolerancia_mm=tol,
                )
            )
            if suspeito:
                continue
            a, b, _ = reprojetar(P, self.scan.T[viz], K, d, tol, usar=candidato)
            suporte += a
            # Pouca sobreposicao nao permite diagnosticar o par com seguranca.
            if med is not None:
                livre += b
        self.diagnosticos.append(dict(quadro=int(self.scan.ids[k]), pares=diagnostico))
        rejeitar = decidir(candidato, suporte, livre, self.minimo)
        ev = dict(apoio=candidato & (suporte > 0), incertos=candidato & (suporte == 0) & ~rejeitar, removidos=rejeitar)
        self.ultima_evidencia = ev if na_gpu else {c: gpu.para_cpu(v) for c, v in ev.items()}
        cont = gpu.para_cpu(m.stack([candidato.sum(), rejeitar.sum(), ev["apoio"].sum(), ev["incertos"].sum()]))
        return (rejeitar if na_gpu else gpu.para_cpu(rejeitar)), dict(
            candidatos=int(cont[0]), confirmados=int(cont[1]), com_apoio=int(cont[2]), inconclusivos=int(cont[3])
        )
