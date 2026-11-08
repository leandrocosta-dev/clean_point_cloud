"""Leitura e limpeza quadro a quadro de projetos do Revo Scan 6 (formato 1.3).

Regras de seguranca:
- O projeto original nunca e alterado. Tudo e gravado numa copia.
- As poses NAO sao alteradas. A limpeza so apaga pixels do mapa de profundidade
  de cada quadro (mascara de bits + valores), mantendo o resto do registro igual.
- Nao depende do Open3D para limpar (registro do fuse.ply usa Open3D so no RANSAC por features, sem equivalente na GPU).
- Todo o calculo numerico roda na GPU (CuPy via gpu.py): entra numpy ou cupy, sai o mesmo tipo. Ficam na CPU so a leitura
  do SQLite/PLY, a decodificacao do blob do quadro, o Nelder-Mead do scipy (otimizador escalar; o custo roda na GPU) e o
  FPFH + RANSAC por features do Open3D.

Formato (confirmado em 5 projetos):
- cache/frames.dataset: SQLite com os 16 primeiros bytes trocados por b'RPAlgo_file_sys\\0'.
- FrameInfo.bin: cabecalho 16 bytes + pose 4x4 float64 (camera para mundo) + cauda.
  Modo do quadro = (hdr[1] >> 16) & 0xff (indice da pasta de calibracao param, param1...).
- FrameRGBD.data: largura, altura, n (uint32) + mascara de bits w*h (ordem little)
  + 4 bytes + n float32 (profundidade em mm, na ordem linha a linha dos pixels validos).
- Ordem temporal: (id >> 32, id & 0xffffffff). Numeracao do Revo Scan = indice entre
  os quadros que tem profundidade, nessa ordem.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import struct
import time
import uuid
from collections.abc import Callable, Sequence
from typing import Any

import gpu
import numpy as np
from comum import gravar_ply, ler_ply
from scipy.optimize import minimize

SIG_REVO = b"RPAlgo_file_sys\x00"
SIG_SQL = b"SQLite format 3\x00"
MAX_VOXELS = 3e8  # voxels da grade da peca (bool + rotulos int32 = ~5 bytes cada)
MAX_VOXELS_TRANS = 1e8  # grade dos transitorios (2 float32 + filtros)
_CPUS = os.cpu_count() or 1
# threads da parte que ficou na CPU (Open3D): REVO_THREADS (definido por recursos.orcamento_threads); sem ele, todos os
# processadores (modo maximo) ou 12 (REVO_RECURSOS=conservador, guia secao 8)
_TETO = 12 if os.environ.get("REVO_RECURSOS", "maximo").lower() == "conservador" else _CPUS
THREADS = int(os.environ.get("REVO_THREADS", str(_TETO)))
THREADS = _TETO if THREADS <= 0 else min(THREADS, _CPUS)
K_PADRAO = (1140.4, 1141.0, 232.8, 319.5)  # modo usado nos scans de teste


# ---------------------------------------------------------------- GPU: conversoes e indice de vizinhos
def _eh_cupy(a):
    return type(a).__module__.split(".")[0] == "cupy"


def _g(a, dtype=None):
    """numpy/cupy -> cupy (uma transferencia; ja em cupy nao copia)."""
    return gpu.para_gpu(a, dtype)


def _saida(x, na_gpu):
    """Devolve `x` (cupy) no tipo da entrada original: cupy se a entrada era cupy, senao numpy."""
    return x if na_gpu else gpu.para_cpu(x)


class ArvoreGPU:
    """Substituto de cKDTree na GPU: `query(Q, distance_upper_bound=inf)` -> (dist, idx).

    Diferenca em relacao ao cKDTree: onde nao ha vizinho dentro do limite, dist = inf e idx = -1 (cKDTree devolve n).
    Q numpy -> saida numpy; Q cupy -> saida cupy. Indices de grade ficam em cache por limite (cada limite = celula).
    """

    def __init__(self, pts: Any) -> None:
        self.pts = _g(pts, np.float64).reshape(-1, 3)
        self.n = len(self.pts)
        self._ind = {}

    def query(self, Q: Any, distance_upper_bound: Any = np.inf, workers: Any = None) -> tuple[Any, Any]:
        na_gpu = _eh_cupy(Q)
        m = gpu.xp()
        Qg = _g(Q, np.float64).reshape(-1, 3)
        if self.n == 0 or len(Qg) == 0:
            d = m.full(len(Qg), m.inf)
            j = m.full(len(Qg), -1, dtype=m.int64)
        elif np.isfinite(distance_upper_bound):
            ub = float(distance_upper_bound)
            if ub not in self._ind:
                if len(self._ind) >= 4:
                    self._ind.pop(next(iter(self._ind)))
                self._ind[ub] = gpu.Indice(self.pts, ub)
            d, j = self._ind[ub].mais_proximo(Qg)
            fora = ~(d < ub)  # como o cKDTree: estritamente menor que o limite
            d, j = m.where(fora, m.inf, d), m.where(fora, -1, j)
        else:
            d, j = gpu.vizinho_mais_proximo(self.pts, Qg, np.inf)
        return _saida(d, na_gpu), _saida(j, na_gpu)


def arvore(pts: Any) -> ArvoreGPU:
    """ArvoreGPU de `pts` (n, 3)."""
    return ArvoreGPU(pts)


def _consultar(arv, P, limite):
    """Distancia ao vizinho mais proximo (<= limite) de P em `arv` (ArvoreGPU, ou cKDTree legado de testes). P cupy ->
    resultado cupy."""
    if isinstance(arv, ArvoreGPU):
        return arv.query(P, distance_upper_bound=limite)[0]
    d = arv.query(gpu.para_cpu(P), distance_upper_bound=limite, workers=THREADS)[0]
    return _g(d)


# ---------------------------------------------------------------- utilidades


def plano_ransac(P: Any, thr: float = 1.0, it: int = 3000, seed: int = 0) -> tuple[np.ndarray, float, float]:
    """Plano dominante (n, d, fracao) por RANSAC na GPU (gpu.ransac_plano: todas as hipoteses em paralelo).

    Mesmo contrato do antigo laco em numpy; a sequencia aleatoria e outra (gerador do CuPy), entao o plano sai igual
    so ate o ruido do RANSAC.
    """
    return gpu.ransac_plano(P, limiar=thr, iteracoes=it, semente=seed)


# ---------------------------------------------------------------- leitura
class RevoScan:
    def __init__(
        self, pasta_scan: str | os.PathLike[str], workdir: str | os.PathLike[str], reusar: bool = False
    ) -> None:
        self.pasta = pasta_scan
        self.nome = os.path.basename(os.path.normpath(pasta_scan))
        src = os.path.join(pasta_scan, "cache", "frames.dataset")
        with open(src, "rb") as arquivo:
            self.sig = arquivo.read(16)
        if self.sig != SIG_REVO:
            raise ValueError(f"{self.nome}: assinatura inesperada {self.sig!r}")
        self.db = os.path.join(workdir, self.nome + ".sqlite")
        if not (reusar and os.path.exists(self.db) and os.path.getsize(self.db) == os.path.getsize(src)):
            shutil.copy(src, self.db)
            with open(self.db, "r+b") as h:
                h.write(SIG_SQL)
        self.c = sqlite3.connect(self.db)
        rows = self.c.execute("select id, bin from FrameInfo").fetchall()
        com_prof = set(r[0] for r in self.c.execute("select id from FrameRGBD"))
        rows.sort(key=lambda r: (r[0] >> 32, r[0] & 0xFFFFFFFF))
        self.ids = [r[0] for r in rows]
        self.hdr = np.array([struct.unpack("<IIIi", r[1][:16]) for r in rows])
        self.modo = (self.hdr[:, 1] >> 16) & 0xFF
        self.T = np.array([np.frombuffer(r[1][16:144], "<f8").reshape(4, 4) for r in rows])
        self.tem_prof = np.array([i in com_prof for i in self.ids])
        # Quadros com mascara e profundidade inconsistentes (n do cabecalho != pixels da mascara)
        # nao sao confiaveis: ficam como sem profundidade e a limpeza nao mexe neles.
        self.inconsistentes = []
        for k, i in enumerate(self.ids):
            if not self.tem_prof[k]:
                continue
            d = self.c.execute("select data from FrameRGBD where id=?", (i,)).fetchone()[0]
            w, h, n = struct.unpack("<III", d[:12])
            nb = w * h // 8
            pop = int(np.unpackbits(np.frombuffer(d[12 : 12 + nb], np.uint8)).sum())
            if pop != n or len(d) < 12 + nb + 4 + 4 * n:
                self.tem_prof[k] = False
                self.inconsistentes.append(k)
        self.K = {}

    def profundidade(self, i: int) -> tuple[int, int, np.ndarray, np.ndarray, bytes] | None:
        r = self.c.execute("select data from FrameRGBD where id=?", (i,)).fetchone()
        if r is None:
            return None
        d = r[0]
        w, h, n = struct.unpack("<III", d[:12])
        nb = w * h // 8
        mask = np.unpackbits(np.frombuffer(d[12 : 12 + nb], np.uint8), bitorder="little").astype(bool)
        z = np.frombuffer(d[12 + nb + 4 : 12 + nb + 4 + 4 * n], "<f4")
        return w, h, mask, z, d[12 + nb : 12 + nb + 4]

    def pontos(
        self, k: int, K: Any = None, passo: int = 1, mundo: bool = True, na_gpu: bool = False, dados: Any = None
    ) -> tuple[Any, Any]:
        """Pontos 3D (n, 3) float64 do quadro k e os indices lineares dos pixels (numpy). A projecao roda na GPU;
        `na_gpu=True` devolve os pontos em cupy (sem voltar para a CPU). `dados` = retorno de `profundidade`
        ja decodificado (evita decodificar o blob duas vezes)."""
        r = self.profundidade(self.ids[k]) if dados is None else dados
        if r is None:
            return None, None
        w, h, mask, z, _ = r
        pix = np.flatnonzero(mask)
        if passo > 1:
            sel = np.arange(0, len(pix), passo)
            pix, z = pix[sel], z[sel]
        m = gpu.xp()
        v, u = m.divmod(_g(pix), w)
        fx, fy, cx, cy = K if K is not None else self.K[int(self.modo[k])]
        z = _g(z, np.float64)
        P = m.stack([(u - cx) * z / fx, -(v - cy) * z / fy, -z], axis=1)
        if mundo:
            T = self.T[k]
            P = P @ _g(T[:3, :3].T) + _g(T[:3, 3])
        return (P if na_gpu else gpu.para_cpu(P)), pix

    def foco_gravado(self, modo: int) -> float | None:
        """Foco do mapa de profundidade a partir de param*/Pl.bin (a camera estereo tem o dobro da resolucao)."""
        pasta = os.path.join(self.pasta, "param" if modo == 0 else f"param{modo}", "Pl.bin")
        try:
            f = float(np.frombuffer(open(pasta, "rb").read(), "<f4")[1]) / 2
            return f if 300 < f < 3000 else None
        except (OSError, IndexError, ValueError):
            return None

    def calibrar(self, modo: int, x0: Any = None, nquadros: int = 8) -> tuple[tuple[float, ...] | None, float | None]:
        """Ajusta fx, fy, cx, cy contra o raw_preview.ply. Retorna (K, erro mediano em mm).

        O custo (distancia media, truncada em 3 mm, dos pontos dos quadros ao preview) roda inteiro na GPU: indice do
        preview construido uma vez e todos os quadros projetados numa unica chamada por avaliacao. So o Nelder-Mead do
        scipy (otimizador escalar de 3 a 4 parametros) fica na CPU.
        """
        m = gpu.xp()
        _, prev = ler_ply(os.path.join(self.pasta, "raw_preview.ply"))
        prev = prev[:, :3].astype(np.float64)
        arv = gpu.Indice(prev, 3.0)  # limite 3 mm = teto do custo: sem vizinho o erro vale 3
        ks = [k for k in range(len(self.ids)) if self.modo[k] == modo and self.tem_prof[k]]
        if not ks:
            return None, None
        ks = [ks[j] for j in np.linspace(0, len(ks) - 1, min(nquadros, len(ks))).astype(int)]
        us, vs, zs, qs = [], [], [], []
        for q, k in enumerate(ks):
            w, h, mask, z, _ = self.profundidade(self.ids[k])
            v, u = np.divmod(np.flatnonzero(mask), w)
            us.append(u)
            vs.append(v)
            zs.append(z.astype(np.float64))
            qs.append(np.full(len(u), q))
        R = _g(np.array([self.T[k][:3, :3] for k in ks]))  # (q, 3, 3)
        t = _g(np.array([self.T[k][:3, 3] for k in ks]))

        def conjunto(passo):  # um unico lote com os pixels [::passo] de cada quadro
            sel = [np.arange(0, len(u), passo) for u in us]
            u_ = _g(np.concatenate([u[s] for u, s in zip(us, sel)]), np.float64)
            v_ = _g(np.concatenate([v[s] for v, s in zip(vs, sel)]), np.float64)
            z_ = _g(np.concatenate([z[s] for z, s in zip(zs, sel)]))
            q_ = _g(np.concatenate([q[s] for q, s in zip(qs, sel)]))
            return u_, v_, z_, R[q_], t[q_]

        def mundo(c, p):
            u, v, z, Rp, tp = c
            fx, fy, cx, cy = p
            Pc = m.stack([(u - cx) * z / fx, -(v - cy) * z / fy, -z], axis=1)
            return m.einsum("nij,nj->ni", Rp, Pc) + tp  # R @ Pc por ponto

        busca = conjunto(12)

        def custo(p):
            d, _ = arv.mais_proximo(mundo(busca, p))
            return float(m.minimum(d, 3.0).mean())

        if x0 is None:
            # Varias partidas: o foco gravado no scan (ou o padrao) e centros tipicos. Foco unico (fx=fy)
            # primeiro, para nao cair em minimo local; depois ajuste livre a partir do melhor.
            f0 = self.foco_gravado(modo) or K_PADRAO[0]
            melhor = None
            for cx0, cy0 in ((232.8, 319.5), (324.0, 327.0), (400.0, 300.0)):
                r0 = minimize(
                    lambda q: custo((q[0], q[0], q[1], q[2])),
                    (f0, cx0, cy0),
                    method="Nelder-Mead",
                    options=dict(maxiter=150, xatol=0.1, fatol=1e-4),
                )
                if melhor is None or r0.fun < melhor.fun:
                    melhor = r0
            x0 = (melhor.x[0], melhor.x[0], melhor.x[1], melhor.x[2])
        r = minimize(custo, x0, method="Nelder-Mead", options=dict(maxiter=400, xatol=0.05, fatol=1e-5))
        # erro final: distancia ao vizinho mais proximo SEM teto, a cada 3 pixels
        d, _ = gpu.vizinho_mais_proximo(prev, mundo(conjunto(3), r.x), np.inf)
        self.K[modo] = tuple(r.x)
        return tuple(r.x), float(m.median(d))


# ---------------------------------------------------------------- ndimage na GPU
def _nd():
    """cupyx.scipy.ndimage (GPU); scipy.ndimage so no modo de depuracao em CPU (REVO_PERMITIR_CPU=1)."""
    if gpu.em_gpu():
        import cupyx.scipy.ndimage as nd
    else:
        from scipy import ndimage as nd
    return nd


def _morf(nome, x, **kw):
    """Operacao morfologica binaria do ndimage; na GPU as iteracoes usam o modo 'brute_force' (resultado identico)."""
    if gpu.em_gpu() and nome in ("binary_dilation", "binary_erosion", "binary_closing", "binary_opening"):
        kw["brute_force"] = True
    return getattr(_nd(), nome)(x, **kw)


def _uns(*forma):
    return gpu.xp().ones(forma, bool)


def _limites(mask):
    """(lo, hi) inclusivos (numpy int) da caixa que envolve os True de `mask` (reducoes por eixo, sem argwhere)."""
    m = gpu.xp()
    lo, hi = [], []
    for ax in range(mask.ndim):
        idx = m.flatnonzero(mask.any(axis=tuple(i for i in range(mask.ndim) if i != ax)))
        lo.append(int(idx[0]))
        hi.append(int(idx[-1]))
    return np.array(lo), np.array(hi)


# ---------------------------------------------------------------- persistencia (espaco livre)
def persistencia(
    scan: RevoScan | None,
    centros: Any,
    voxel: float,
    nquadros: int = 120,
    tol_hit: float | None = None,
    tol_free: float | None = None,
) -> tuple[Any, Any]:
    """Conta, por voxel, em quantos quadros ele foi visto ocupado (hit) e em quantos foi visto
    VAZIO (o raio da camera atravessa o voxel e acha superficie mais atras: free).

    Mesa giratoria: a peca fica parada no referencial do scan e e vista ocupada em todo quadro que a
    enxerga. Braco, mao, operador e fundo nao giram com a mesa: nesse referencial eles aparecem em
    poucos quadros e, nos demais, o espaco onde estariam foi visto vazio.

    Roda na GPU; `centros` numpy ou cupy (a saida segue o tipo da entrada).
    """
    tol_hit = tol_hit or 1.0 * voxel
    tol_free = tol_free or 1.75 * voxel
    m = gpu.xp()
    na_gpu = _eh_cupy(centros)
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    ks = [ks[j] for j in np.linspace(0, len(ks) - 1, min(nquadros, len(ks))).astype(int)]
    C = _g(centros, np.float32)
    hits = m.zeros(len(C), m.int32)
    frees = m.zeros(len(C), m.int32)
    for k in ks:
        r = scan.profundidade(scan.ids[k])
        if r is None:
            continue
        w, h, mask, z, _ = r
        D = m.full(w * h, m.nan, m.float32)
        D[_g(mask)] = _g(z, np.float32)
        D = D.reshape(h, w)
        fx, fy, cx, cy = scan.K[int(scan.modo[k])]
        T = scan.T[k]
        Pc = (C - _g(T[:3, 3], np.float32)) @ _g(T[:3, :3], np.float32)
        zc = -Pc[:, 2]
        ok = zc > 20
        zs = m.where(ok, zc, 1.0)
        u = m.rint(Pc[:, 0] * fx / zs + cx).astype(m.int32)
        v = m.rint(-Pc[:, 1] * fy / zs + cy).astype(m.int32)
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = m.full(len(C), m.nan, m.float32)
        d[ok] = D[v[ok], u[ok]]
        diff = d - zc
        hits += m.abs(diff) <= tol_hit
        frees += diff > tol_free
    return _saida(hits, na_gpu), _saida(frees, na_gpu)


# ---------------------------------------------------------------- segmentacao da peca
def _caixa(mask, pad):
    """Fatias da caixa que envolve os True de `mask`, com folga `pad` (limitada a grade)."""
    lo, hi = _limites(mask)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def cruzamento_base(A: Any, n: Any, d: Any, celula: float = 2.0, dens_rel: float = 0.25, folga: float = 3.0) -> float:
    """Fracao das celulas ocupadas pela peca (acima do plano) que tambem tem material DENSO abaixo dele.

    Uma mesa/prato de verdade deixa a peca so de um lado: ~0. Um plano que fatia a propria peca
    (busto, peca sem mesa) tem corpo dos dois lados: valores altos. Material denso = pelo menos
    `dens_rel` da densidade mediana da peca na mesma celula (ruido espalhado nao conta).
    """
    m = gpu.xp()
    A = _g(A, np.float64)
    n = np.asarray(n, float)
    h = A @ _g(n) + d
    u = np.cross(n, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    w = np.cross(n, u)
    cel = m.floor(m.stack([A @ _g(u), A @ _g(w)], axis=1) / celula).astype(m.int64)
    key = cel[:, 0] * 1000003 + cel[:, 1]
    ka, ca = m.unique(key[h > folga], return_counts=True)
    kb, cb = m.unique(key[h < -folga], return_counts=True)
    if len(ka) == 0:
        return 0.0
    kb = kb[cb >= dens_rel * float(m.median(ca))]
    return float(m.isin(kb, ka).sum() / len(ka))


def segmentar_peca(
    scan: RevoScan | None,
    voxel: float = 1.0,
    faixa_base: float = 1.2,
    passo: int = 6,
    log: Callable[..., None] = print,
    min_votos_frac: float = 0.5,
    min_tam_frac: float = 0.05,
    ext_min_mm: float = 15.0,
    colado_mm: float = 5.0,
    margem_mm: float = 4.0,
    transitorio: bool = True,
    trans_min_free: int = 3,
    trans_razao: float = 1.0,
    trans_viz: int = 5,
    trans_dens: float = 0.6,
    amostra: Any = None,
    modo_base: str = "sem",
    cruza_max: float = 0.1,
    selecao: str = "hibrido",
    dominancia: float = 0.5,
    periferia_mm: float = 0.0,
    per_altura: float = 15.0,
    per_banda: float = 2.5,
    per_abertura: float = 6.0,
    per_alargue: float = 20.0,
    grandes_frac: float = 0.02,
    per_corpo_frac: float = 0.1,
    ext_max_mm: float = 0.0,
    so_peca: bool = False,
) -> dict[str, Any]:
    """Separa a peca do prato e das pecas de apoio (tudo na GPU: grades densas em cupy + cupyx.scipy.ndimage).

    1. Funde uma amostra de todos os quadros no referencial do mundo.
    2. Acha o plano da base (prato) por RANSAC.
    3. Agrupa os pontos acima da base em componentes conexas (grade de voxels).
    4. A peca e a componente para onde a camera mais mira ao longo da captura
       (ponto no eixo optico, na profundidade mediana de cada quadro).
    Retorna um dicionario com o plano, a grade e o conjunto de voxels a manter (numpy; as copias na GPU ficam em
    seg["_g"] para `mascara_manter`).
    """
    m = gpu.xp()
    nd = _nd()
    if amostra is not None:
        A, miras = amostra
    else:
        ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
        lista, miras = [], []
        for k in ks[:: max(1, len(ks) // 150)]:
            P, _ = scan.pontos(k, passo=passo, na_gpu=True)
            if P is None or len(P) < 100:
                continue
            lista.append(P)
            T = scan.T[k]
            eixo = T[:3, :3] @ np.array([0, 0, -1.0])
            dcam = float(m.median(m.linalg.norm(P - _g(T[:3, 3]), axis=1)))
            miras.append(T[:3, 3] + eixo * dcam)
        A = m.vstack(lista)
    A = _g(A, np.float64)
    n, d, frac = plano_ransac(A[:: max(1, len(A) // 40000)], thr=1.0, it=800)
    # O plano e mesa/prato de verdade ou uma fatia da propria peca? Cada projeto e diferente.
    cruza = cruzamento_base(A, n, d)
    sem_base = modo_base == "sem" or (modo_base == "auto" and cruza > cruza_max)
    d_ref = d  # plano real (RANSAC), mesmo quando nao e usado como base
    ng = _g(n)
    if sem_base:
        d = -float((A @ ng).min()) + 10.0  # plano virtual abaixo de tudo: nada e cortado por ele
    h = A @ ng + d
    acima = A[h > faixa_base]
    log(
        f"   base: {frac * 100:.0f}% da amostra no plano | cruzamento {cruza * 100:.0f}% -> "
        f"{'SEM base (plano atravessa a peca)' if sem_base else 'mesa/prato valido'} | {len(acima):,} pontos acima"
    )

    # Objetos transitorios (braco, mao, operador): vistos vazios em mais quadros do que ocupados.
    trans = None
    if transitorio and scan is not None:
        v2 = 2.0
        amn, amx = gpu.para_cpu(acima.min(0)), gpu.para_cpu(acima.max(0))
        while float(np.prod((amx - amn + 4 * v2) / v2)) > MAX_VOXELS_TRANS:
            v2 *= 1.25
        lo2 = amn - 2 * v2
        idx2 = m.floor((acima - _g(lo2)) / v2).astype(m.int64)
        dims2 = [int(x) for x in gpu.para_cpu(idx2.max(0)) + 1]
        chave = (idx2[:, 0] * dims2[1] + idx2[:, 1]) * dims2[2] + idx2[:, 2]
        uk, inv, _ = gpu.agrupar(chave)  # chaves ordenadas = ordem lexicografica (x, y, z)
        ix, iy, iz = uk // (dims2[1] * dims2[2]), (uk // dims2[2]) % dims2[1], uk % dims2[2]
        uniq = m.stack([ix, iy, iz], axis=1)
        hits, frees = persistencia(scan, (uniq + 0.5) * v2 + _g(lo2), v2)
        marc = (frees >= trans_min_free) & (frees > trans_razao * hits)
        # Na peca os voxels marcados sao esparsos (bordas e cantos vistos de perto). Objeto
        # transitorio e uma regiao inteira marcada: so vale quem esta cercado por marcados.
        occ_g = m.zeros(tuple(dims2), m.float32)
        occ_g[ix, iy, iz] = 1
        mar_g = m.zeros(tuple(dims2), m.float32)
        mar_g[ix, iy, iz] = marc.astype(m.float32)
        viz = nd.uniform_filter(mar_g, size=trans_viz, mode="constant") / m.maximum(
            nd.uniform_filter(occ_g, size=trans_viz, mode="constant"), 1e-6
        )
        t_v = marc & (viz[ix, iy, iz] >= trans_dens)
        del occ_g, mar_g, viz
        trans = dict(
            lo=lo2,
            voxel=v2,
            uniq=gpu.para_cpu(uniq),
            flag=gpu.para_cpu(t_v),
            hits=gpu.para_cpu(hits),
            frees=gpu.para_cpu(frees),
        )
        log(f"   transitorio: {int(t_v.sum()):,} de {len(t_v):,} voxels de 2 mm vistos vazios (braco/mao/operador)")
        acima = acima[~t_v[inv.ravel()]]
        log(f"   pontos acima apos remover transitorios: {len(acima):,}")

    # Cenas grandes (1 m ou mais, ou pontos soltos longe) viram grades de bilhoes de voxels de 1 mm e
    # estouram a RAM (36 GB medidos num processo). Acima de MAX_VOXELS o voxel cresce ate caber.
    voxel0 = voxel
    amn, amx = gpu.para_cpu(acima.min(0)), gpu.para_cpu(acima.max(0))
    while float(np.prod((amx - amn + 6 * voxel) / voxel)) > MAX_VOXELS:
        voxel *= 1.25
    if voxel != voxel0:
        log(f"   cena grande: voxel {voxel0:g} -> {voxel:.2f} mm para a grade caber na memoria")
    lo = amn - 3 * voxel
    dims = np.ceil((amx + 3 * voxel - lo) / voxel).astype(int)
    idx = m.floor((acima - _g(lo)) / voxel).astype(m.int64)
    ii = (idx[:, 0], idx[:, 1], idx[:, 2])
    grade = m.zeros(tuple(int(x) for x in dims), bool)
    grade[ii] = True
    rot, ncomp = nd.label(grade, structure=_uns(3, 3, 3))
    del grade
    ncomp = int(ncomp)
    tam = m.bincount(rot[ii], minlength=ncomp + 1)
    if ext_max_mm > 0:
        # gabarito/placa de marcadores maior que a peca: sai da disputa (nunca e peca nem parte dela)
        grande = [
            c
            for c, sl in enumerate(nd.find_objects(rot), start=1)
            if sl is not None and max(x.stop - x.start for x in sl) * voxel > ext_max_mm
        ]
        if grande:
            log(f"   ext-max-mm {ext_max_mm:g}: {len(grande)} componente(s) maiores ignorados {grande[:8]}")
            tam[_g(grande)] = 0
            lut = m.arange(ncomp + 1, dtype=m.int32)
            lut[_g(grande)] = 0
            rot = lut[rot]
    tam = gpu.para_cpu(tam)

    # mira da camera -> componente
    occ = m.argwhere(rot > 0)
    arv = ArvoreGPU(occ * voxel + _g(lo) + voxel / 2)
    dist, j = arv.query(np.array(miras), distance_upper_bound=8.0)
    ok = np.isfinite(dist)
    oj = occ[_g(j[ok])]
    comps = rot[oj[:, 0], oj[:, 1], oj[:, 2]]
    votos = gpu.para_cpu(m.bincount(comps, minlength=ncomp + 1))
    del occ, arv, oj
    ordem = np.argsort(-tam)
    rp = rot[ii]  # rotulo de cada ponto acima da base
    log("   objetos acima da base (pontos | votos de mira):")
    for c in ordem[:6]:
        if c == 0 or tam[c] == 0:
            continue
        pts = acima[rp == c]
        ext = gpu.para_cpu(pts.max(0) - pts.min(0))
        log(
            f"     objeto {c}: {tam[c]:,} pts | {votos[c]} miras | extensao {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm"
        )
    maior = int(ordem[0] if ordem[0] else ordem[1])
    por_votos = int(np.argmax(votos)) if votos.max() > 0 else maior
    # selecao: "votos" (onde a camera mais mira), "maior" (maior componente) ou "hibrido": o maior
    # vence quando domina a cena (>= `dominancia` dos pontos acima da base), senao decidem os votos.
    if selecao == "maior":
        peca = maior
    elif selecao == "hibrido":
        peca = maior if tam[maior] >= dominancia * tam[1:].sum() else por_votos
    else:
        peca = por_votos
    # A peca pode se partir em varias componentes (miolo separado do anel por uma fresta
    # estreita). Mantem toda componente grande e bem mirada, nao so a mais votada.
    # Marcadores e apoios pequenos tem ~0 votos e ficam de fora.
    if so_peca:
        ext_min_mm = 1e9  # nada "colado" entra: so o componente escolhido
    partes = [
        c
        for c in range(1, ncomp + 1)
        if c == peca
        or (votos[c] >= min_votos_frac * votos[peca] and tam[c] >= min_tam_frac * tam[peca])
        # a peca costuma aparecer partida em varios componentes grandes (amostra esparsa, fendas):
        # todo componente grande em relacao ao maior tambem entra, com ou sem voto de mira
        or (grandes_frac > 0 and tam[c] >= grandes_frac * tam[maior])
    ]
    if so_peca:
        partes = [peca]

    # Componente grande colado na peca (ex.: cubo central separado por uma fresta) tambem e peca,
    # mesmo sem voto de mira. Marcadores sao pequenos (~6 mm) e ficam de fora.
    # Em cena suja (braco, blocos encostados) so entra o que esta DENTRO da caixa da peca, como o
    # cubo central; objeto que se estende para fora dela e do ambiente, nao da peca.
    # A distancia so interessa perto da peca: calcula numa caixa em volta dela (cenas grandes tem
    # grades de centenas de milhoes de voxels e a transformada de distancia inteira leva minutos).
    def so_partes(rotulos):
        tab = m.zeros(ncomp + 1, bool)
        tab[_g(partes)] = True
        return tab[rotulos]

    pm = so_partes(rot)
    caixa = _caixa(pm, int(np.ceil(colado_mm / voxel)) + 2)
    dist_sub = nd.distance_transform_edt(~pm[caixa])
    nuc_lo, nuc_hi = _limites(pm)
    nlo, nhi = nuc_lo - int(round(3 / voxel)), nuc_hi + int(round(3 / voxel))
    del pm
    for c, sl in enumerate(nd.find_objects(rot), start=1):
        if c in partes or sl is None:
            continue
        ext = max(s.stop - s.start for s in sl) * voxel
        if ext < ext_min_mm:
            continue
        inter = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(sl, caixa))
        if any(i.stop <= i.start for i in inter):
            continue  # longe da peca: nao esta colado
        loc = tuple(slice(i.start - b.start, i.stop - b.start) for i, b in zip(inter, caixa))
        no_c = rot[inter] == c
        if not bool(no_c.any()) or float(dist_sub[loc][no_c].min()) * voxel > colado_mm:
            continue
        vc = m.argwhere(rot[sl] == c) + _g([x.start for x in sl])
        dentro = float(((vc >= _g(nlo)) & (vc <= _g(nhi))).all(axis=1).mean())
        if dentro >= 0.9:
            partes.append(c)
    del dist_sub
    # Limpeza branda: mantem tudo que esta a ate `margem_mm` da peca e remove so o que esta longe
    # (marcadores, prato distante, apoios).
    nucleo = so_partes(rot)
    cx = _caixa(nucleo, int(np.ceil(margem_mm / voxel)) + 2)
    manter = m.zeros(nucleo.shape, bool)
    manter[cx] = nd.distance_transform_edt(~nucleo[cx], sampling=voxel) <= margem_mm
    # Pegada da peca no plano da base: sob ela, a face de baixo encosta no prato e nao e cortada.
    u = np.cross(n, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    centros = m.argwhere(nucleo) * voxel + _g(lo) + voxel / 2
    del nucleo
    uv = m.stack([centros @ _g(u), centros @ _g(v)], axis=1)
    uv_lo = gpu.para_cpu(uv.min(0)) - 3
    uv_hi = gpu.para_cpu(uv.max(0))
    pegada = m.zeros(tuple(int(x) for x in (np.ceil((uv_hi + 3 - uv_lo) / 1.0).astype(int) + 1)), bool)
    ij = m.floor((uv - _g(uv_lo)) / 1.0).astype(m.int64)
    jj = (ij[:, 0], ij[:, 1])
    pegada[jj] = True
    # A peca e vazada (cavidades, anel): o piso dentro dela fica na altura do prato e e peca, nao prato.
    # Fecha fendas e preenche os buracos da projecao.
    pegada = _morf("binary_dilation", pegada, iterations=1)
    pegada = _morf("binary_closing", m.pad(pegada, 8), iterations=6)[8:-8, 8:-8]
    pegada = nd.binary_fill_holes(pegada)
    # Altura local da peca em cada celula da pegada. A amostra dos quadros e esparsa e a casca fina
    # da face de cima fica com buracos no nucleo 3D; a coluna vertical sob a pegada nao tem esse problema.
    hcent = centros @ ng + d
    hcol = m.full(pegada.shape, -m.inf)
    m.maximum.at(hcol, jj, hcent)
    hcol = nd.maximum_filter(hcol, size=9)
    vazio = ~m.isfinite(hcol) & pegada
    for _ in range(20):  # propaga para celulas da pegada sem nucleo (buracos)
        if not bool(vazio.any()):
            break
        hcol = m.where(vazio, nd.maximum_filter(hcol, size=5), hcol)
        vazio = ~m.isfinite(hcol) & pegada
    hcol[~m.isfinite(hcol)] = float(hcent.max())
    # Periferia da mesa: no modo sem base, a faixa do plano real LONGE da peca e mesa (so quando a peca
    # tem altura de verdade acima do plano; peca achatada nao tem como distinguir de mesa).
    per = None
    per_g = None
    if sem_base and periferia_mm > 0 and cruza > cruza_max:
        log(
            f"   periferia da mesa DESLIGADA: o plano atravessa a peca (cruzamento {cruza * 100:.0f}% > {cruza_max * 100:.0f}%), nao e mesa"
        )
    elif sem_base and periferia_mm > 0:
        h_real = centros @ ng + d_ref
        alta = h_real > 4.0
        # So vale se a peca tem corpo de verdade: >= per_corpo_frac do volume mantido acima de `per_altura`.
        # Peca achatada (ou plano que nao e mesa) tem quase tudo rente ao plano e a regra a comeria inteira.
        corpo_frac = float((h_real > per_altura).mean())
        if bool(alta.any()) and corpo_frac < per_corpo_frac:
            log(
                f"   periferia da mesa DESLIGADA: so {corpo_frac * 100:.0f}% da peca passa de {per_altura:g} mm acima do plano (peca achatada)"
            )
        elif bool(alta.any()) and (float(h_real.max()) >= per_altura):
            # Pegada = so o CORPO da peca. O componente da peca costuma vir colado na mesa, e relevos e
            # marcadores da mesa (>4 mm) entrariam na pegada e a protegeriam. A abertura apaga o que e fino
            # e o corpo e o que tem altura real; so as manchas 2D que contem corpo ficam.
            fp = m.zeros(pegada.shape, bool)
            ia = ij[alta]
            fp[ia[:, 0], ia[:, 1]] = True
            corpo = m.zeros(pegada.shape, bool)
            ic = ij[h_real > per_altura]
            corpo[ic[:, 0], ic[:, 1]] = True
            r = int(round(per_abertura))
            yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
            fp = _morf("binary_opening", m.pad(fp, r + 2), structure=_g((xx**2 + yy**2) <= r * r))[
                r + 2 : -(r + 2), r + 2 : -(r + 2)
            ]
            # relevo baixo (4 a `per_altura` mm) so conta como peca ate `per_alargue` mm do corpo: cobre bases
            # alargadas sem deixar o relevo da mesa, que encosta no corpo, proteger a mesa inteira
            perto = nd.distance_transform_edt(~corpo) <= per_alargue
            fp = (fp & perto) | _morf("binary_dilation", corpo, iterations=1)  # o corpo nunca sai da pegada
            fp = _morf("binary_closing", m.pad(fp, 8), iterations=4)[8:-8, 8:-8]
            dist2 = nd.distance_transform_edt(~fp).astype(m.float32)
            per = dict(n=n, d=d_ref, uv_lo=uv_lo, dist=gpu.para_cpu(dist2), mm=periferia_mm, banda=per_banda)
            per_g = dist2
    if sem_base:
        pegada = m.zeros_like(pegada)  # sem prato nao ha "piso" nem "coluna" sob a peca
    exts = {}
    for c in partes:
        lo_c, hi_c = _limites(rot == c)
        exts[c] = tuple(int(round(float(x) * voxel)) for x in (hi_c - lo_c))
    log(
        f"   peca escolhida: objeto {peca} ({votos[peca]} de {ok.sum()} miras) | componentes mantidos: {partes} "
        f"(mm {exts}, votos {[int(votos[c]) for c in partes]}) | margem {margem_mm:g} mm"
    )
    h_max = float((centros @ ng + d).max() + voxel + 1.0)  # marcadores colados ficam acima do topo
    seg = dict(
        n=n,
        d=d,
        faixa_base=faixa_base,
        voxel=voxel,
        lo=lo,
        manter=gpu.para_cpu(manter),
        sem_base=bool(sem_base),
        cruza=cruza,
        frac_plano=float(frac),
        periferia=per,
        u=u,
        v=v,
        uv_lo=uv_lo,
        pegada=gpu.para_cpu(pegada),
        hcol=gpu.para_cpu(hcol),
        trans=trans,
        h_max=h_max,
        peca=peca,
        votos=votos.tolist(),
        tam=tam.tolist(),
    )
    # copias na GPU para mascara_manter (evita subir as grades de novo a cada quadro)
    seg["_g"] = {"manter": (seg["manter"], manter), "pegada": (seg["pegada"], pegada), "hcol": (seg["hcol"], hcol)}
    if per is not None:
        seg["_g"]["per_dist"] = (per["dist"], per_g)
    del rot, centros, uv, ij, acima, A
    gpu.liberar_memoria()
    return seg


def _seg_g(seg, nome, fonte=None, dtype=None):
    """Copia GPU (em cache em seg['_g']) de um array de `seg` (ou de `fonte`); refaz se o array foi trocado."""
    cache = seg.setdefault("_g", {})
    arr = seg[nome] if fonte is None else fonte
    ref = cache.get(nome)
    if ref is None or ref[0] is not arr:
        ref = (arr, _g(arr, dtype))
        cache[nome] = ref
    return ref[1]


def mascara_manter(P: Any, seg: dict[str, Any]) -> Any:
    """Mascara bool (True = fica) dos pontos P (n, 3) segundo `seg` de `segmentar_peca`. Roda na GPU; P numpy ou cupy
    (a saida segue o tipo da entrada)."""
    m = gpu.xp()
    na_gpu = _eh_cupy(P)
    P = _g(P, np.float64)
    ng = _g(seg["n"], np.float64)
    h = P @ ng + float(seg["d"])
    manter = _seg_g(seg, "manter")
    forma = _g(manter.shape)
    idx = m.floor((P - _g(seg["lo"], np.float64)) / seg["voxel"]).astype(m.int64)
    dentro = ((idx >= 0) & (idx < forma)).all(axis=1)
    ic = m.clip(idx, 0, forma - 1)
    ok = manter[ic[:, 0], ic[:, 1], ic[:, 2]] & dentro
    # Perto da base: so vale sob a pegada da peca (corpo encostado no prato); fora dela e prato.
    ij = m.floor(m.stack([P @ _g(seg["u"]), P @ _g(seg["v"])], axis=1) - _g(seg["uv_lo"])).astype(m.int64)
    pegada = _seg_g(seg, "pegada")
    forma2 = _g(pegada.shape)
    na_pegada = ((ij >= 0) & (ij < forma2)).all(axis=1)
    ic2 = m.clip(ij, 0, forma2 - 1)
    na_pegada &= pegada[ic2[:, 0], ic2[:, 1]]
    acima = h > seg["faixa_base"]
    # Sob a pegada e junto do prato e a face de baixo/piso da peca: vale mesmo longe do nucleo em 3D.
    piso = na_pegada & (h > -1.5) & (h <= seg["faixa_base"])
    # Coluna vertical sob a pegada ate a altura local da peca (+3 mm): vale mesmo onde o nucleo 3D
    # tem buracos (face de cima fina). Braco/objeto que passa mais alto fica de fora.
    hl = _seg_g(seg, "hcol", dtype=np.float64)[ic2[:, 0], ic2[:, 1]]
    coluna = na_pegada & (h > -1.5) & (h <= hl + 3.0)
    ok = (ok & acima | piso | coluna | (ok & na_pegada & (h > -1.5))) & (h < seg["h_max"] + 2.0)
    if seg.get("trans") is not None:
        ok &= ~_em_transitorio(P, seg["trans"])
    # Referencia: a peca fundida pelo proprio Revo Scan (fuse.ply). Tudo que esta perto dela e peca,
    # inclusive a face de baixo e detalhes que as regras de altura/componente cortariam.
    if seg.get("ref_arvore") is not None:
        ok |= m.isfinite(_consultar(seg["ref_arvore"], P, seg["ref_margem"]))
    # A periferia da mesa vale por ultimo: o fuse.ply do Revo inclui a mesa e nao pode ressuscita-la.
    per = seg.get("periferia")
    if per is not None:
        hr = P @ _g(per["n"], np.float64) + float(per["d"])
        faixa = m.abs(hr) <= per["banda"]
        if bool(faixa.any()):
            dist = _seg_g(seg, "per_dist", fonte=per["dist"])
            ijp = m.floor(m.stack([P @ _g(seg["u"]), P @ _g(seg["v"])], axis=1) - _g(per["uv_lo"])).astype(m.int64)
            formap = _g(dist.shape)
            dentro_p = ((ijp >= 0) & (ijp < formap)).all(axis=1)
            icp = m.clip(ijp, 0, formap - 1)
            dd = m.where(dentro_p, dist[icp[:, 0], icp[:, 1]], m.inf)
            ok &= ~(faixa & (dd > per["mm"]))
    # Exclusoes identificadas (suportes, gabaritos, fixadores) valem POR ULTIMO: prevalecem sobre a recuperacao
    # pelo fuse.ply, a pegada/coluna e, no modo precisao, sobre a evidencia temporal (final = base & ~rejeitar).
    if seg.get("exclusoes"):
        ok &= ~dentro_exclusoes(P, seg["exclusoes"])
    return _saida(ok, na_gpu)


def carregar_exclusoes(caminho: str | os.PathLike[str], scan: str) -> list[dict[str, Any]]:
    """Le exclusoes de um JSON {guid|"*": [regioes]} no referencial do mundo do scan (o do raw_preview).

    Regioes: {"tipo": "caixa", "min": [x,y,z], "max": [x,y,z]} | {"tipo": "esfera", "centro": [..], "raio": r}
    | {"tipo": "semiespaco", "normal": [..], "d": d}  (remove n.p + d > 0)
    | {"tipo": "pontos", "arquivo": "suporte.ply|.npy", "raio": r}  (remove a <= r de qualquer ponto do arquivo).
    """
    with open(caminho, encoding="utf-8") as f:
        dados = json.load(f)
    regioes = list(dados.get("*", [])) + list(dados.get(scan, []))
    base = os.path.dirname(os.path.abspath(caminho))
    out = []
    for r in regioes:
        r = dict(r)
        t = r.get("tipo")
        if t == "pontos":
            arq = r["arquivo"] if os.path.isabs(r["arquivo"]) else os.path.join(base, r["arquivo"])
            X = np.load(arq)[:, :3] if arq.endswith(".npy") else ler_ply(arq)[1][:, :3]
            r["arvore"] = ArvoreGPU(np.asarray(X, float))
        elif t not in ("caixa", "esfera", "semiespaco"):
            raise ValueError(f"exclusao com tipo desconhecido: {t!r}")
        # campos obrigatorios validados aqui (antes de gravar), nao no meio da limpeza de cada scan
        campos = {
            "caixa": (("min", 3), ("max", 3)),
            "esfera": (("centro", 3), ("raio", 1)),
            "semiespaco": (("normal", 3), ("d", 1)),
            "pontos": (("raio", 1),),
        }[t]
        for nome, n in campos:
            v = np.atleast_1d(np.asarray(r[nome], float))
            if v.shape != (n,) or not np.all(np.isfinite(v)):
                raise ValueError(f"exclusao {t}: campo {nome!r} deve ter {n} numero(s) finito(s)")
        if t == "caixa" and np.any(np.asarray(r["min"], float) > np.asarray(r["max"], float)):
            raise ValueError("exclusao caixa: min > max")
        if t in ("esfera", "pontos") and float(r["raio"]) <= 0:
            raise ValueError(f"exclusao {t}: raio deve ser > 0")
        if t == "semiespaco" and np.linalg.norm(np.asarray(r["normal"], float)) < 1e-9:
            raise ValueError("exclusao semiespaco: normal nula")
        out.append(r)
    return out


def dentro_exclusoes(P: Any, regioes: Sequence[dict[str, Any]]) -> Any:
    """True para os pontos dentro de qualquer regiao de exclusao (GPU; P numpy ou cupy, saida do mesmo tipo)."""
    m = gpu.xp()
    na_gpu = _eh_cupy(P)
    P = _g(P, np.float64)
    fora = m.zeros(len(P), bool)
    for r in regioes:
        t = r["tipo"]
        if t == "caixa":
            fora |= ((P >= _g(r["min"], np.float64)) & (P <= _g(r["max"], np.float64))).all(axis=1)
        elif t == "esfera":
            fora |= m.linalg.norm(P - _g(r["centro"], np.float64), axis=1) <= float(r["raio"])
        elif t == "semiespaco":
            fora |= P @ _g(np.asarray(r["normal"], float)) + float(r["d"]) > 0
        elif t == "pontos":
            fora |= m.isfinite(_consultar(r["arvore"], P, float(r["raio"])))
    return _saida(fora, na_gpu)


def _em_transitorio(P, trans):
    """True para os pontos que caem num voxel marcado como transitorio (GPU)."""
    m = gpu.xp()
    na_gpu = _eh_cupy(P)
    P = _g(P, np.float64)
    idx = m.floor((P - _g(trans["lo"], np.float64)) / trans["voxel"]).astype(m.int64)
    uniq = trans["uniq"]
    dims = [int(x) for x in (uniq.max(0) + 1)]
    if "_ks" not in trans:  # chaves dos voxels marcados (uma vez)
        sel = uniq[trans["flag"]].astype(np.int64)
        trans["_ks"] = _g((sel[:, 0] * dims[1] + sel[:, 1]) * dims[2] + sel[:, 2])
    dentro = ((idx >= 0) & (idx < _g(dims))).all(axis=1)
    chave = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    res = dentro & m.isin(chave, trans["_ks"])
    return _saida(res, na_gpu)


def _amostra_mantida(scan, seg):
    """Pontos (cupy) de uma amostra dos quadros que a mascara da peca mantem: o alvo do registro do fuse.ply."""
    m = gpu.xp()
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]][::6]
    amostra = [p for p in (scan.pontos(k, passo=3, na_gpu=True)[0] for k in ks) if p is not None]
    W = m.vstack(amostra)
    return W[mascara_manter(W, seg)]


def _nuvem_reg(X, v):
    """Nuvem de registro na GPU: centroides por voxel de `v` mm (grade como a do Open3D, origem = minimo - v/2) e
    normais PCA (vizinhos a ate 3v) como no `estimate_normals` hibrido. Devolve (pontos, normais) cupy."""
    m = gpu.xp()
    X = _g(X, np.float64)
    C, _ = gpu.reduzir_voxel(X, v, origem=X.min(0) - v / 2)
    N = gpu.normais_pca(C, raio=3 * v)
    bom = m.isfinite(N).all(axis=1)
    return C[bom], N[bom]


def _icp(src, dst, ndst, T0, dist):
    """ICP ponto-plano na GPU (30 iteracoes, minimos quadrados puros como o Open3D). Devolve T 4x4."""
    return gpu.icp_ponto_plano(src, dst, ndst, T0=T0, dist_max=dist, iteracoes=30, perda="l2")[0]


def _qualidade_registro(src, dst, T, dist):
    """(fitness, rmse) do alinhamento T como o Open3D: fracao de `src` com par a <= dist e RMSE das distancias."""
    m = gpu.xp()
    p = src @ _g(T[:3, :3].T) + _g(T[:3, 3])
    d, j = gpu.Indice(dst, dist).mais_proximo(p)
    ok = j >= 0
    n = int(ok.sum())
    return n / max(len(src), 1), (float(m.sqrt((d[ok] ** 2).mean())) if n else float("inf"))


def _ransac_features(src, nsrc, dst, ndst, iteracoes, dist=3.0):
    """RANSAC por casamento de FPFH (Open3D, CPU: nao ha equivalente na GPU). Devolve a transformacao 4x4."""
    import open3d as o3d

    reg = o3d.pipelines.registration

    def nuvem(X, N):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(X)))
        p.normals = o3d.utility.Vector3dVector(gpu.para_cpu(N))
        return p

    s, d = nuvem(src, nsrc), nuvem(dst, ndst)
    feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=100))
    r = reg.registration_ransac_based_on_feature_matching(
        s,
        d,
        feat(s),
        feat(d),
        True,
        dist,
        reg.TransformationEstimationPointToPoint(False),
        3,
        [reg.CorrespondenceCheckerBasedOnDistance(dist)],
        reg.RANSACConvergenceCriteria(iteracoes, 0.999),
    )
    return np.array(r.transformation)


def _ler_fuse(caminho):
    """Pontos do fuse.ply (leitura de arquivo com Open3D: aceita ascii/binario)."""
    import open3d as o3d

    return np.asarray(o3d.io.read_point_cloud(caminho).points, float)


def registrar_fuse(
    scan: RevoScan | None,
    seg: dict[str, Any],
    caminho: str | os.PathLike[str],
    log: Callable[..., None] = print,
    iteracoes: int = 200000,
) -> tuple[np.ndarray, float, float] | None:
    """Registra um fuse.ply no referencial dos quadros. Retorna (T, fitness, rmse) ou None.

    O alvo sao os pontos dos quadros que a mascara mantem (a peca e o que a rodeia de perto).
    """
    F = _ler_fuse(caminho)
    return _registrar_nuvens(F, _amostra_mantida(scan, seg), log, iteracoes)


def adicionar_referencia_fuse(
    scan: RevoScan | None, seg: dict[str, Any], margem_mm: float = 2.5, log: Callable[..., None] = print
) -> dict[str, Any]:
    """Registra o fuse.ply do scan no referencial dos quadros e o guarda em `seg` como referencia.

    Sem fuse.ply ou com registro ruim, nao altera `seg` (a limpeza segue so com as regras geometricas).
    ICP na GPU; o RANSAC por features (so se o ICP direto nao servir) fica no Open3D.
    """
    caminho = os.path.join(scan.pasta, "fuse.ply")
    if not os.path.exists(caminho):
        log("   referencia fuse.ply ausente: seguindo so com as regras geometricas")
        return seg
    F = _g(_ler_fuse(caminho), np.float64)
    W = _amostra_mantida(scan, seg)
    src, nsrc = _nuvem_reg(F, 0.8)
    dst, ndst = _nuvem_reg(W, 0.8)
    # o fuse.ply costuma ja estar no referencial dos quadros: ICP direto primeiro, RANSAC so se nao servir
    T = _icp(src, dst, ndst, np.eye(4), 2.0)
    T = _icp(src, dst, ndst, T, 0.8)
    fit, rmse = _qualidade_registro(src, dst, T, 0.8)
    if fit < 0.85 or rmse > 0.6:
        T = _ransac_features(src, nsrc, dst, ndst, 200000)
        T = _icp(src, dst, ndst, T, 0.8)
        fit, rmse = _qualidade_registro(src, dst, T, 0.8)
    log(f"   registro do fuse.ply: fitness {fit:.3f} | rmse {rmse:.3f} mm")
    if fit < 0.85 or rmse > 0.6:
        log("   registro ruim: referencia fuse.ply ignorada")
        return seg
    Fw = F @ _g(T[:3, :3].T) + _g(T[:3, 3])
    C, _ = gpu.reduzir_voxel(Fw, 0.3, origem=Fw.min(0) - 0.15)
    seg["ref_arvore"] = ArvoreGPU(C)
    seg["ref_margem"] = margem_mm
    return seg


def _registrar_nuvens(F, W, log=print, iteracoes=200000):
    """Registra F em W (FPFH + RANSAC no Open3D, depois ICP ponto-plano na GPU). Retorna (T, fitness, rmse)."""
    src, nsrc = _nuvem_reg(F, 0.8)
    dst, ndst = _nuvem_reg(W, 0.8)
    T = _ransac_features(src, nsrc, dst, ndst, iteracoes)
    T = _icp(src, dst, ndst, T, 0.8)
    fit, rmse = _qualidade_registro(src, dst, T, 0.8)
    log(f"   registro do fuse.ply: fitness {fit:.3f} | rmse {rmse:.3f} mm")
    return T, float(fit), float(rmse)


def limpar_fuse(
    scan: RevoScan | None,
    seg: dict[str, Any],
    fuse_entrada: str | os.PathLike[str],
    fuse_saida: str | os.PathLike[str],
    log: Callable[..., None] = print,
) -> dict[str, Any] | None:
    """Limpa a nuvem FUNDIDA pelo Revo (fuse.ply) com a mesma mascara de selecao da peca.

    Em scans sem marcadores o Revo se orienta pela geometria dos objetos de rastreamento em volta
    da peca: apagar esses objetos dos quadros antes da fusao faz a fusao perder regioes da peca.
    Por isso a limpeza e feita DEPOIS da fusao, na nuvem fundida, que fica no referencial do scan.
    Retorna dict com os numeros, ou None se o registro for ruim (nada e gravado).
    """
    hdr, arr = ler_ply(fuse_entrada)
    F = _g(arr[:, :3], np.float64)
    r = registrar_fuse(scan, seg, fuse_entrada, log)
    T, fit, rmse = r
    if fit < 0.85 or rmse > 0.6:
        log("   registro ruim: fuse.ply NAO foi limpo")
        return None
    Fw = F @ _g(T[:3, :3].T) + _g(T[:3, 3])
    ok = gpu.para_cpu(mascara_manter(Fw, seg))
    gravar_ply(fuse_saida, hdr, arr[ok])
    log(f"   fuse.ply: {len(arr):,} -> {ok.sum():,} pontos ({ok.mean() * 100:.1f}% mantidos)")
    return dict(antes=int(len(arr)), depois=int(ok.sum()), fitness=fit, rmse=rmse)


# ---------------------------------------------------------------- gravacao
def _residuos_pixels(D):
    """(res, pares) de um mapa D (cupy, NaN onde invalido): menor desvio de cada pixel para a media de um par de
    vizinhos opostos (4 pares) e quantos pares tinham os dois vizinhos validos."""
    m = gpu.xp()
    h, w = D.shape
    pad = m.pad(D, 1, constant_values=m.nan)
    sh = lambda dy, dx: pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    res = m.full(D.shape, m.inf, m.float32)
    pares = m.zeros(D.shape, m.uint8)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r = m.abs(D - (sh(dy, dx) + sh(-dy, -dx)) / 2)
        pares += m.isfinite(r)
        res = m.minimum(res, m.where(m.isnan(r), m.inf, r))
    return res, pares


def preparar_pixels(D: Any) -> tuple[Any, Any, Any]:
    """Pre-calculo de `filtro_pixels` (mapa limpo, residuos, pares) em cupy; reaproveitavel por varias chamadas de
    `filtrar_pixels` com tol/modos diferentes sobre o MESMO mapa."""
    m = gpu.xp()
    D = _g(D)
    if D.dtype.kind != "f":
        D = D.astype(m.float64)
    D = m.where(m.isfinite(D) & (D > 0), D, m.nan)
    res, pares = _residuos_pixels(D)
    return D, res, pares


def filtrar_pixels(prep: Any, tol: float = 1.0, comp_min: int = 30, adaptativo: bool = False) -> Any:
    """Decisao do `filtro_pixels` sobre o pre-calculo de `preparar_pixels`; devolve cupy bool (h, w)."""
    if not np.isfinite(tol) or tol <= 0 or comp_min < 1:
        raise ValueError("tol deve ser positivo e comp_min >= 1")
    m = gpu.xp()
    nd = _nd()
    D, res, pares = prep
    # sem nenhum par de vizinhos validos nao ha como julgar: nao e voador (so a regra das manchas atua)
    limite = tol
    if adaptativo:
        # Estima dispersao das segundas diferencas localmente. Mediana/MAD
        # nao deixam um pico isolado aumentar seu proprio limite de rejeicao.
        validos = m.isfinite(res)
        preenchido = m.where(validos, res, res.dtype.type(0))
        med = nd.median_filter(preenchido, size=5, mode="nearest")
        mad = nd.median_filter(m.abs(preenchido - med), size=5, mode="nearest")
        # contagem exata de pixels validos na janela 5x5 (inteiros): apoio >= 0,8 equivale a >= 20 de 25
        # (o uniform_filter em ponto flutuante cai ora de um lado ora do outro exatamente nesse empate)
        apoio = nd.convolve(validos.astype(m.int32), m.ones((5, 5), m.int32), mode="nearest")
        limite = m.maximum(res.dtype.type(tol), med + res.dtype.type(4.5 * 1.4826) * mad)
        # Pouco apoio nao permite estimativa robusta: conservar o pixel.
        limite = m.where((apoio >= 20) & (pares >= 3), limite, m.inf)
    val = ~m.isnan(D) & ~(m.isfinite(res) & (res > limite))
    lab, n = nd.label(val, structure=_uns(3, 3))
    tam = m.bincount(lab.ravel(), minlength=int(n) + 1)
    tam[0] = 0
    return val & (tam[lab] >= comp_min)


def filtro_pixels(D: Any, tol: float = 1.0, comp_min: int = 30, adaptativo: bool = False) -> Any:
    """Mascara dos pixels a manter num mapa de profundidade D (h x w, NaN onde invalido). GPU; D numpy ou cupy
    (saida do mesmo tipo).

    Remove (1) pixels voadores: o pixel desvia mais de `tol` mm da media dos dois vizinhos em
    TODAS as direcoes (4 pares opostos). Superficie lisa, inclinada ou curva fica ~0 em alguma
    direcao e borda real fica ~0 ao longo da borda; so ruido entre a peca e o fundo desvia nas
    quatro. (2) manchas pequenas de pixels desconectados do resto (< `comp_min` px).
    """
    if not np.isfinite(tol) or tol <= 0 or comp_min < 1:
        raise ValueError("tol deve ser positivo e comp_min >= 1")
    na_gpu = _eh_cupy(D)
    return _saida(filtrar_pixels(preparar_pixels(D), tol, comp_min, adaptativo), na_gpu)


def ilhas_3d(A: Any, voxel: float = 1.0, gap_vox: int = 2, frac_min: float = 0.02) -> dict[str, Any] | None:
    """Grade booleana dos voxels que pertencem a peca principal ou a pedacos grandes dela (GPU).

    A: amostra de pontos mantidos (N x 3). Voxels ocupados sao dilatados `gap_vox` voxels para
    que fendas finas nao separem a peca; componente com menos de `frac_min` do maior e ilha solta.
    """
    m = gpu.xp()
    nd = _nd()
    A = _g(A, np.float64)
    if np.prod(gpu.para_cpu(A.max(0) - A.min(0)) / voxel + 8) > 4e8:  # cena muito grande: grade mais grossa
        voxel *= 2
    lo = gpu.para_cpu(A.min(0)) - (gap_vox + 2) * voxel
    idx = m.floor((A - _g(lo)) / voxel).astype(m.int64)
    ii = (idx[:, 0], idx[:, 1], idx[:, 2])
    g = m.zeros(tuple(int(x) for x in gpu.para_cpu(idx.max(0)) + gap_vox + 3), bool)
    g[ii] = True
    gd = _morf("binary_dilation", g, structure=_uns(3, 3, 3), iterations=gap_vox)
    del g
    lab, n = nd.label(gd, structure=_uns(3, 3, 3))
    del gd
    tam = m.bincount(lab[ii], minlength=int(n) + 1)
    tam[0] = 0
    bom = tam >= frac_min * tam.max()
    bom[0] = False
    grade = bom[lab]
    ilh = dict(
        lo=lo,
        voxel=voxel,
        grade=gpu.para_cpu(grade),
        n_comp=int((tam > 0).sum()),
        n_removidas=int(((tam > 0) & ~bom).sum()),
        tam_max=int(tam.max()),
    )
    ilh["_g"] = (ilh["grade"], grade)
    return ilh


def mascara_ilhas(P: Any, ilh: dict[str, Any]) -> Any:
    """True para os pontos dentro das ilhas boas (GPU; P numpy ou cupy, saida do mesmo tipo)."""
    m = gpu.xp()
    na_gpu = _eh_cupy(P)
    P = _g(P, np.float64)
    ref = ilh.get("_g")
    if ref is None or ref[0] is not ilh["grade"]:
        ref = ilh["_g"] = (ilh["grade"], _g(ilh["grade"]))
    grade = ref[1]
    idx = m.floor((P - _g(ilh["lo"], np.float64)) / ilh["voxel"]).astype(m.int64)
    forma = _g(grade.shape)
    dentro = ((idx >= 0) & (idx < forma)).all(axis=1)
    ic = m.clip(idx, 0, forma - 1)
    return _saida(grade[ic[:, 0], ic[:, 1], ic[:, 2]] & dentro, na_gpu)


def _n_celulas(P, celula):
    """Numero de celulas cubicas de `celula` mm (grade absoluta, origem 0) com algum ponto de P (cupy)."""
    if len(P) == 0:
        return 0
    m = gpu.xp()
    origem = m.floor(P.min(0) / celula) * celula
    return int(len(gpu.agrupar(gpu.chaves_voxel(P, celula, origem))[0]))


PLY_ANTIGOS = ("fuse.ply", "fuse_mesh.ply", "mark.ply")


def limpar_scan(
    scan: RevoScan | None,
    seg: dict[str, Any],
    destino_scan: str | os.PathLike[str],
    log: Callable[..., None] = print,
    pixels: bool = True,
    ilhas: bool = True,
    min_mantido: float = 0.0,
    sem_ply_antigos: bool = False,
    zerar_vazios: bool = False,
    min_pix_quadro: int = 1,
    pixel_tol: float = 1.0,
    pixel_comp_min: int = 30,
    pixel_adaptativo: bool = False,
    precisao: bool = False,
    temporal_passo: int = 8,
    temporal_min: int = 2,
) -> dict[str, Any]:
    """Grava a copia do scan com os pixels fora da peca apagados em cada quadro.

    Alem da mascara da peca (`mascara_manter`): `pixels` tira pixels voadores e manchas 2D
    minusculas de cada mapa de profundidade; `ilhas` tira pedacos 3D soltos da peca principal.

    Todo o calculo por quadro roda na GPU (pontos, mascaras, filtro de pixels, evidencia temporal) e os resultados
    ficam em cupy ate a gravacao; na CPU so o SQLite, o empacotamento dos bits e os PLY. `mascara_manter` (e quem a
    substitui nos scripts de placa/merge/componentes) recebe os pontos em cupy e devolve bool em cupy.
    """
    m = gpu.xp()
    os.makedirs(destino_scan, exist_ok=True)
    for item in os.listdir(scan.pasta):
        src = os.path.join(scan.pasta, item)
        if item == "cache":
            continue
        if sem_ply_antigos and item in PLY_ANTIGOS:  # nuvem/malha/marcadores ja fundidos, com a mesa
            continue
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(destino_scan, item), dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(destino_scan, item))
    db = scan.db + ".limpo"
    scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
    shutil.copy(scan.db, db)
    cn = sqlite3.connect(db)
    from precisao import ValidadorTemporal, render_cortes

    temporal = ValidadorTemporal(scan, pixel_tol, temporal_passo, temporal_min) if precisao else None
    vazios, oks, amostra, tirados = [], {}, [], []
    bases, fixos, detalhes = {}, {}, {}
    n_total = n_mask = n_pix = 0
    _nd()
    # passada 1: mascara da peca + filtro de pixels por quadro
    for k, i in enumerate(scan.ids):
        if not scan.tem_prof[k]:
            continue
        dados = scan.profundidade(i)
        w, h, mask, z, quatro = dados
        P, pix = scan.pontos(k, na_gpu=True, dados=dados)
        pix_g = _g(pix)
        ok = _g(mascara_manter(P, seg), bool)
        n_total += len(pix)
        n_mask += int(ok.sum())
        base = ok
        bases[k] = base.copy()
        if pixels and bool(ok.any()):
            D = m.full(w * h, m.nan, m.float32)
            D[_g(mask)] = _g(z, np.float32)
            mapa = D.reshape(h, w)
            prep = preparar_pixels(mapa)  # residuos dos 4 pares: uma vez para os filtros abaixo
            ok = ok & filtrar_pixels(prep, pixel_tol, pixel_comp_min, pixel_adaptativo).ravel()[pix_g]
            if precisao:
                fixos[k] = base & filtrar_pixels(prep, pixel_tol, pixel_comp_min, False).ravel()[pix_g]
                # Contorno de cobertura e saltos de profundidade: apenas mede;
                # a decisao de apagar exige evidencia em outros quadros.
                valido = m.isfinite(mapa)
                borda = valido & ~_morf("binary_erosion", valido)
                for eixo in (0, 1):
                    salto = m.abs(m.diff(mapa, axis=eixo)) > pixel_tol
                    a = [slice(None), slice(None)]
                    b = a.copy()
                    a[eixo] = slice(None, -1)
                    b[eixo] = slice(1, None)
                    borda[tuple(a)] |= salto
                    borda[tuple(b)] |= salto
                detalhes[k] = borda.ravel()[pix_g] & base
        n_pix += int(ok.sum())
        oks[k] = ok
        par = m.arange(len(P)) % 2 == 0  # mesma amostra (pixels pares) nos dois lados
        amostra.append(P[ok & par].astype(m.float32))
        tirados.append(P[base & ~ok & par].astype(m.float32))
    if (n_mask if precisao else n_pix) < min_mantido * max(1, n_total):
        log(
            f"   REVISAR: a limpeza manteria so {n_pix / max(1, n_total) * 100:.0f}% dos pontos (minimo {min_mantido * 100:.0f}%); "
            f"escolha da peca suspeita, scan copiado sem alteracao"
        )
        cn.close()
        os.remove(db)
        shutil.copytree(scan.pasta, destino_scan, dirs_exist_ok=True)
        return dict(revisar=True, antes=n_total, depois=n_pix)
    A = m.vstack(amostra) if amostra else m.empty((0, 3))
    T0 = m.vstack(tirados) if tirados else m.empty((0, 3))
    del amostra, tirados
    # ilhas 3D soltas
    ilh = ilhas_3d(A) if ilhas and len(A) else None
    seg["ilhas"] = ilh  # render_comparacao mostra o resultado final
    if ilh is not None:
        log(f"   ilhas 3D: {ilh['n_comp']} componentes, {ilh['n_removidas']} removidas (menores que 2% da peca)")

    # auditoria: dos pontos que os filtros novos tiraram, quantos estavam colados na superficie do fuse.ply
    # (indica risco de perder superficie real; o restante e ruido fora da superficie)
    def auditar(R, rotulo):
        if seg.get("ref_arvore") is not None and len(R):
            perto = m.isfinite(_consultar(seg["ref_arvore"], R, 0.4))
            log(
                f"   auditoria {rotulo}: {len(R):,} pontos (amostra) | {int(perto.sum()):,} "
                f"({float(perto.mean()) * 100:.1f}%) a menos de 0,4 mm do fuse.ply"
            )

    auditar(T0, "pixels voadores")
    if ilh is not None:
        auditar(A[~_g(mascara_ilhas(A, ilh), bool)], "ilhas")
    del A, T0
    # passada 2: grava
    tot_antes = tot_depois = 0
    qualidade = dict(
        candidatos=0,
        confirmados=0,
        com_apoio=0,
        inconclusivos=0,
        borda_antes=0,
        borda_depois=0,
        fixo_depois=0,
        adaptativo_depois=0,
        cobertura_amostrada=None,
    )
    cob_base, cob_local, cob_final, removidos_precisao = [], [], [], []
    por_quadro = []
    revisao_pixels = {}
    for k, i in enumerate(scan.ids):
        if not scan.tem_prof[k]:
            continue
        dados = scan.profundidade(i)
        w, h, mask, z, quatro = dados
        P, pix = scan.pontos(k, na_gpu=True, dados=dados)
        ok = oks[k]
        if ilh is not None:
            ok = ok & _g(mascara_ilhas(P, ilh), bool)
        if precisao:
            base = bases[k]
            proposta_local = ok.copy()
            rejeitar, evidencia = temporal.validar(k, P, base & ~ok, referencia=base)
            if evidencia["candidatos"]:
                revisao_pixels[f"q{i}_dimensoes"] = np.array([h, w], np.int32)
                for nome_evidencia, mascara in temporal.ultima_evidencia.items():
                    revisao_pixels[f"q{i}_{nome_evidencia}"] = pix[gpu.para_cpu(mascara)].astype(np.int32)
            for chave, valor in evidencia.items():
                qualidade[chave] += valor
            qualidade["fixo_depois"] += int(fixos.get(k, base).sum())
            qualidade["adaptativo_depois"] += int(oks[k].sum())
            borda = detalhes.get(k, m.zeros(len(P), bool))
            ok = base & ~rejeitar
            qualidade["borda_antes"] += int(borda.sum())
            qualidade["borda_depois"] += int((borda & ok).sum())
            # Mesmos pontos amostrados nos dois lados; nao mede exatidao absoluta.
            am = m.arange(len(P)) % 8 == 0
            cob_base.append(P[base & am])
            cob_local.append(P[proposta_local & am])
            cob_final.append(P[ok & am])
            removidos_precisao.append(P[rejeitar & am])
            por_quadro.append(
                dict(quadro=int(i), antes=len(P), selecionados=int(base.sum()), depois=int(ok.sum()), **evidencia)
            )
        oks[k] = ok
        tot_antes += len(pix)
        n_ok = int(ok.sum())
        if n_ok < min_pix_quadro:
            vazios.append(k)  # quadro so com prato/apoio
            if zerar_vazios:  # remove a profundidade do quadro (estado que o Revo ja usa em quadros sem profundidade)
                cn.execute("delete from FrameRGBD where id=?", (i,))
            else:  # padrao: mantem o original
                tot_depois += len(pix)
            continue
        okc = gpu.para_cpu(ok)
        nova = np.zeros(w * h, bool)
        nova[pix[okc]] = True
        dados_novos = (
            struct.pack("<III", w, h, n_ok)
            + np.packbits(nova, bitorder="little").tobytes()
            + quatro
            + z[okc].astype("<f4").tobytes()
        )
        cn.execute("update FrameRGBD set data=? where id=?", (dados_novos, i))
        tot_depois += n_ok
    cn.commit()
    integ = cn.execute("pragma integrity_check").fetchone()[0]
    # consolida o journal (o Revo Scan usa modo WAL) dentro do arquivo principal
    cn.execute("pragma wal_checkpoint(TRUNCATE)")
    cn.close()
    if integ != "ok":
        raise RuntimeError("integridade SQLite: " + integ)
    for ext in ("-wal", "-shm", "-journal"):
        if os.path.exists(db + ext):
            raise RuntimeError("sobrou arquivo de journal " + ext)
    with open(db, "r+b") as f:
        f.write(scan.sig)
    os.makedirs(os.path.join(destino_scan, "cache"), exist_ok=True)
    # Criar o arquivo no destino faz herdar as permissoes da entrega. Mover do
    # TemporaryDirectory preserva a ACL privada do temporario no Windows e
    # impede o Revo Scan (processo do usuario) de ler o banco.
    shutil.copyfile(db, os.path.join(destino_scan, "cache", "frames.dataset"))

    # raw_preview filtrado pela mesma regra (esta no referencial do mundo)
    hdr, prev = ler_ply(os.path.join(scan.pasta, "raw_preview.ply"))
    prev_g = _g(prev[:, :3], np.float64)
    okp = _g(mascara_manter(prev_g, seg), bool)
    # No modo precisao as ilhas sao so CANDIDATAS (a evidencia temporal decide nos quadros); tirar do preview
    # mostraria ausencia onde os quadros ainda tem o dado. Classico: preview segue a mesma regra dos quadros.
    if ilh is not None and not precisao:
        okp &= _g(mascara_ilhas(prev_g, ilh), bool)
    okp = gpu.para_cpu(okp)
    gravar_ply(os.path.join(destino_scan, "raw_preview.ply"), hdr, prev[okp])
    seg["_mascaras_finais"] = {k: gpu.para_cpu(v) for k, v in oks.items()}
    if precisao:
        cat = lambda partes: m.vstack(partes) if partes else m.empty((0, 3))
        cb, cf = cat(cob_base), cat(cob_final)
        render_cortes(
            gpu.para_cpu(cb),
            gpu.para_cpu(cat(cob_local)),
            gpu.para_cpu(cf),
            os.path.join(destino_scan, "cortes_qualidade.png"),
        )
        n_cb, n_cf = _n_celulas(cb, 0.5), _n_celulas(cf, 0.5)
        qualidade["cobertura_amostrada"] = dict(
            celula_mm=0.5, passo_pixels=8, celulas_base=n_cb, celulas_mantidas=n_cf, fracao=n_cf / max(1, n_cb)
        )
        qualidade["pares_suspeitos"] = sum(
            p["suspeita_desalinhamento"] for q in temporal.diagnosticos for p in q["pares"]
        )
        qualidade["diagnostico"] = (
            "vies de profundidade por reprojecao; nao e erro absoluto ou alinhamento ponto-a-plano"
        )
        qualidade["por_quadro"] = por_quadro
        qualidade["reprojecoes"] = temporal.diagnosticos
        ref = seg.get("ref_arvore")
        rr = cat(removidos_precisao)
        if ref is not None and len(rr):
            perto = m.isfinite(_consultar(ref, rr, 0.4))  # (dist < 0,4) == tem vizinho a menos de 0,4 mm
            qualidade["removidos_proximos_fuse_amostra"] = dict(n=len(rr), abaixo_04mm=int(perto.sum()))
        with open(os.path.join(destino_scan, "qualidade_precisao.json"), "w", encoding="utf-8") as f:
            json.dump(qualidade, f, indent=2)
        np.savez_compressed(os.path.join(destino_scan, "revisao_pixels.npz"), **revisao_pixels)
        log(
            f"   precisao: {qualidade['candidatos']:,} candidatos; {qualidade['confirmados']:,} removidos por espaco livre em outros quadros; "
            f"{qualidade['com_apoio']:,} com apoio; {qualidade['inconclusivos']:,} preservados por incerteza"
        )
    log(
        f"   pontos dos quadros: {tot_antes:,} -> {tot_depois:,} ({tot_depois / max(1, tot_antes) * 100:.0f}% mantidos; "
        f"peca {n_mask:,} -> filtro local propoe {n_mask - n_pix:,} descartes) | preview {len(prev):,} -> {okp.sum():,} | integridade: {integ}"
    )
    if vazios:
        log(
            f"   AVISO: {len(vazios)} quadros sem ponto da peca "
            + ("tiveram a profundidade removida" if zerar_vazios else "foram mantidos como estavam")
        )
    del bases, fixos, detalhes, oks
    gpu.liberar_memoria()
    return dict(
        antes=tot_antes,
        depois=tot_depois,
        vazios=len(vazios),
        integridade=integ,
        voadores=qualidade["confirmados"] if precisao else n_mask - n_pix,
        ilhas_removidas=ilh["n_removidas"] if ilh and not precisao else 0,
        precisao={k: v for k, v in qualidade.items() if k not in ("por_quadro", "reprojecoes")} if precisao else None,
    )


def gravar_revo(
    origem_revo: str | os.PathLike[str], destino_proj: str | os.PathLike[str], nome: str, nos: Sequence[str]
) -> dict[str, Any]:
    r = json.load(open(origem_revo, encoding="utf-8"))
    r["guid"] = str(uuid.uuid4())
    r["name"] = nome
    r["edit_time"] = str(int(time.time()))
    r["nodes"] = [n for n in r["nodes"] if n["guid"] in nos]
    # igual ao Revo Scan: soma so os arquivos da pasta data
    tam = sum(
        os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(destino_proj, "data")) for f in fs
    )
    r["size"] = str(tam)
    with open(os.path.join(destino_proj, nome + ".revo"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=4)
    return r


def render_comparacao(
    scan: RevoScan | None, seg: dict[str, Any], path: str | os.PathLike[str], nquadros: int = 60
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    ks = ks[:: max(1, len(ks) // nquadros)]
    partes, manter = [], []
    for k in ks:
        p = scan.pontos(k, passo=6)[0]
        if p is not None:
            partes.append(p)
            if "_mascaras_finais" in seg:
                manter.append(seg["_mascaras_finais"][k][::6])
            else:
                m = mascara_manter(p, seg)
                if seg.get("ilhas") is not None:
                    m &= mascara_ilhas(p, seg["ilhas"])
                manter.append(m)
    P, ok = _g(np.vstack(partes)), np.concatenate(manter)
    n = seg["n"]
    u = np.cross(n, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    L = gpu.para_cpu(
        gpu.xp().stack([P @ _g(u), P @ _g(v), P @ _g(n) + seg["d"]], axis=1)
    )  # projecao na GPU; so o grafico e CPU
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), dpi=90)
    for col, (titulo, m) in enumerate((("ANTES (todos os pontos)", np.ones(len(L), bool)), ("DEPOIS (so a peca)", ok))):
        ax[0, col].scatter(L[m, 0], L[m, 1], s=0.2, c=L[m, 2], cmap="viridis", linewidths=0)
        ax[0, col].set_aspect("equal")
        ax[0, col].set_title(titulo + ": vista de cima")
        ax[1, col].scatter(L[m, 0], L[m, 2], s=0.2, c="#333", linewidths=0)
        ax[1, col].set_aspect("equal")
        ax[1, col].set_title(titulo + ": vista lateral")
    for a in ax[:, 1]:
        a.set_xlim(ax[0, 0].get_xlim())
    ax[0, 1].set_ylim(ax[0, 0].get_ylim())
    ax[1, 1].set_ylim(ax[1, 0].get_ylim())
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
