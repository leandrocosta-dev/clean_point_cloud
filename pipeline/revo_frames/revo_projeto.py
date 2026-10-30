"""Leitura e limpeza quadro a quadro de projetos do Revo Scan 6 (formato 1.3).

Regras de seguranca:
- O projeto original nunca e alterado. Tudo e gravado numa copia.
- As poses NAO sao alteradas. A limpeza so apaga pixels do mapa de profundidade
  de cada quadro (mascara de bits + valores), mantendo o resto do registro igual.
- Nao depende do Open3D (so numpy, scipy e matplotlib).

Formato (confirmado em 5 projetos):
- cache/frames.dataset: SQLite com os 16 primeiros bytes trocados por b'RPAlgo_file_sys\\0'.
- FrameInfo.bin: cabecalho 16 bytes + pose 4x4 float64 (camera para mundo) + cauda.
  Modo do quadro = (hdr[1] >> 16) & 0xff (indice da pasta de calibracao param, param1...).
- FrameRGBD.data: largura, altura, n (uint32) + mascara de bits w*h (ordem little)
  + 4 bytes + n float32 (profundidade em mm, na ordem linha a linha dos pixels validos).
- Ordem temporal: (id >> 32, id & 0xffffffff). Numeracao do Revo Scan = indice entre
  os quadros que tem profundidade, nessa ordem.
"""
import json
import os
import shutil
import sqlite3
import struct
import tempfile
import uuid
import time

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

SIG_REVO = b"RPAlgo_file_sys\x00"
SIG_SQL = b"SQLite format 3\x00"
MAX_VOXELS = 3e8          # voxels da grade da peca (bool + rotulos int32 = ~5 bytes cada)
MAX_VOXELS_TRANS = 1e8    # grade dos transitorios (2 float32 + filtros)
THREADS = int(os.environ.get("REVO_THREADS", "-1"))   # threads por consulta de KD-tree (-1 = todas)
K_PADRAO = (1140.4, 1141.0, 232.8, 319.5)   # modo usado nos scans de teste


# ---------------------------------------------------------------- utilidades
def ler_ply(path):
    b = open(path, "rb").read()
    k = b.find(b"end_header\n") + 11
    h = b[:k].decode("latin1")
    n = int([l for l in h.split("\n") if l.startswith("element vertex")][0].split()[-1])
    nprop = len([l for l in h.split("\n") if l.startswith("property")])
    return h, np.frombuffer(b[k:k + n * 4 * nprop], "<f4").reshape(n, nprop).copy()


def gravar_ply(path, header, arr):
    linhas = header.split("\n")
    linhas = [f"element vertex {len(arr)}" if l.startswith("element vertex") else l for l in linhas]
    with open(path, "wb") as f:
        f.write("\n".join(linhas).encode("latin1"))
        f.write(arr.astype("<f4").tobytes())


def plano_ransac(P, thr=1.0, it=3000, seed=0):
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        k = int((np.abs(P @ n + d) < thr).sum())
        if k > best[0]:
            best = (k, (n, d))
    n, d = best[1]
    m = np.abs(P @ n + d) < thr
    c = P[m].mean(0)
    _, _, V = np.linalg.svd(P[m] - c, full_matrices=False)
    n = V[2]
    d = -n @ c
    # orienta a normal para o lado onde estao os objetos
    if np.mean(P @ n + d > 3 * thr) < np.mean(P @ n + d < -3 * thr):
        n, d = -n, -d
    return n, d, float(m.mean())


# ---------------------------------------------------------------- leitura
class RevoScan:
    def __init__(self, pasta_scan, workdir, reusar=False):
        self.pasta = pasta_scan
        self.nome = os.path.basename(os.path.normpath(pasta_scan))
        src = os.path.join(pasta_scan, "cache", "frames.dataset")
        self.sig = open(src, "rb").read(16)
        if self.sig != SIG_REVO:
            raise ValueError(f"{self.nome}: assinatura inesperada {self.sig!r}")
        self.db = os.path.join(workdir, self.nome + ".sqlite")
        if not (reusar and os.path.exists(self.db)
                and os.path.getsize(self.db) == os.path.getsize(src)):
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
            pop = int(np.unpackbits(np.frombuffer(d[12:12 + nb], np.uint8)).sum())
            if pop != n or len(d) < 12 + nb + 4 + 4 * n:
                self.tem_prof[k] = False
                self.inconsistentes.append(k)
        self.K = {}

    def profundidade(self, i):
        r = self.c.execute("select data from FrameRGBD where id=?", (i,)).fetchone()
        if r is None:
            return None
        d = r[0]
        w, h, n = struct.unpack("<III", d[:12])
        nb = w * h // 8
        mask = np.unpackbits(np.frombuffer(d[12:12 + nb], np.uint8), bitorder="little").astype(bool)
        z = np.frombuffer(d[12 + nb + 4:12 + nb + 4 + 4 * n], "<f4")
        return w, h, mask, z, d[12 + nb:12 + nb + 4]

    def pontos(self, k, K=None, passo=1, mundo=True):
        r = self.profundidade(self.ids[k])
        if r is None:
            return None, None
        w, h, mask, z, _ = r
        pix = np.flatnonzero(mask)
        if passo > 1:
            sel = np.arange(0, len(pix), passo)
            pix, z = pix[sel], z[sel]
        v, u = np.divmod(pix, w)
        fx, fy, cx, cy = K if K is not None else self.K[int(self.modo[k])]
        z = z.astype(float)
        P = np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z]
        if mundo:
            T = self.T[k]
            P = P @ T[:3, :3].T + T[:3, 3]
        return P, pix

    def foco_gravado(self, modo):
        """Foco do mapa de profundidade a partir de param*/Pl.bin (a camera estereo tem o dobro da resolucao)."""
        pasta = os.path.join(self.pasta, "param" if modo == 0 else f"param{modo}", "Pl.bin")
        try:
            f = float(np.frombuffer(open(pasta, "rb").read(), "<f4")[1]) / 2
            return f if 300 < f < 3000 else None
        except (OSError, IndexError, ValueError):
            return None

    def calibrar(self, modo, x0=None, nquadros=8):
        """Ajusta fx, fy, cx, cy contra o raw_preview.ply. Retorna (K, erro mediano em mm)."""
        _, prev = ler_ply(os.path.join(self.pasta, "raw_preview.ply"))
        arv = cKDTree(prev[:, :3].astype(float))
        ks = [k for k in range(len(self.ids)) if self.modo[k] == modo and self.tem_prof[k]]
        if not ks:
            return None, None
        ks = [ks[j] for j in np.linspace(0, len(ks) - 1, min(nquadros, len(ks))).astype(int)]
        dados = []
        for k in ks:
            w, h, mask, z, _ = self.profundidade(self.ids[k])
            v, u = np.divmod(np.flatnonzero(mask), w)
            dados.append((u.astype(float), v.astype(float), z.astype(float), self.T[k]))

        def custo(p, sub=12):
            fx, fy, cx, cy = p
            e = []
            for u, v, z, T in dados:
                u, v, z = u[::sub], v[::sub], z[::sub]
                P = np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z] @ T[:3, :3].T + T[:3, 3]
                e.append(np.minimum(arv.query(P, workers=THREADS)[0], 3.0))
            return float(np.mean(np.concatenate(e)))

        if x0 is None:
            # Varias partidas: o foco gravado no scan (ou o padrao) e centros tipicos. Foco unico (fx=fy)
            # primeiro, para nao cair em minimo local; depois ajuste livre a partir do melhor.
            f0 = self.foco_gravado(modo) or K_PADRAO[0]
            melhor = None
            for cx0, cy0 in ((232.8, 319.5), (324.0, 327.0), (400.0, 300.0)):
                r0 = minimize(lambda q: custo((q[0], q[0], q[1], q[2])), (f0, cx0, cy0), method="Nelder-Mead",
                              options=dict(maxiter=150, xatol=0.1, fatol=1e-4))
                if melhor is None or r0.fun < melhor.fun:
                    melhor = r0
            x0 = (melhor.x[0], melhor.x[0], melhor.x[1], melhor.x[2])
        r = minimize(custo, x0, method="Nelder-Mead", options=dict(maxiter=400, xatol=0.05, fatol=1e-5))
        fx, fy, cx, cy = r.x
        e = []
        for u, v, z, T in dados:
            P = np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z] @ T[:3, :3].T + T[:3, 3]
            e.append(arv.query(P[::3], workers=THREADS)[0])
        self.K[modo] = tuple(r.x)
        return tuple(r.x), float(np.median(np.concatenate(e)))


# ---------------------------------------------------------------- persistencia (espaco livre)
def persistencia(scan, centros, voxel, nquadros=120, tol_hit=None, tol_free=None):
    """Conta, por voxel, em quantos quadros ele foi visto ocupado (hit) e em quantos foi visto
    VAZIO (o raio da camera atravessa o voxel e acha superficie mais atras: free).

    Mesa giratoria: a peca fica parada no referencial do scan e e vista ocupada em todo quadro que a
    enxerga. Braco, mao, operador e fundo nao giram com a mesa: nesse referencial eles aparecem em
    poucos quadros e, nos demais, o espaco onde estariam foi visto vazio.
    """
    tol_hit = tol_hit or 1.0 * voxel
    tol_free = tol_free or 1.75 * voxel
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    ks = [ks[j] for j in np.linspace(0, len(ks) - 1, min(nquadros, len(ks))).astype(int)]
    C = centros.astype(np.float32)
    hits = np.zeros(len(C), np.int32)
    frees = np.zeros(len(C), np.int32)
    for k in ks:
        r = scan.profundidade(scan.ids[k])
        if r is None:
            continue
        w, h, mask, z, _ = r
        D = np.full(w * h, np.nan, np.float32)
        D[mask] = z
        D = D.reshape(h, w)
        fx, fy, cx, cy = scan.K[int(scan.modo[k])]
        T = scan.T[k]
        Pc = (C - T[:3, 3].astype(np.float32)) @ T[:3, :3].astype(np.float32)
        zc = -Pc[:, 2]
        ok = zc > 20
        zs = np.where(ok, zc, 1.0)
        u = np.rint(Pc[:, 0] * fx / zs + cx).astype(np.int32)
        v = np.rint(-Pc[:, 1] * fy / zs + cy).astype(np.int32)
        ok &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        d = np.full(len(C), np.nan, np.float32)
        d[ok] = D[v[ok], u[ok]]
        diff = d - zc
        hits += (np.abs(diff) <= tol_hit)
        frees += (diff > tol_free)
    return hits, frees


# ---------------------------------------------------------------- segmentacao da peca
def _caixa(mask, pad):
    """Fatias da caixa que envolve os True de `mask`, com folga `pad` (limitada a grade)."""
    idx = np.argwhere(mask)
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def cruzamento_base(A, n, d, celula=2.0, dens_rel=0.25, folga=3.0):
    """Fracao das celulas ocupadas pela peca (acima do plano) que tambem tem material DENSO abaixo dele.

    Uma mesa/prato de verdade deixa a peca so de um lado: ~0. Um plano que fatia a propria peca
    (busto, peca sem mesa) tem corpo dos dois lados: valores altos. Material denso = pelo menos
    `dens_rel` da densidade mediana da peca na mesma celula (ruido espalhado nao conta).
    """
    h = A @ n + d
    u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u); w = np.cross(n, u)
    cel = np.floor(np.c_[A @ u, A @ w] / celula).astype(np.int64)
    key = cel[:, 0] * 1000003 + cel[:, 1]
    ka, ca = np.unique(key[h > folga], return_counts=True)
    kb, cb = np.unique(key[h < -folga], return_counts=True)
    if len(ka) == 0:
        return 0.0
    kb = kb[cb >= dens_rel * np.median(ca)]
    return float(np.isin(kb, ka).sum() / len(ka))


def segmentar_peca(scan, voxel=1.0, faixa_base=1.2, passo=6, log=print,
                   min_votos_frac=0.5, min_tam_frac=0.05, ext_min_mm=15.0, colado_mm=5.0,
                   margem_mm=4.0, transitorio=True, trans_min_free=3, trans_razao=1.0,
                   trans_viz=5, trans_dens=0.6, amostra=None, modo_base="sem", cruza_max=0.10,
                   selecao="hibrido", dominancia=0.5, periferia_mm=0.0, per_altura=15.0, per_banda=2.5, per_abertura=6.0, per_alargue=20.0, grandes_frac=0.02, per_corpo_frac=0.10, ext_max_mm=0.0, so_peca=False):
    """Separa a peca do prato e das pecas de apoio.

    1. Funde uma amostra de todos os quadros no referencial do mundo.
    2. Acha o plano da base (prato) por RANSAC.
    3. Agrupa os pontos acima da base em componentes conexas (grade de voxels).
    4. A peca e a componente para onde a camera mais mira ao longo da captura
       (ponto no eixo optico, na profundidade mediana de cada quadro).
    Retorna um dicionario com o plano, a grade e o conjunto de voxels a manter.
    """
    if amostra is not None:
        A, miras = amostra
    else:
        ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
        lista, miras = [], []
        for k in ks[:: max(1, len(ks) // 150)]:
            P, _ = scan.pontos(k, passo=passo)
            if P is None or len(P) < 100:
                continue
            lista.append(P)
            T = scan.T[k]
            eixo = T[:3, :3] @ np.array([0, 0, -1.0])
            dcam = np.median(np.linalg.norm(P - T[:3, 3], axis=1))
            miras.append(T[:3, 3] + eixo * dcam)
        A = np.vstack(lista)
    n, d, frac = plano_ransac(A[:: max(1, len(A) // 40000)], thr=1.0, it=800)
    # O plano e mesa/prato de verdade ou uma fatia da propria peca? Cada projeto e diferente.
    cruza = cruzamento_base(A, n, d)
    sem_base = modo_base == "sem" or (modo_base == "auto" and cruza > cruza_max)
    d_ref = d                                     # plano real (RANSAC), mesmo quando nao e usado como base
    if sem_base:
        d = -float((A @ n).min()) + 10.0          # plano virtual abaixo de tudo: nada e cortado por ele
    h = A @ n + d
    acima = A[h > faixa_base]
    log(f"   base: {frac*100:.0f}% da amostra no plano | cruzamento {cruza*100:.0f}% -> "
        f"{'SEM base (plano atravessa a peca)' if sem_base else 'mesa/prato valido'} | {len(acima):,} pontos acima")

    # Objetos transitorios (braco, mao, operador): vistos vazios em mais quadros do que ocupados.
    trans = None
    if transitorio and scan is not None:
        v2 = 2.0
        while float(np.prod((acima.max(0) - acima.min(0) + 4 * v2) / v2)) > MAX_VOXELS_TRANS:
            v2 *= 1.25
        lo2 = acima.min(0) - 2 * v2
        idx2 = np.floor((acima - lo2) / v2).astype(np.int64)
        uniq, inv = np.unique(idx2, axis=0, return_inverse=True)
        hits, frees = persistencia(scan, (uniq + 0.5) * v2 + lo2, v2)
        marc = (frees >= trans_min_free) & (frees > trans_razao * hits)
        # Na peca os voxels marcados sao esparsos (bordas e cantos vistos de perto). Objeto
        # transitorio e uma regiao inteira marcada: so vale quem esta cercado por marcados.
        dims2 = uniq.max(0) + 1
        occ_g = np.zeros(dims2, np.float32); occ_g[tuple(uniq.T)] = 1
        mar_g = np.zeros(dims2, np.float32); mar_g[tuple(uniq.T)] = marc
        viz = ndimage.uniform_filter(mar_g, size=trans_viz, mode="constant") / np.maximum(
            ndimage.uniform_filter(occ_g, size=trans_viz, mode="constant"), 1e-6)
        t_v = marc & (viz[tuple(uniq.T)] >= trans_dens)
        trans = dict(lo=lo2, voxel=v2, uniq=uniq, flag=t_v, hits=hits, frees=frees)
        log(f"   transitorio: {t_v.sum():,} de {len(t_v):,} voxels de 2 mm vistos vazios (braco/mao/operador)")
        acima = acima[~t_v[inv.ravel()]]
        log(f"   pontos acima apos remover transitorios: {len(acima):,}")

    # Cenas grandes (1 m ou mais, ou pontos soltos longe) viram grades de bilhoes de voxels de 1 mm e
    # estouram a RAM (36 GB medidos num processo). Acima de MAX_VOXELS o voxel cresce ate caber.
    voxel0 = voxel
    while float(np.prod((acima.max(0) - acima.min(0) + 6 * voxel) / voxel)) > MAX_VOXELS:
        voxel *= 1.25
    if voxel != voxel0:
        log(f"   cena grande: voxel {voxel0:g} -> {voxel:.2f} mm para a grade caber na memoria")
    lo = acima.min(0) - 3 * voxel
    dims = np.ceil((acima.max(0) + 3 * voxel - lo) / voxel).astype(int)
    idx = np.floor((acima - lo) / voxel).astype(int)
    grade = np.zeros(dims, bool)
    grade[tuple(idx.T)] = True
    rot, ncomp = ndimage.label(grade, structure=np.ones((3, 3, 3)))
    tam = np.bincount(rot[tuple(idx.T)], minlength=ncomp + 1)
    if ext_max_mm > 0:
        # gabarito/placa de marcadores maior que a peca: sai da disputa (nunca e peca nem parte dela)
        grande = [c for c, sl in enumerate(ndimage.find_objects(rot), start=1)
                  if sl is not None and max(x.stop - x.start for x in sl) * voxel > ext_max_mm]
        if grande:
            log(f"   ext-max-mm {ext_max_mm:g}: {len(grande)} componente(s) maiores ignorados {grande[:8]}")
        tam[grande] = 0
        rot[np.isin(rot, grande)] = 0

    # mira da camera -> componente
    occ = np.argwhere(rot > 0)
    arv = cKDTree(occ * voxel + lo + voxel / 2)
    dist, j = arv.query(np.array(miras), distance_upper_bound=8.0, workers=THREADS)
    ok = np.isfinite(dist)
    comps = rot[tuple(occ[j[ok]].T)]
    votos = np.bincount(comps, minlength=ncomp + 1)
    ordem = np.argsort(-tam)
    log("   objetos acima da base (pontos | votos de mira):")
    for c in ordem[:6]:
        if c == 0 or tam[c] == 0:
            continue
        pts = acima[rot[tuple(idx.T)] == c]
        ext = np.ptp(pts, 0)
        log(f"     objeto {c}: {tam[c]:,} pts | {votos[c]} miras | extensao {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
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
        ext_min_mm = 1e9                          # nada "colado" entra: so o componente escolhido
    partes = [c for c in range(1, ncomp + 1)
              if c == peca or (votos[c] >= min_votos_frac * votos[peca] and tam[c] >= min_tam_frac * tam[peca])
              # a peca costuma aparecer partida em varios componentes grandes (amostra esparsa, fendas):
              # todo componente grande em relacao ao maior tambem entra, com ou sem voto de mira
              or (grandes_frac > 0 and tam[c] >= grandes_frac * tam[maior])]
    if so_peca:
        partes = [peca]
    # Componente grande colado na peca (ex.: cubo central separado por uma fresta) tambem e peca,
    # mesmo sem voto de mira. Marcadores sao pequenos (~6 mm) e ficam de fora.
    # Em cena suja (braco, blocos encostados) so entra o que esta DENTRO da caixa da peca, como o
    # cubo central; objeto que se estende para fora dela e do ambiente, nao da peca.
    # A distancia so interessa perto da peca: calcula numa caixa em volta dela (cenas grandes tem
    # grades de centenas de milhoes de voxels e a transformada de distancia inteira leva minutos).
    def so_partes():
        tab = np.zeros(ncomp + 1, bool)
        tab[partes] = True
        return tab[rot]
    pm = so_partes()
    caixa = _caixa(pm, int(np.ceil(colado_mm / voxel)) + 2)
    dist_sub = ndimage.distance_transform_edt(~pm[caixa])
    nuc = np.argwhere(pm)
    nlo, nhi = nuc.min(0) - int(round(3 / voxel)), nuc.max(0) + int(round(3 / voxel))
    for c, sl in enumerate(ndimage.find_objects(rot), start=1):
        if c in partes or sl is None:
            continue
        ext = max(s.stop - s.start for s in sl) * voxel
        if ext < ext_min_mm:
            continue
        inter = tuple(slice(max(a.start, b.start), min(a.stop, b.stop)) for a, b in zip(sl, caixa))
        if any(i.stop <= i.start for i in inter):
            continue                          # longe da peca: nao esta colado
        loc = tuple(slice(i.start - b.start, i.stop - b.start) for i, b in zip(inter, caixa))
        no_c = rot[inter] == c
        if not no_c.any() or dist_sub[loc][no_c].min() * voxel > colado_mm:
            continue
        vc = np.argwhere(rot[sl] == c) + np.array([x.start for x in sl])
        dentro = np.all((vc >= nlo) & (vc <= nhi), axis=1).mean()
        if dentro >= 0.9:
            partes.append(c)
    # Limpeza branda: mantem tudo que esta a ate `margem_mm` da peca e remove so o que esta longe
    # (marcadores, prato distante, apoios).
    nucleo = so_partes()
    cx = _caixa(nucleo, int(np.ceil(margem_mm / voxel)) + 2)
    manter = np.zeros(nucleo.shape, bool)
    manter[cx] = ndimage.distance_transform_edt(~nucleo[cx], sampling=voxel) <= margem_mm
    # Pegada da peca no plano da base: sob ela, a face de baixo encosta no prato e nao e cortada.
    u = np.cross(n, [1.0, 0, 0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    centros = np.argwhere(nucleo) * voxel + lo + voxel / 2
    uv = np.c_[centros @ u, centros @ v]
    uv_lo = uv.min(0) - 3
    pegada = np.zeros(np.ceil((uv.max(0) + 3 - uv_lo) / 1.0).astype(int) + 1, bool)
    ij = np.floor((uv - uv_lo) / 1.0).astype(int)
    pegada[tuple(ij.T)] = True
    # A peca e vazada (cavidades, anel): o piso dentro dela fica na altura do prato e e peca, nao prato.
    # Fecha fendas e preenche os buracos da projecao.
    pegada = ndimage.binary_dilation(pegada, iterations=1)
    pegada = ndimage.binary_closing(np.pad(pegada, 8), iterations=6)[8:-8, 8:-8]
    pegada = ndimage.binary_fill_holes(pegada)
    # Altura local da peca em cada celula da pegada. A amostra dos quadros e esparsa e a casca fina
    # da face de cima fica com buracos no nucleo 3D; a coluna vertical sob a pegada nao tem esse problema.
    hcent = centros @ n + d
    hcol = np.full(pegada.shape, -np.inf)
    np.maximum.at(hcol, tuple(ij.T), hcent)
    hcol = ndimage.maximum_filter(hcol, size=9)
    vazio = ~np.isfinite(hcol) & pegada
    for _ in range(20):                      # propaga para celulas da pegada sem nucleo (buracos)
        if not vazio.any():
            break
        hcol = np.where(vazio, ndimage.maximum_filter(hcol, size=5), hcol)
        vazio = ~np.isfinite(hcol) & pegada
    hcol[~np.isfinite(hcol)] = float(hcent.max())
    # Periferia da mesa: no modo sem base, a faixa do plano real LONGE da peca e mesa (so quando a peca
    # tem altura de verdade acima do plano; peca achatada nao tem como distinguir de mesa).
    per = None
    if sem_base and periferia_mm > 0 and cruza > cruza_max:
        log(f"   periferia da mesa DESLIGADA: o plano atravessa a peca (cruzamento {cruza*100:.0f}% > {cruza_max*100:.0f}%), nao e mesa")
    elif sem_base and periferia_mm > 0:
        h_real = centros @ n + d_ref
        alta = h_real > 4.0
        # So vale se a peca tem corpo de verdade: >= per_corpo_frac do volume mantido acima de `per_altura`.
        # Peca achatada (ou plano que nao e mesa) tem quase tudo rente ao plano e a regra a comeria inteira.
        corpo_frac = float((h_real > per_altura).mean())
        if alta.any() and corpo_frac < per_corpo_frac:
            log(f"   periferia da mesa DESLIGADA: so {corpo_frac*100:.0f}% da peca passa de {per_altura:g} mm acima do plano (peca achatada)")
        elif alta.any() and (h_real.max() >= per_altura):
            # Pegada = so o CORPO da peca. O componente da peca costuma vir colado na mesa, e relevos e
            # marcadores da mesa (>4 mm) entrariam na pegada e a protegeriam. A abertura apaga o que e fino
            # e o corpo e o que tem altura real; so as manchas 2D que contem corpo ficam.
            fp = np.zeros(pegada.shape, bool)
            fp[tuple(ij[alta].T)] = True
            corpo = np.zeros(pegada.shape, bool)
            corpo[tuple(ij[h_real > per_altura].T)] = True
            r = int(round(per_abertura))
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            fp = ndimage.binary_opening(np.pad(fp, r + 2), structure=(xx ** 2 + yy ** 2) <= r * r)[r + 2:-(r + 2), r + 2:-(r + 2)]
            # relevo baixo (4 a `per_altura` mm) so conta como peca ate `per_alargue` mm do corpo: cobre bases
            # alargadas sem deixar o relevo da mesa, que encosta no corpo, proteger a mesa inteira
            perto = ndimage.distance_transform_edt(~corpo) <= per_alargue
            fp = (fp & perto) | ndimage.binary_dilation(corpo, iterations=1)   # o corpo nunca sai da pegada
            fp = ndimage.binary_closing(np.pad(fp, 8), iterations=4)[8:-8, 8:-8]
            dist2 = ndimage.distance_transform_edt(~fp).astype(np.float32)
            per = dict(n=n, d=d_ref, uv_lo=uv_lo, dist=dist2, mm=periferia_mm, banda=per_banda)
    if sem_base:
        pegada = np.zeros_like(pegada)             # sem prato nao ha "piso" nem "coluna" sob a peca
    exts = {c: tuple(int(round(np.ptp(np.argwhere(rot == c), 0)[i] * voxel)) for i in range(3)) for c in partes}
    log(f"   peca escolhida: objeto {peca} ({votos[peca]} de {ok.sum()} miras) | componentes mantidos: {partes} "
        f"(mm {exts}, votos {[int(votos[c]) for c in partes]}) | margem {margem_mm:g} mm")
    return dict(n=n, d=d, faixa_base=faixa_base, voxel=voxel, lo=lo, manter=manter,
                sem_base=bool(sem_base), cruza=cruza, frac_plano=float(frac), periferia=per,
                u=u, v=v, uv_lo=uv_lo, pegada=pegada, hcol=hcol, trans=trans,
                h_max=float((centros @ n + d).max() + voxel + 1.0),   # marcadores colados ficam acima do topo
                peca=peca, votos=votos.tolist(), tam=tam.tolist())


def mascara_manter(P, seg):
    h = P @ seg["n"] + seg["d"]
    idx = np.floor((P - seg["lo"]) / seg["voxel"]).astype(int)
    dentro = np.all((idx >= 0) & (idx < np.array(seg["manter"].shape)), axis=1)
    ok = np.zeros(len(P), bool)
    ii = idx[dentro]
    ok[dentro] = seg["manter"][tuple(ii.T)]
    # Perto da base: so vale sob a pegada da peca (corpo encostado no prato); fora dela e prato.
    ij = np.floor((np.c_[P @ seg["u"], P @ seg["v"]] - seg["uv_lo"])).astype(int)
    na_pegada = np.all((ij >= 0) & (ij < np.array(seg["pegada"].shape)), axis=1)
    na_pegada[na_pegada] = seg["pegada"][tuple(ij[na_pegada].T)]
    acima = h > seg["faixa_base"]
    # Sob a pegada e junto do prato e a face de baixo/piso da peca: vale mesmo longe do nucleo em 3D.
    piso = na_pegada & (h > -1.5) & (h <= seg["faixa_base"])
    # Coluna vertical sob a pegada ate a altura local da peca (+3 mm): vale mesmo onde o nucleo 3D
    # tem buracos (face de cima fina). Braco/objeto que passa mais alto fica de fora.
    coluna = np.zeros(len(P), bool)
    if na_pegada.any():
        hl = seg["hcol"][tuple(ij[na_pegada].T)]
        coluna[na_pegada] = (h[na_pegada] > -1.5) & (h[na_pegada] <= hl + 3.0)
    ok = (ok & acima | piso | coluna | (ok & na_pegada & (h > -1.5))) & (h < seg["h_max"] + 2.0)
    if seg.get("trans") is not None:
        ok &= ~_em_transitorio(P, seg["trans"])
    # Referencia: a peca fundida pelo proprio Revo Scan (fuse.ply). Tudo que esta perto dela e peca,
    # inclusive a face de baixo e detalhes que as regras de altura/componente cortariam.
    if seg.get("ref_arvore") is not None:
        d, _ = seg["ref_arvore"].query(P, distance_upper_bound=seg["ref_margem"], workers=THREADS)
        ok |= np.isfinite(d)
    # A periferia da mesa vale por ultimo: o fuse.ply do Revo inclui a mesa e nao pode ressuscita-la.
    per = seg.get("periferia")
    if per is not None:
        hr = P @ per["n"] + per["d"]
        faixa = np.abs(hr) <= per["banda"]
        if faixa.any():
            ijp = np.floor(np.c_[P[faixa] @ seg["u"], P[faixa] @ seg["v"]] - per["uv_lo"]).astype(int)
            dentro = np.all((ijp >= 0) & (ijp < np.array(per["dist"].shape)), axis=1)
            dd = np.full(len(ijp), np.inf, np.float32)
            dd[dentro] = per["dist"][tuple(ijp[dentro].T)]
            kill = np.zeros(len(P), bool)
            kill[np.flatnonzero(faixa)] = dd > per["mm"]
            ok &= ~kill
    return ok


def _em_transitorio(P, trans):
    """True para os pontos que caem num voxel marcado como transitorio."""
    idx = np.floor((P - trans["lo"]) / trans["voxel"]).astype(np.int64)
    uniq = trans["uniq"]
    dims = uniq.max(0) + 1
    chave = lambda a: (a[:, 0] * dims[1] + a[:, 1]) * dims[2] + a[:, 2]
    dentro = np.all((idx >= 0) & (idx < dims), axis=1)
    ks = chave(uniq[trans["flag"]])
    res = np.zeros(len(P), bool)
    res[dentro] = np.isin(chave(idx[dentro]), ks)
    return res


def registrar_fuse(scan, seg, caminho, log=print, iteracoes=200000):
    """Registra um fuse.ply no referencial dos quadros. Retorna (T, fitness, rmse) ou None.

    O alvo sao os pontos dos quadros que a mascara mantem (a peca e o que a rodeia de perto).
    """
    import open3d as o3d
    F = np.asarray(o3d.io.read_point_cloud(caminho).points, float)
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]][::6]
    amostra = [p for p in (scan.pontos(k, passo=3)[0] for k in ks) if p is not None]
    W = np.vstack(amostra)
    W = W[mascara_manter(W, seg)]
    return _registrar_nuvens(F, W, log, iteracoes)


def adicionar_referencia_fuse(scan, seg, margem_mm=2.5, log=print):
    """Registra o fuse.ply do scan no referencial dos quadros e o guarda em `seg` como referencia.

    Sem fuse.ply ou com registro ruim, nao altera `seg` (a limpeza segue so com as regras geometricas).
    """
    import open3d as o3d
    caminho = os.path.join(scan.pasta, "fuse.ply")
    if not os.path.exists(caminho):
        log("   referencia fuse.ply ausente: seguindo so com as regras geometricas")
        return seg
    F = np.asarray(o3d.io.read_point_cloud(caminho).points, float)
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]][::6]
    amostra = [p for p in (scan.pontos(k, passo=3)[0] for k in ks) if p is not None]
    W = np.vstack(amostra)
    W = W[mascara_manter(W, seg)]

    def nuvem(X, v):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)).voxel_down_sample(v)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * v, max_nn=30))
        return p
    reg = o3d.pipelines.registration
    src, dst = nuvem(F, 0.8), nuvem(W, 0.8)
    # o fuse.ply costuma ja estar no referencial dos quadros: ICP direto primeiro, RANSAC so se nao servir
    pp = reg.TransformationEstimationPointToPlane()
    ic = reg.registration_icp(src, dst, 2.0, np.eye(4), pp)
    ic = reg.registration_icp(src, dst, 0.8, ic.transformation, pp)
    if ic.fitness < 0.85 or ic.inlier_rmse > 0.6:
        feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=100))
        r = reg.registration_ransac_based_on_feature_matching(
            src, dst, feat(src), feat(dst), True, 3.0, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnDistance(3.0)], reg.RANSACConvergenceCriteria(200000, 0.999))
        ic = reg.registration_icp(src, dst, 0.8, r.transformation, pp)
    log(f"   registro do fuse.ply: fitness {ic.fitness:.3f} | rmse {ic.inlier_rmse:.3f} mm")
    if ic.fitness < 0.85 or ic.inlier_rmse > 0.6:
        log("   registro ruim: referencia fuse.ply ignorada")
        return seg
    T = ic.transformation
    Fw = (T[:3, :3] @ F.T).T + T[:3, 3]
    seg["ref_arvore"] = cKDTree(np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Fw))
                                           .voxel_down_sample(0.3).points))
    seg["ref_margem"] = margem_mm
    return seg


def _registrar_nuvens(F, W, log=print, iteracoes=200000):
    """Registra F em W (FPFH + RANSAC + ICP ponto-plano). Retorna (T, fitness, rmse)."""
    import open3d as o3d
    reg = o3d.pipelines.registration

    def nuvem(X, v):
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(X)).voxel_down_sample(v)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * v, max_nn=30))
        return p
    src, dst = nuvem(F, 0.8), nuvem(W, 0.8)
    feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=6, max_nn=100))
    r = reg.registration_ransac_based_on_feature_matching(
        src, dst, feat(src), feat(dst), True, 3.0, reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnDistance(3.0)], reg.RANSACConvergenceCriteria(iteracoes, 0.999))
    ic = reg.registration_icp(src, dst, 0.8, r.transformation, reg.TransformationEstimationPointToPlane())
    log(f"   registro do fuse.ply: fitness {ic.fitness:.3f} | rmse {ic.inlier_rmse:.3f} mm")
    return ic.transformation, float(ic.fitness), float(ic.inlier_rmse)


def limpar_fuse(scan, seg, fuse_entrada, fuse_saida, log=print):
    """Limpa a nuvem FUNDIDA pelo Revo (fuse.ply) com a mesma mascara de selecao da peca.

    Em scans sem marcadores o Revo se orienta pela geometria dos objetos de rastreamento em volta
    da peca: apagar esses objetos dos quadros antes da fusao faz a fusao perder regioes da peca.
    Por isso a limpeza e feita DEPOIS da fusao, na nuvem fundida, que fica no referencial do scan.
    Retorna dict com os numeros, ou None se o registro for ruim (nada e gravado).
    """
    hdr, arr = ler_ply(fuse_entrada)
    F = arr[:, :3].astype(float)
    r = registrar_fuse(scan, seg, fuse_entrada, log)
    T, fit, rmse = r
    if fit < 0.85 or rmse > 0.6:
        log("   registro ruim: fuse.ply NAO foi limpo")
        return None
    Fw = (T[:3, :3] @ F.T).T + T[:3, 3]
    ok = mascara_manter(Fw, seg)
    gravar_ply(fuse_saida, hdr, arr[ok])
    log(f"   fuse.ply: {len(arr):,} -> {ok.sum():,} pontos ({ok.mean()*100:.1f}% mantidos)")
    return dict(antes=int(len(arr)), depois=int(ok.sum()), fitness=fit, rmse=rmse)


# ---------------------------------------------------------------- gravacao
def filtro_pixels(D, tol=1.0, comp_min=30):
    """Mascara dos pixels a manter num mapa de profundidade D (h x w, NaN onde invalido).

    Remove (1) pixels voadores: o pixel desvia mais de `tol` mm da media dos dois vizinhos em
    TODAS as direcoes (4 pares opostos). Superficie lisa, inclinada ou curva fica ~0 em alguma
    direcao e borda real fica ~0 ao longo da borda; so ruido entre a peca e o fundo desvia nas
    quatro. (2) manchas pequenas de pixels desconectados do resto (< `comp_min` px).
    """
    h, w = D.shape
    pad = np.pad(D, 1, constant_values=np.nan)
    sh = lambda dy, dx: pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
    res = np.full(D.shape, np.inf, np.float32)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        r = np.abs(D - (sh(dy, dx) + sh(-dy, -dx)) / 2)
        res = np.minimum(res, np.where(np.isnan(r), np.inf, r))
    # sem nenhum par de vizinhos validos nao ha como julgar: nao e voador (so a regra das manchas atua)
    val = ~np.isnan(D) & ~(np.isfinite(res) & (res > tol))
    lab, n = ndimage.label(val, structure=np.ones((3, 3)))
    tam = np.bincount(lab.ravel(), minlength=n + 1)
    tam[0] = 0
    return val & (tam[lab] >= comp_min)


def ilhas_3d(A, voxel=1.0, gap_vox=2, frac_min=0.02):
    """Grade booleana dos voxels que pertencem a peca principal ou a pedacos grandes dela.

    A: amostra de pontos mantidos (N x 3). Voxels ocupados sao dilatados `gap_vox` voxels para
    que fendas finas nao separem a peca; componente com menos de `frac_min` do maior e ilha solta.
    """
    if np.prod(np.ptp(A, 0) / voxel + 8) > 4e8:     # cena muito grande: grade mais grossa
        voxel *= 2
    lo = A.min(0) - (gap_vox + 2) * voxel
    idx = np.floor((A - lo) / voxel).astype(np.int64)
    g = np.zeros(idx.max(0) + gap_vox + 3, bool)
    g[tuple(idx.T)] = True
    gd = ndimage.binary_dilation(g, structure=np.ones((3, 3, 3)), iterations=gap_vox)
    lab, n = ndimage.label(gd, structure=np.ones((3, 3, 3)))
    tam = np.bincount(lab[tuple(idx.T)], minlength=n + 1)
    tam[0] = 0
    bom = tam >= frac_min * tam.max()
    bom[0] = False
    return dict(lo=lo, voxel=voxel, grade=bom[lab], n_comp=int((tam > 0).sum()),
                n_removidas=int(((tam > 0) & ~bom).sum()), tam_max=int(tam.max()))


def mascara_ilhas(P, ilh):
    idx = np.floor((P - ilh["lo"]) / ilh["voxel"]).astype(np.int64)
    dentro = np.all((idx >= 0) & (idx < np.array(ilh["grade"].shape)), axis=1)
    ok = np.zeros(len(P), bool)
    ok[dentro] = ilh["grade"][tuple(idx[dentro].T)]
    return ok


PLY_ANTIGOS = ("fuse.ply", "fuse_mesh.ply", "mark.ply")


def limpar_scan(scan, seg, destino_scan, log=print, pixels=True, ilhas=True, min_mantido=0.0, sem_ply_antigos=False,
                zerar_vazios=False, min_pix_quadro=1):
    """Grava a copia do scan com os pixels fora da peca apagados em cada quadro.

    Alem da mascara da peca (`mascara_manter`): `pixels` tira pixels voadores e manchas 2D
    minusculas de cada mapa de profundidade; `ilhas` tira pedacos 3D soltos da peca principal.
    """
    os.makedirs(destino_scan, exist_ok=True)
    for item in os.listdir(scan.pasta):
        src = os.path.join(scan.pasta, item)
        if item == "cache":
            continue
        if sem_ply_antigos and item in PLY_ANTIGOS:        # nuvem/malha/marcadores ja fundidos, com a mesa
            continue
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(destino_scan, item), dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(destino_scan, item))
    db = scan.db + ".limpo"
    scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
    shutil.copy(scan.db, db)
    cn = sqlite3.connect(db)
    vazios, oks, amostra, tirados = [], {}, [], []
    n_total = n_mask = n_pix = 0
    # passada 1: mascara da peca + filtro de pixels por quadro
    for k, i in enumerate(scan.ids):
        if not scan.tem_prof[k]:
            continue
        w, h, mask, z, quatro = scan.profundidade(i)
        P, pix = scan.pontos(k)
        ok = mascara_manter(P, seg)
        n_total += len(pix)
        n_mask += int(ok.sum())
        base = ok
        if pixels and ok.any():
            D = np.full(w * h, np.nan, np.float32)
            D[mask] = z
            ok = ok & filtro_pixels(D.reshape(h, w)).ravel()[pix]
        n_pix += int(ok.sum())
        oks[k] = ok
        par = np.arange(len(P)) % 2 == 0          # mesma amostra (pixels pares) nos dois lados
        amostra.append(P[ok & par].astype(np.float32))
        tirados.append(P[base & ~ok & par].astype(np.float32))
    if n_pix < min_mantido * max(1, n_total):
        log(f"   REVISAR: a limpeza manteria so {n_pix/max(1,n_total)*100:.0f}% dos pontos (minimo {min_mantido*100:.0f}%); "
            f"escolha da peca suspeita, scan copiado sem alteracao")
        cn.close()
        os.remove(db)
        shutil.copytree(scan.pasta, destino_scan, dirs_exist_ok=True)
        return dict(revisar=True, antes=n_total, depois=n_pix)
    A, T0 = np.vstack(amostra), np.vstack(tirados)
    # ilhas 3D soltas
    ilh = ilhas_3d(A) if ilhas and len(A) else None
    seg["ilhas"] = ilh          # render_comparacao mostra o resultado final
    if ilh is not None:
        log(f"   ilhas 3D: {ilh['n_comp']} componentes, {ilh['n_removidas']} removidas (menores que 2% da peca)")
    # auditoria: dos pontos que os filtros novos tiraram, quantos estavam colados na superficie do fuse.ply
    # (indica risco de perder superficie real; o restante e ruido fora da superficie)
    def auditar(R, rotulo):
        if seg.get("ref_arvore") is not None and len(R):
            d_ref = seg["ref_arvore"].query(R, distance_upper_bound=0.4, workers=THREADS)[0]
            log(f"   auditoria {rotulo}: {len(R):,} pontos (amostra) | {np.isfinite(d_ref).sum():,} "
                f"({np.isfinite(d_ref).mean()*100:.1f}%) a menos de 0,4 mm do fuse.ply")
    auditar(T0, "pixels voadores")
    if ilh is not None:
        auditar(A[~mascara_ilhas(A, ilh)], "ilhas")
    # passada 2: grava
    tot_antes = tot_depois = 0
    for k, i in enumerate(scan.ids):
        if not scan.tem_prof[k]:
            continue
        w, h, mask, z, quatro = scan.profundidade(i)
        P, pix = scan.pontos(k)
        ok = oks[k]
        if ilh is not None:
            ok = ok & mascara_ilhas(P, ilh)
        tot_antes += len(pix)
        if ok.sum() < min_pix_quadro:
            vazios.append(k)       # quadro so com prato/apoio
            if zerar_vazios:       # remove a profundidade do quadro (estado que o Revo ja usa em quadros sem profundidade)
                cn.execute("delete from FrameRGBD where id=?", (i,))
            else:                  # padrao: mantem o original
                tot_depois += len(pix)
            continue
        nova = np.zeros(w * h, bool)
        nova[pix[ok]] = True
        dados = (struct.pack("<III", w, h, int(ok.sum()))
                 + np.packbits(nova, bitorder="little").tobytes() + quatro
                 + z[ok].astype("<f4").tobytes())
        cn.execute("update FrameRGBD set data=? where id=?", (dados, i))
        tot_depois += int(ok.sum())
    cn.commit()
    integ = cn.execute("pragma integrity_check").fetchone()[0]
    # consolida o journal (o Revo Scan usa modo WAL) dentro do arquivo principal
    cn.execute("pragma wal_checkpoint(TRUNCATE)")
    cn.close()
    for ext in ("-wal", "-shm", "-journal"):
        if os.path.exists(db + ext):
            raise RuntimeError("sobrou arquivo de journal " + ext)
    with open(db, "r+b") as f:
        f.write(scan.sig)
    os.makedirs(os.path.join(destino_scan, "cache"), exist_ok=True)
    shutil.move(db, os.path.join(destino_scan, "cache", "frames.dataset"))

    # raw_preview filtrado pela mesma regra (esta no referencial do mundo)
    hdr, prev = ler_ply(os.path.join(scan.pasta, "raw_preview.ply"))
    okp = mascara_manter(prev[:, :3].astype(float), seg)
    if ilh is not None:
        okp &= mascara_ilhas(prev[:, :3].astype(float), ilh)
    gravar_ply(os.path.join(destino_scan, "raw_preview.ply"), hdr, prev[okp])
    log(f"   pontos dos quadros: {tot_antes:,} -> {tot_depois:,} ({tot_depois/max(1,tot_antes)*100:.0f}% mantidos; "
        f"peca {n_mask:,} -> pixels voadores {n_mask-n_pix:,} tirados) | preview {len(prev):,} -> {okp.sum():,} | integridade: {integ}")
    if vazios:
        log(f"   AVISO: {len(vazios)} quadros sem ponto da peca " + ("tiveram a profundidade removida" if zerar_vazios else "foram mantidos como estavam"))
    return dict(antes=tot_antes, depois=tot_depois, vazios=len(vazios), integridade=integ,
                voadores=n_mask - n_pix, ilhas_removidas=ilh["n_removidas"] if ilh else 0)


def gravar_revo(origem_revo, destino_proj, nome, nos):
    r = json.load(open(origem_revo, encoding="utf-8"))
    r["guid"] = str(uuid.uuid4())
    r["name"] = nome
    r["edit_time"] = str(int(time.time()))
    r["nodes"] = [n for n in r["nodes"] if n["guid"] in nos]
    # igual ao Revo Scan: soma so os arquivos da pasta data
    tam = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(destino_proj, "data")) for f in fs)
    r["size"] = str(tam)
    with open(os.path.join(destino_proj, nome + ".revo"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=4)
    return r


def render_comparacao(scan, seg, path, nquadros=60):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    ks = ks[:: max(1, len(ks) // nquadros)]
    P = np.vstack([p for k in ks if (p := scan.pontos(k, passo=6)[0]) is not None])
    ok = mascara_manter(P, seg)
    if seg.get("ilhas") is not None:
        ok &= mascara_ilhas(P, seg["ilhas"])
    n = seg["n"]
    u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    L = np.c_[P @ u, P @ v, P @ n + seg["d"]]
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), dpi=90)
    for col, (titulo, m) in enumerate((("ANTES (todos os pontos)", np.ones(len(L), bool)), ("DEPOIS (so a peca)", ok))):
        ax[0, col].scatter(L[m, 0], L[m, 1], s=0.2, c=L[m, 2], cmap="viridis", linewidths=0)
        ax[0, col].set_aspect("equal"); ax[0, col].set_title(titulo + ": vista de cima")
        ax[1, col].scatter(L[m, 0], L[m, 2], s=0.2, c="#333", linewidths=0)
        ax[1, col].set_aspect("equal"); ax[1, col].set_title(titulo + ": vista lateral")
    for a in ax[:, 1]:
        a.set_xlim(ax[0, 0].get_xlim())
    ax[0, 1].set_ylim(ax[0, 0].get_ylim()); ax[1, 1].set_ylim(ax[1, 0].get_ylim())
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
