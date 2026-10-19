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
import time
import uuid

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import cKDTree

SIG_REVO = b"RPAlgo_file_sys\x00"
SIG_SQL = b"SQLite format 3\x00"
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
        self.K = {}

    def fechar(self):
        try:
            self.c.close()
        except Exception:
            pass

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

    def calibrar(self, modo, x0=K_PADRAO, nquadros=8):
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
                e.append(np.minimum(arv.query(P)[0], 3.0))
            return float(np.mean(np.concatenate(e)))

        r = minimize(custo, x0, method="Nelder-Mead", options=dict(maxiter=400, xatol=0.05, fatol=1e-5))
        fx, fy, cx, cy = r.x
        e = []
        for u, v, z, T in dados:
            P = np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z] @ T[:3, :3].T + T[:3, 3]
            e.append(arv.query(P[::3])[0])
        self.K[modo] = tuple(r.x)
        return tuple(r.x), float(np.median(np.concatenate(e)))


# ---------------------------------------------------------------- segmentacao da peca
def achar_base(A, cams, candidatos=4, min_frac=0.03, min_cam=0.8, max_abaixo=0.05, log=print):
    """Procura o plano do prato entre os maiores planos da cena.

    Um plano so e aceito como base se:
      - as cameras estao do lado de cima dele (>= 80%): o scanner sempre olha o
        prato de cima. Isso rejeita faces planas da propria peca, que sao vistas
        do lado de fora (teste real: face de cima do difusor tinha 0% de cameras
        do lado da peca);
      - quase nenhum ponto fica abaixo dele (<= 5%);
      - tem pelo menos 3% dos pontos da amostra.
    Retorna (n, d) com a normal apontando para as cameras, ou None se nao achar.
    """
    resto = A
    for i in range(candidatos):
        if len(resto) < 1000:
            break
        n, d, _ = plano_ransac(resto[:: max(1, len(resto) // 40000)], thr=1.0, it=800)
        if np.mean(cams @ n + d > 0) < 0.5:          # normal para o lado das cameras
            n, d = -n, -d
        hA = A @ n + d
        frac = float(np.mean(np.abs(hA) < 1.0))
        cam_ok = float(np.mean(cams @ n + d > 0))
        abaixo = float(np.mean(hA < -2.5))
        aceito = frac >= min_frac and cam_ok >= min_cam and abaixo <= max_abaixo
        log(f"   plano {i}: {frac*100:.1f}% dos pontos | cameras acima {cam_ok*100:.0f}% | "
            f"pontos abaixo {abaixo*100:.1f}% -> {'BASE' if aceito else 'rejeitado'}")
        if aceito:
            return n, d
        resto = resto[np.abs(resto @ n + d) >= 1.5]
    log("   nenhum plano passou como base: nada sera removido por altura, so objetos soltos")
    return None

def segmentar_peca(scan, voxel=1.0, faixa_base=2.5, passo=6, raio_centro=0.15, log=print):
    """Separa a peca do prato e das pecas de apoio.

    1. Funde uma amostra dos quadros (ate cerca de 150) no referencial do mundo.
    2. Acha o plano da base (prato), so aceitando planos vistos de cima (achar_base).
    3. Agrupa os pontos acima da base em componentes conexas (grade de voxels).
    4. A peca e a componente que mais aparece no CENTRO da imagem ao longo da
       captura: cada quadro vota no objeto com mais pixels num circulo de raio
       raio_centro * largura em volta do ponto principal (cx, cy). O operador sempre
       aponta o scanner para a peca. Desempate pelo tamanho.
       (Versao anterior usava um ponto no eixo optico na profundidade mediana; falhou
       no teste com gabarito quando o prato dominava a imagem e escolheu a esfera de apoio.)
    Retorna um dicionario com o plano, a grade e o conjunto de voxels a manter.
    """
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    amostra, centrais, cams = [], [], []
    for k in ks[:: max(1, len(ks) // 150)]:
        r = scan.profundidade(scan.ids[k])
        if r is None:
            continue
        w = r[0]
        P, pix = scan.pontos(k, passo=passo)
        if P is None or len(P) < 100:
            continue
        amostra.append(P)
        cams.append(scan.T[k][:3, 3])
        fx, fy, cx, cy = scan.K[int(scan.modo[k])]
        v, u = np.divmod(pix, w)
        centro = (u - cx) ** 2 + (v - cy) ** 2 < (raio_centro * w) ** 2
        centrais.append(P[centro])
    A = np.vstack(amostra)
    base = achar_base(A, np.array(cams), log=log)
    if base is None:
        n, d = np.zeros(3), 1e9          # sem base: todo ponto conta como "acima"
        acima = A
    else:
        n, d = base
        acima = A[A @ n + d > faixa_base]
        log(f"   {len(acima):,} pontos acima da base")

    lo = acima.min(0) - 3 * voxel
    dims = np.ceil((acima.max(0) + 3 * voxel - lo) / voxel).astype(int)
    idx = np.floor((acima - lo) / voxel).astype(int)
    grade = np.zeros(dims, bool)
    grade[tuple(idx.T)] = True
    rot, ncomp = ndimage.label(grade, structure=np.ones((3, 3, 3)))
    tam = np.bincount(rot[tuple(idx.T)], minlength=ncomp + 1)

    def componente(Q):
        ii = np.floor((Q - lo) / voxel).astype(int)
        dentro = np.all((ii >= 0) & (ii < np.array(dims)), axis=1)
        c = np.zeros(len(Q), int)
        c[dentro] = rot[tuple(ii[dentro].T)]
        if base is not None:
            c[Q @ n + d <= faixa_base] = 0
        return c

    votos = np.zeros(ncomp + 1, int)
    validos = 0
    for Q in centrais:
        c = componente(Q)
        c = c[c > 0]
        if len(c) < 20:
            continue
        votos[np.bincount(c).argmax()] += 1
        validos += 1
    ordem = np.argsort(-tam)
    log("   objetos acima da base (pontos | quadros com o objeto no centro):")
    for c in ordem[:6]:
        if c == 0 or tam[c] == 0:
            continue
        pts = acima[rot[tuple(idx.T)] == c]
        ext = np.ptp(pts, 0)
        log(f"     objeto {c}: {tam[c]:,} pts | {votos[c]} quadros | extensao {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
    if votos.max() > 0:
        melhor = votos.max()
        empatados = [c for c in range(1, ncomp + 1) if votos[c] == melhor]
        peca = max(empatados, key=lambda c: tam[c])
    else:
        peca = int(ordem[0] if ordem[0] else ordem[1])
    manter = ndimage.binary_dilation(rot == peca, structure=np.ones((3, 3, 3)), iterations=1)
    log(f"   peca escolhida: objeto {peca} ({votos[peca]} de {validos} quadros)")
    if validos and votos[peca] < 0.5 * validos:
        log("   AVISO: a peca escolhida esta no centro em menos da metade dos quadros; conferir a imagem")
    return dict(n=n, d=d, faixa_base=faixa_base, voxel=voxel, lo=lo, manter=manter,
                peca=peca, votos=votos.tolist(), tam=tam.tolist(), tem_base=base is not None)


def mascara_manter(P, seg):
    h = P @ seg["n"] + seg["d"]
    idx = np.floor((P - seg["lo"]) / seg["voxel"]).astype(int)
    dentro = np.all((idx >= 0) & (idx < np.array(seg["manter"].shape)), axis=1)
    ok = np.zeros(len(P), bool)
    ii = idx[dentro]
    ok[dentro] = seg["manter"][tuple(ii.T)]
    return ok & (h > seg["faixa_base"])


# ---------------------------------------------------------------- gravacao
def limpar_scan(scan, seg, destino_scan, log=print):
    """Grava a copia do scan com os pixels fora da peca apagados em cada quadro."""
    os.makedirs(destino_scan, exist_ok=True)
    for item in os.listdir(scan.pasta):
        src = os.path.join(scan.pasta, item)
        if item == "cache":
            continue
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(destino_scan, item), dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(destino_scan, item))
    db = scan.db + ".limpo"
    # o Revo Scan usa modo WAL: consolida antes de copiar
    scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
    shutil.copy(scan.db, db)
    cn = sqlite3.connect(db)
    tot_antes = tot_depois = 0
    vazios = []
    for k, i in enumerate(scan.ids):
        if not scan.tem_prof[k]:
            continue
        w, h, mask, z, quatro = scan.profundidade(i)
        P, pix = scan.pontos(k)
        ok = mascara_manter(P, seg)
        tot_antes += len(pix)
        if ok.sum() == 0:
            # quadro sem nenhum ponto da peca: fica como estava
            vazios.append(k)
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
    # consolida o journal dentro do arquivo principal
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
    gravar_ply(os.path.join(destino_scan, "raw_preview.ply"), hdr, prev[okp])
    log(f"   pontos dos quadros: {tot_antes:,} -> {tot_depois:,} ({tot_depois/max(1,tot_antes)*100:.0f}% mantidos) | "
        f"preview {len(prev):,} -> {okp.sum():,} | integridade: {integ}")
    if vazios:
        log(f"   AVISO: {len(vazios)} quadros sem nenhum ponto da peca foram mantidos como estavam")
    return dict(antes=tot_antes, depois=tot_depois, vazios=len(vazios), integridade=integ)


def gravar_revo(origem_revo, destino_proj, nome, nos):
    r = json.load(open(origem_revo, encoding="utf-8"))
    r["guid"] = str(uuid.uuid4())
    r["name"] = nome
    r["edit_time"] = str(int(time.time()))
    r["nodes"] = [n for n in r["nodes"] if n["guid"] in nos]
    # igual ao Revo Scan: soma so os arquivos da pasta data
    tam = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(os.path.join(destino_proj, "data")) for f in fs)
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
    n = seg["n"] if seg.get("tem_base", True) else np.array([0.0, 0.0, 1.0])
    d = seg["d"] if seg.get("tem_base", True) else 0.0
    u = np.cross(n, [1, 0, 0]); u /= np.linalg.norm(u); v = np.cross(n, u)
    L = np.c_[P @ u, P @ v, P @ n + d]
    fig, ax = plt.subplots(2, 2, figsize=(14, 11), dpi=90)
    for col, (titulo, m) in enumerate((("ANTES (todos os pontos)", np.ones(len(L), bool)),
                                       ("DEPOIS (so a peca)", ok))):
        ax[0, col].scatter(L[m, 0], L[m, 1], s=0.2, c=L[m, 2], cmap="viridis", linewidths=0)
        ax[0, col].set_aspect("equal"); ax[0, col].set_title(titulo + ": vista de cima")
        ax[1, col].scatter(L[m, 0], L[m, 2], s=0.2, c="#333", linewidths=0)
        ax[1, col].set_aspect("equal"); ax[1, col].set_title(titulo + ": vista lateral")
    for a in ax[:, 1]:
        a.set_xlim(ax[0, 0].get_xlim())
    ax[0, 1].set_ylim(ax[0, 0].get_ylim()); ax[1, 1].set_ylim(ax[1, 0].get_ylim())
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
