"""Formato do projeto do Revo Scan 6 (versao 1.3): leitura dos quadros e gravacao de copias.

Regras de seguranca:
- O projeto original nunca e alterado. Tudo e gravado numa copia.
- Nao depende do Open3D (so numpy e scipy).

Formato (confirmado em 5 projetos):
- cache/frames.dataset: SQLite com os 16 primeiros bytes trocados por b'RPAlgo_file_sys\\0'.
- FrameInfo.bin: cabecalho 16 bytes + pose 4x4 float64 (camera para mundo) + cauda.
  Modo do quadro = (hdr[1] >> 16) & 0xff (indice da pasta de calibracao param, param1...).
- FrameRGBD.data: largura, altura, n (uint32) + mascara de bits w*h (ordem little)
  + 4 bytes + n float32 (profundidade em mm, na ordem linha a linha dos pixels validos).
- Ordem temporal: (id >> 32, id & 0xffffffff). Numeracao do Revo Scan = indice entre
  os quadros que tem profundidade, nessa ordem.
- .revo: campo size = soma dos arquivos da pasta data. Copias recebem guid novo.
"""
import json
import os
import shutil
import sqlite3
import struct
import time
import uuid

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

SIG_REVO = b"RPAlgo_file_sys\x00"
SIG_SQL = b"SQLite format 3\x00"
K_PADRAO = (1140.4, 1141.0, 232.8, 319.5)   # modo usado nos scans de teste
ERRO_MAX_CALIB = 0.6                         # mm; acima disso a calibracao nao e confiavel


# ---------------------------------------------------------------- utilidades
def ler_ply(path):
    """PLY binario so com propriedades float32 (como o raw_preview do Revo). Retorna (cabecalho, matriz)."""
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


def gravar_ply_xyz(path, P):
    h = ("ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
         "property float x\nproperty float y\nproperty float z\nend_header\n")
    gravar_ply(path, h, np.asarray(P))


def retroprojetar(u, v, z, K):
    """Pixel (u, v) com profundidade z para o referencial da camera."""
    fx, fy, cx, cy = K
    return np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z]


def transformar(P, T):
    return P @ T[:3, :3].T + T[:3, 3]


def eixos_do_plano(n):
    """Base (u, v) do plano de normal n, usada nas vistas e nas coordenadas locais da mesa."""
    u = np.cross(n, [1, 0, 0])
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def voxel_media(P, voxel):
    """Reduz a nuvem a um ponto (a media) por voxel."""
    if len(P) == 0:
        return P
    idx = np.floor((P - P.min(0)) / voxel).astype(np.int64)
    chave = (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]
    _, inv, cnt = np.unique(chave, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    soma = np.c_[[np.bincount(inv, P[:, j], len(cnt)) for j in range(3)]].T
    return soma / cnt[:, None]


def arquivo_revo(proj):
    revos = [f for f in os.listdir(proj) if f.endswith(".revo")]
    if not revos:
        raise FileNotFoundError(f"nenhum arquivo .revo em {proj}")
    return os.path.join(proj, revos[0])


def scans_do_projeto(proj):
    return sorted(os.listdir(os.path.join(proj, "data")))


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
        self.bins = [r[1] for r in rows]
        self.hdr = np.array([struct.unpack("<IIIi", r[1][:16]) for r in rows])
        self.modo = (self.hdr[:, 1] >> 16) & 0xFF
        self.T = np.array([np.frombuffer(r[1][16:144], "<f8").reshape(4, 4) for r in rows])
        self.tem_prof = np.array([i in com_prof for i in self.ids])
        self.K = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.fechar()

    def fechar(self):
        try:
            self.c.close()
        except Exception:
            pass

    def numero_revo(self):
        """Numero que o Revo Scan mostra para cada quadro (-1 = quadro sem profundidade)."""
        return np.where(self.tem_prof, np.cumsum(self.tem_prof) - 1, -1)

    def propriedades(self):
        with open(os.path.join(self.pasta, "property.rvproj"), encoding="utf-8") as f:
            return json.load(f)

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
        P = retroprojetar(u, v, z.astype(float), K if K is not None else self.K[int(self.modo[k])])
        if mundo:
            P = transformar(P, self.T[k])
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
            e = []
            for u, v, z, T in dados:
                P = transformar(retroprojetar(u[::sub], v[::sub], z[::sub], p), T)
                e.append(np.minimum(arv.query(P)[0], 3.0))
            return float(np.mean(np.concatenate(e)))

        r = minimize(custo, x0, method="Nelder-Mead", options=dict(maxiter=400, xatol=0.05, fatol=1e-5))
        e = [arv.query(transformar(retroprojetar(u, v, z, r.x), T)[::3])[0] for u, v, z, T in dados]
        self.K[modo] = tuple(r.x)
        return tuple(r.x), float(np.median(np.concatenate(e)))


def calibrar_modos(scan, log=print):
    """Calibra todos os modos com profundidade. Retorna {modo: erro mediano em mm}."""
    erros = {}
    for m in sorted(set(int(m) for m, p in zip(scan.modo, scan.tem_prof) if p)):
        K, e = scan.calibrar(m)
        erros[m] = e
        log(f"   calibracao modo {m}: fx {K[0]:.1f} fy {K[1]:.1f} cx {K[2]:.1f} cy {K[3]:.1f} | erro {e:.3f} mm")
    return erros


def exigir_calibracao(scan, log=print):
    erros = calibrar_modos(scan, log)
    if max(erros.values()) > ERRO_MAX_CALIB:
        raise ValueError(f"{scan.nome}: calibracao com erro acima de {ERRO_MAX_CALIB} mm, pontos nao confiaveis")
    return erros


# ---------------------------------------------------------------- gravacao
def gravar_copia_scan(scan, destino_scan, alterar):
    """Copia a pasta do scan para destino_scan e grava o frames.dataset depois de alterar(conexao).

    alterar recebe a conexao com a copia do banco (o original nunca e aberto para escrita).
    Retorna o resultado do pragma integrity_check.
    """
    os.makedirs(destino_scan, exist_ok=True)
    for item in os.listdir(scan.pasta):
        src = os.path.join(scan.pasta, item)
        if item == "cache":
            continue
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(destino_scan, item), dirs_exist_ok=True)
        else:
            shutil.copy2(src, os.path.join(destino_scan, item))
    db = scan.db + ".copia"
    # o Revo Scan usa modo WAL: consolida antes de copiar
    scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
    shutil.copy(scan.db, db)
    cn = sqlite3.connect(db)
    alterar(cn)
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
    return integ


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


def preparar_destino(proj, sufixo, saida=None):
    """Pasta de saida da copia do projeto. Recusa sobrescrever. Retorna (pasta, nome do projeto novo)."""
    nome = os.path.splitext(os.path.basename(arquivo_revo(proj)))[0] + sufixo
    saida = os.path.abspath(saida or os.path.join(proj, "saida", nome))
    if os.path.exists(saida):
        raise FileExistsError(f"destino ja existe: {saida}")
    os.makedirs(os.path.join(saida, "data"))
    return saida, nome
