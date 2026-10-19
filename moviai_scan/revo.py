"""Leitura dos quadros do Revo Scan para analise (versao de investigacao).

Para o uso normal, prefira moviai_scan/moviai_scan/revo_projeto.py.
"""
import os
import shutil
import sqlite3
import struct
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import minimize

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def read_ply(path):
    b = open(path, 'rb').read(); k = b.find(b'end_header\n') + 11; h = b[:k].decode()
    n = int([l for l in h.split('\n') if l.startswith('element vertex')][0].split()[-1])
    props = [l for l in h.split('\n') if l.startswith('property')]
    return np.frombuffer(b[k:k + n * 4 * len(props)], '<f4').reshape(n, len(props))[:, :3].astype(float)


def preparar_sqlite(name):
    """Cria <name>.sqlite na pasta atual: copia do frames.dataset com o cabecalho SQLite padrao."""
    dst = f'{name}.sqlite'
    src = os.path.join(DATA, name, 'cache', 'frames.dataset')
    if not (os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src)):
        shutil.copy(src, dst)
        with open(dst, 'r+b') as f:
            f.write(b'SQLite format 3\x00')
    return dst


class Scan:
    def __init__(s, name):
        s.name = name; s.c = sqlite3.connect(preparar_sqlite(name))
        s.rows = s.c.execute('select id,bin from FrameInfo order by id').fetchall()
        s.ids = np.array([r[0] for r in s.rows])
        s.hdr = np.array([struct.unpack('<IIIi', r[1][:16]) for r in s.rows])
        s.mode = (s.hdr[:, 1] >> 16) & 0xff
        s.T = np.array([np.frombuffer(r[1][16:144], '<f8').reshape(4, 4) for r in s.rows])
        s.order = np.lexsort((s.ids & 0xffffffff, s.ids >> 32))
        s.K = {}

    def depth(s, k):
        i = s.rows[k][0]; d = s.c.execute('select data from FrameRGBD where id=?', (i,)).fetchone()
        if d is None: return None
        d = d[0]; w, h, n = struct.unpack('<III', d[:12]); m = np.frombuffer(d[12:12 + w * h // 8], np.uint8)
        z = np.frombuffer(d[12 + w * h // 8 + 4:12 + w * h // 8 + 4 + 4 * n], '<f4').astype(float)
        v, u = np.nonzero(np.unpackbits(m, bitorder='little').astype(bool).reshape(h, w))
        return u.astype(float), v.astype(float), z

    def cam(s, k, K=None, step=1):
        r = s.depth(k)
        if r is None: return None
        u, v, z = [a[::step] for a in r]; fx, fy, cx, cy = K if K is not None else s.K[s.mode[k]]
        return np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z]

    def world(s, k, K=None, step=1):
        P = s.cam(k, K, step)
        if P is None: return None
        T = s.T[k]; return P @ T[:3, :3].T + T[:3, 3]

    def fit_K(s, mode=0, x0=(1140.4, 1141.0, 232.8, 319.5), nframes=6):
        prev = read_ply(f'{DATA}/{s.name}/raw_preview.ply'); tree = cKDTree(prev)
        ks = [k for k in s.order if s.mode[k] == mode]
        ks = [ks[j] for j in np.linspace(0, len(ks) - 1, nframes).astype(int)]
        dat = [(s.depth(k), s.T[k]) for k in ks]

        def sc(p):
            fx, fy, cx, cy = p; e = []
            for (u, v, z), T in dat:
                u, v, z = u[::15], v[::15], z[::15]
                P = np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z] @ T[:3, :3].T + T[:3, 3]
                d, _ = tree.query(P); e.append(np.minimum(d, 3.0))
            return np.mean(np.concatenate(e))
        r = minimize(sc, x0, method='Nelder-Mead', options=dict(maxiter=400, xatol=0.05, fatol=1e-5))
        s.K[mode] = tuple(r.x)
        e = []
        for (u, v, z), T in dat:
            fx, fy, cx, cy = r.x
            P = np.c_[(u - cx) * z / fx, -(v - cy) * z / fy, -z] @ T[:3, :3].T + T[:3, 3]
            e.append(tree.query(P[::3])[0])
        e = np.concatenate(e); return r.x, float(np.median(e)), float(np.percentile(e, 90)), len(prev)
