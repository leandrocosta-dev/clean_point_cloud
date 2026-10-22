"""Fusao propria: nuvem COMPLETA a partir dos quadros limpos e alinhados, sem o filtro do Revo.

    python pipeline/fundir_proprio.py <projeto_limpo> <segmentos.json> --saida <pasta> [--voxel 0.4] [--minimo 3]

Por que: o Revo descarta ~11% dos pontos dos segmentos usados (estruturas inteiras, 98% delas confirmadas por outros
scans). Aqui cada pixel de cada quadro vira ponto 3D no referencial comum (pose do segmento do quadro) e cada voxel
de `--voxel` mm com pelo menos `--minimo` pontos vira o seu CENTROIDE (promedia o ruido de medicao). Mesmo
referencial dos segmentos (o do scan de referencia).
Saida: nuvem.npy (N x 3 float32), contagem.npy (N, uint16) e nuvem.ply.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan
from alinhar_varios import baixa_prioridade


def _agrega(chaves, soma, cnt):
    u, inv = np.unique(chaves, return_inverse=True)
    s = np.stack([np.bincount(inv, weights=soma[:, a]) for a in range(soma.shape[1])], 1)
    c = np.bincount(inv, weights=cnt)
    return u, s, c


def _fundir_scan(args):
    proj, g, segs, voxel, mn, destino = args
    baixa_prioridade()
    K = np.array(segs[0]["T"])
    res = None
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(os.path.join(proj, "data", g), tmp)
        try:
            for m in sorted(set(int(x) for x, p in zip(scan.modo, scan.tem_prof) if p)):
                scan.K[m], _ = scan.calibrar(m)
            lote = []
            vis = []
            for k in range(len(scan.ids)):
                if not scan.tem_prof[k]:
                    continue
                seg = next((s for s in segs if s["k0"] <= k <= s["k1"]), None)
                if seg is None:
                    continue
                T = np.array(seg["T"])
                P, _ = scan.pontos(k)
                Pg = P @ T[:3, :3].T + T[:3, 3]
                cam = scan.T[k][:3, 3] @ T[:3, :3].T + T[:3, 3]          # centro da camera no referencial comum
                v = cam - Pg
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                lote.append(np.hstack([Pg, v]))
                if len(lote) >= 12:
                    X = np.vstack(lote); lote = []
                    kk = np.floor((X[:, :3] - mn) / voxel).astype(np.int64)
                    key = (kk[:, 0] * 20011 + kk[:, 1]) * 20021 + kk[:, 2]
                    u, s, c = _agrega(key, X, np.ones(len(X)))
                    if res is None:
                        res = (u, s, c)
                    else:
                        res = _agrega(np.concatenate([res[0], u]), np.vstack([res[1], s]), np.concatenate([res[2], c]))
            if lote:
                X = np.vstack(lote)
                kk = np.floor((X[:, :3] - mn) / voxel).astype(np.int64)
                key = (kk[:, 0] * 20011 + kk[:, 1]) * 20021 + kk[:, 2]
                u, s, c = _agrega(key, X, np.ones(len(X)))
                res = (u, s, c) if res is None else _agrega(np.concatenate([res[0], u]), np.vstack([res[1], s]), np.concatenate([res[2], c]))
        finally:
            scan.c.close()
    np.savez(os.path.join(destino, g + ".npz"), u=res[0], s=res[1], c=res[2])
    return g, len(res[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto"); ap.add_argument("segmentos"); ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--minimo", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scans", nargs="*")
    a = ap.parse_args()
    baixa_prioridade()
    os.makedirs(a.saida, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "3"; os.environ.setdefault("REVO_THREADS", "3")
    proj = os.path.abspath(a.projeto)
    pj = json.load(open(a.segmentos))["segmentos"]
    scans = a.scans or sorted(pj)
    # caixa: cobre todos os scans (estimada pelos centroides dos segmentos nao e necessaria: usa limites largos)
    mn = np.array([-2500.0, -2500.0, -4000.0])
    t0 = time.time()
    parcial = os.path.join(a.saida, "parcial"); os.makedirs(parcial, exist_ok=True)
    tarefas = [(proj, g, sorted(pj[g], key=lambda s: s["seg"]), a.voxel, mn, parcial) for g in scans if not os.path.exists(os.path.join(parcial, g + ".npz"))]
    with ProcessPoolExecutor(a.jobs) as ex:
        for g, n in ex.map(_fundir_scan, tarefas):
            print(f"  {g}: {n:,} voxels ({time.time()-t0:.0f}s)", flush=True)
    # junta os scans
    res = None
    for g in scans:
        z = np.load(os.path.join(parcial, g + ".npz"))
        res = (z["u"], z["s"], z["c"]) if res is None else _agrega(np.concatenate([res[0], z["u"]]), np.vstack([res[1], z["s"]]), np.concatenate([res[2], z["c"]]))
        print(f"  somado {g}: {len(res[0]):,} voxels", flush=True)
    u, s, c = res
    bom = c >= a.minimo
    P = (s[bom, :3] / c[bom, None]).astype(np.float32)
    V = s[bom, 3:]
    V = (V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-9)).astype(np.float32)
    np.save(os.path.join(a.saida, "visada.npy"), V)
    n = c[bom].astype(np.uint16)
    np.save(os.path.join(a.saida, "nuvem.npy"), P); np.save(os.path.join(a.saida, "contagem.npy"), n)
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem.ply"), pc, write_ascii=False)
    print(f"voxels {len(u):,} -> {int(bom.sum()):,} com >= {a.minimo} pontos | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
