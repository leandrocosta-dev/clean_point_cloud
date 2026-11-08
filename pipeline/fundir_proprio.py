"""Fusao propria: nuvem COMPLETA a partir dos quadros limpos e alinhados, sem o filtro do Revo.

    python pipeline/fundir_proprio.py <projeto_limpo> <segmentos.json> --saida <pasta> [--voxel 0.4] [--minimo 3]

Por que: o Revo descarta ~11% dos pontos dos segmentos usados (estruturas inteiras, 98% delas confirmadas por outros
scans). Aqui cada pixel de cada quadro vira ponto 3D no referencial comum (pose do segmento do quadro) e cada voxel
de `--voxel` mm com pelo menos `--minimo` pontos vira o seu CENTROIDE (promedia o ruido de medicao). Mesmo
referencial dos segmentos (o do scan de referencia).
Saida: nuvem.npy (N x 3 float32), contagem.npy (N, uint16) e nuvem.ply.
Projecao dos pixels, chaves de voxel, somas e centroides na GPU (gpu.py; cada processo --jobs abre o seu contexto CUDA);
na CPU ficam o SQLite/decodificacao dos quadros, a calibracao e a gravacao de PLY/NPY.
"""

import argparse
import json
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import gpu
import numpy as np
from alinhar_varios import baixa_prioridade
from cache_observacoes import pontos_gpu
from revo_projeto import RevoScan


def _agrega(chaves, soma, cnt):
    """(chaves unicas, somas por coluna, contagens) por chave, na GPU."""
    gpu.xp()
    u, inv, _ = gpu.agrupar(gpu.para_gpu(chaves))
    return u, gpu.somar_por_grupo(soma, inv, len(u)), gpu.somar_por_grupo(cnt, inv, len(u))


def _chave(X, mn, voxel):
    kk = gpu.xp().floor((X[:, :3] - gpu.para_gpu(mn)) / voxel).astype(gpu.xp().int64)
    return (kk[:, 0] * 20011 + kk[:, 1]) * 20021 + kk[:, 2]


def _fundir_scan(args):
    proj, g, segs, voxel, mn, destino = args
    baixa_prioridade()
    m = gpu.xp()
    res = None
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(os.path.join(proj, "data", g), tmp)
        try:
            for m in sorted(set(int(x) for x, p in zip(scan.modo, scan.tem_prof) if p)):
                scan.K[m], _ = scan.calibrar(m)
            lote = []
            for k in range(len(scan.ids)):
                if not scan.tem_prof[k]:
                    continue
                seg = next((s for s in segs if s["k0"] <= k <= s["k1"]), None)
                if seg is None:
                    continue
                T = np.array(seg["T"])
                P = pontos_gpu(scan, k)[0]
                Rt, t = gpu.para_gpu(T[:3, :3].T.copy(), m.float64), gpu.para_gpu(T[:3, 3].copy(), m.float64)
                Pg = P @ Rt + t
                cam = gpu.para_gpu(scan.T[k][:3, 3], m.float64) @ Rt + t  # centro da camera no referencial comum
                v = cam - Pg
                v /= m.maximum(m.linalg.norm(v, axis=1, keepdims=True), 1e-9)
                lote.append(m.hstack([Pg, v]))
                if len(lote) >= 12:
                    X = m.vstack(lote)
                    lote = []
                    u, s, c = _agrega(_chave(X, mn, voxel), X, m.ones(len(X)))
                    if res is None:
                        res = (u, s, c)
                    else:
                        res = _agrega(m.concatenate([res[0], u]), m.vstack([res[1], s]), m.concatenate([res[2], c]))
            if lote:
                X = m.vstack(lote)
                u, s, c = _agrega(_chave(X, mn, voxel), X, m.ones(len(X)))
                res = (
                    (u, s, c)
                    if res is None
                    else _agrega(m.concatenate([res[0], u]), m.vstack([res[1], s]), m.concatenate([res[2], c]))
                )
        finally:
            scan.c.close()
    np.savez(os.path.join(destino, g + ".npz"), u=gpu.para_cpu(res[0]), s=gpu.para_cpu(res[1]), c=gpu.para_cpu(res[2]))
    return g, len(res[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("segmentos")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.4)
    ap.add_argument("--minimo", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--scans", nargs="*")
    a = ap.parse_args()
    baixa_prioridade()
    os.makedirs(a.saida, exist_ok=True)
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ.setdefault("REVO_THREADS", "3")
    proj = os.path.abspath(a.projeto)
    pj = json.load(open(a.segmentos))["segmentos"]
    scans = a.scans or sorted(pj)
    # caixa: cobre todos os scans (estimada pelos centroides dos segmentos nao e necessaria: usa limites largos)
    mn = np.array([-2500.0, -2500.0, -4000.0])
    t0 = time.time()
    parcial = os.path.join(a.saida, "parcial")
    os.makedirs(parcial, exist_ok=True)
    tarefas = [
        (proj, g, sorted(pj[g], key=lambda s: s["seg"]), a.voxel, mn, parcial)
        for g in scans
        if not os.path.exists(os.path.join(parcial, g + ".npz"))
    ]
    gpu.xp()  # falha cedo se nao houver CUDA
    with ProcessPoolExecutor(a.jobs, initializer=baixa_prioridade) as ex:
        for g, n in ex.map(_fundir_scan, tarefas):
            print(f"  {g}: {n:,} voxels ({time.time() - t0:.0f}s)", flush=True)
    # junta os scans
    m = gpu.xp()
    res = None
    for g in scans:
        z = np.load(os.path.join(parcial, g + ".npz"))
        zu, zs, zc = (gpu.para_gpu(z[k]) for k in ("u", "s", "c"))
        res = (
            (zu, zs, zc)
            if res is None
            else _agrega(m.concatenate([res[0], zu]), m.vstack([res[1], zs]), m.concatenate([res[2], zc]))
        )
        print(f"  somado {g}: {len(res[0]):,} voxels", flush=True)
    u, s, c = res
    bom = c >= a.minimo
    P = gpu.para_cpu((s[bom, :3] / c[bom, None]).astype(m.float32))
    V = s[bom, 3:]
    V = gpu.para_cpu((V / m.maximum(m.linalg.norm(V, axis=1, keepdims=True), 1e-9)).astype(m.float32))
    np.save(os.path.join(a.saida, "visada.npy"), V)
    n = gpu.para_cpu(c[bom]).astype(np.uint16)
    np.save(os.path.join(a.saida, "nuvem.npy"), P)
    np.save(os.path.join(a.saida, "contagem.npy"), n)
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem.ply"), pc, write_ascii=False)
    print(f"voxels {len(u):,} -> {int(bom.sum()):,} com >= {a.minimo} pontos | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
