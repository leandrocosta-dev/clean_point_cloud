"""Suaviza uma nuvem ao longo da normal, so entre pontos da MESMA face (normais concordantes).

    python pipeline/suavizar_nuvem.py <nuvem.ply (com normais)> --saida <nova.ply> [--raio 0.8] [--passadas 2] [--cos-min 0.8]

Cada ponto vai para a media (ponderada por gaussiana) da altura dos vizinhos ate --raio mm cuja normal
tem cos > --cos-min com a dele, medida ao longo da normal do ponto. Desloca so ao longo da normal (nao encolhe
bordas no plano). A outra face de uma chapa fina nao entra (normal oposta). Reduz o ruido de camada entre scans.
Calculo na GPU (gpu.py; --k vizinhos exatos, no maximo 64); na CPU ficam so a leitura e a gravacao do PLY (Open3D).
"""

import argparse

import gpu
import numpy as np
import open3d as o3d


def suavizar(P, N, raio=0.8, passadas=2, cos_min=0.8, k=32, min_vizinhos=5, log=None):
    """Passadas de suavizacao ao longo da normal. P, N (n, 3) numpy ou cupy; devolve P novo (numpy float64)."""
    m = gpu.xp()
    P = gpu.para_gpu(P, m.float64).copy()
    N = gpu.para_gpu(N, m.float64)
    sig = raio / 2
    for it in range(passadas):
        g = gpu.Grade(P, raio)
        novo = P.copy()
        desl = m.empty(len(P), m.float64)
        for a, b in gpu.blocos(len(P), bytes_por_item=k * 120):
            d, i = g.knn(P[a:b], k, gpu=True)
            ok = i >= 0
            i2 = m.where(ok, i, 0)
            ok &= m.einsum("nkc,nc->nk", N[i2], N[a:b]) > cos_min
            w = m.exp(-(m.minimum(d, raio) ** 2) / (2 * sig**2)) * ok
            h = m.einsum("nkc,nc->nk", P[i2] - P[a:b][:, None, :], N[a:b])
            dh = (w * h).sum(1) / m.maximum(w.sum(1), 1e-9)
            dh = m.where(ok.sum(1) < min_vizinhos, 0, dh)
            novo[a:b] = P[a:b] + N[a:b] * dh[:, None]
            desl[a:b] = m.abs(dh)
        P = novo
        if log:
            log(f"passada {it + 1}: deslocamento mediano {float(m.median(desl)) * 1000:.1f} um")
    return gpu.para_cpu(P)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nuvem")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--raio", type=float, default=0.8)
    ap.add_argument("--passadas", type=int, default=2)
    ap.add_argument("--cos-min", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    if not 1 <= a.k <= 64:
        ap.error("--k deve estar entre 1 e 64 (kNN exato na GPU)")
    gpu.xp()
    pc = o3d.io.read_point_cloud(a.nuvem)
    P = suavizar(
        np.asarray(pc.points),
        np.asarray(pc.normals),
        a.raio,
        a.passadas,
        a.cos_min,
        a.k,
        log=lambda t: print(t, flush=True),
    )
    pc.points = o3d.utility.Vector3dVector(P)
    o3d.io.write_point_cloud(a.saida, pc)
    print("gravado", a.saida)


if __name__ == "__main__":
    main()
