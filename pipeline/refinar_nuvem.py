"""Refina a nuvem da fusao propria: reamostra, suaviza, tira isolados e orienta as normais.

    python pipeline/refinar_nuvem.py <pasta_fusao_propria> --saida <pasta> [--voxel 0.6] [--raio 1.8]

1. Reagrupa os voxels de 0,4 mm em voxels maiores (centroide ponderado pela contagem de pontos; a media de mais
   pontos reduz o ruido de medicao).
2. Suavizacao por projecao no plano local: para cada ponto, plano ajustado (PCA ponderado por gaussiana) aos vizinhos
   dentro de `--raio` mm; o ponto e levado ao plano, com deslocamento limitado a `--max-desloc` mm (nao colapsa
   detalhes finos). Pontos com poucos vizinhos (isolados) sao descartados.
3. Normais = autovetor de menor autovalor, orientadas para o lado de onde a superficie foi VISTA (soma dos vetores
   ponto->camera de cada voxel): paredes interna e externa de uma casca fina ficam com sentidos opostos.
Saida: nuvem_refinada.ply (com normais) e relatorio.json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alinhar_varios import baixa_prioridade


def reagrupar(P, c, V, voxel):
    mn = P.min(0) - 1
    k = np.floor((P - mn) / voxel).astype(np.int64)
    d = k.max(0) + 1
    key = (k[:, 0] * d[1] + k[:, 1]) * d[2] + k[:, 2]
    u, inv = np.unique(key, return_inverse=True)
    w = c.astype(np.float64)
    soma_w = np.bincount(inv, weights=w)
    Q = np.stack([np.bincount(inv, weights=w * P[:, a]) for a in range(3)], 1) / soma_w[:, None]
    W = np.stack([np.bincount(inv, weights=w * V[:, a]) for a in range(3)], 1)
    W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    return Q.astype(np.float32), soma_w.astype(np.float32), W.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta"); ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.6)
    ap.add_argument("--raio", type=float, default=1.8)
    ap.add_argument("--k", type=int, default=28)
    ap.add_argument("--max-desloc", type=float, default=0.5)
    ap.add_argument("--min-vizinhos", type=int, default=6)
    ap.add_argument("--passadas", type=int, default=1)
    a = ap.parse_args()
    baixa_prioridade()
    os.makedirs(a.saida, exist_ok=True)
    t0 = time.time()
    P = np.load(os.path.join(a.pasta, "nuvem.npy")); c = np.load(os.path.join(a.pasta, "contagem.npy")); V = np.load(os.path.join(a.pasta, "visada.npy"))
    P1, W1, V1 = reagrupar(P, c, V, a.voxel)
    print(f"{len(P):,} pontos (0,4 mm) -> {len(P1):,} voxels de {a.voxel:g} mm ({time.time()-t0:.0f}s)", flush=True)
    del P, c, V
    sig = a.raio / 2
    for passada in range(a.passadas):
        arv = cKDTree(P1)
        n = len(P1)
        saida = np.empty_like(P1); normal = np.empty_like(P1); ok = np.zeros(n, bool); desl = np.zeros(n, np.float32)
        tam = 1_000_000
        for i0 in range(0, n, tam):
            Pc = P1[i0:i0 + tam].astype(np.float64)
            d, idx = arv.query(Pc, k=a.k, distance_upper_bound=a.raio, workers=4)
            val = idx < n
            idx = np.where(val, idx, 0)
            w = np.where(val, np.exp(-(d ** 2) / (2 * sig ** 2)), 0.0)
            Q = P1[idx].astype(np.float64)
            sw = np.maximum(w.sum(1), 1e-12)
            cen = (w[..., None] * Q).sum(1) / sw[:, None]
            D = Q - cen[:, None, :]
            cov = np.einsum("nk,nki,nkj->nij", w, D, D) / sw[:, None, None]
            ev, evec = np.linalg.eigh(cov)
            nrm = evec[:, :, 0]
            # orienta pela visada
            flip = np.einsum("ij,ij->i", nrm, V1[i0:i0 + tam]) < 0
            nrm[flip] *= -1
            dist = np.einsum("ij,ij->i", Pc - cen, nrm)
            dist = np.clip(dist, -a.max_desloc, a.max_desloc)
            saida[i0:i0 + tam] = (Pc - dist[:, None] * nrm).astype(np.float32)
            normal[i0:i0 + tam] = nrm.astype(np.float32)
            ok[i0:i0 + tam] = val.sum(1) >= a.min_vizinhos
            desl[i0:i0 + tam] = np.abs(dist)
        print(f"  passada {passada+1}: {int(ok.sum()):,} de {n:,} mantidos ({(~ok).sum():,} isolados) | deslocamento mediano {np.median(desl[ok])*1000:.0f} um, p90 {np.percentile(desl[ok],90)*1000:.0f} um ({time.time()-t0:.0f}s)", flush=True)
        P1, V1, W1 = saida[ok], V1[ok], W1[ok]
        N1 = normal[ok]
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P1.astype(np.float64)))
    pc.normals = o3d.utility.Vector3dVector(N1.astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_refinada.ply"), pc, write_ascii=False)
    json.dump(dict(pontos=int(len(P1)), voxel=a.voxel, raio=a.raio, max_desloc=a.max_desloc, passadas=a.passadas), open(os.path.join(a.saida, "relatorio.json"), "w"))
    print(f"nuvem refinada: {len(P1):,} pontos com normais | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
