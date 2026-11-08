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
Todo o calculo (voxels, vizinhos exatos --k <= 64, PCA, suavizacao) roda na GPU (gpu.py); na CPU ficam so a leitura dos
NPY e a gravacao do PLY (Open3D).
"""

import argparse
import json
import os
import time

import gpu
import numpy as np
import pca3_gpu
from alinhar_varios import baixa_prioridade


def reagrupar(P, c, V, voxel):
    """(centroides ponderados por c, soma dos pesos, visada media unitaria) de cada voxel de `voxel` mm. GPU; devolve cupy."""
    m = gpu.xp()
    P = gpu.para_gpu(P, m.float64)
    mn = P.min(0) - 1
    _, inv, _ = gpu.agrupar(gpu.chaves_voxel(P, voxel, mn))
    w = gpu.para_gpu(c, m.float64)
    soma_w = gpu.somar_por_grupo(w, inv)
    Q = gpu.somar_por_grupo(P * w[:, None], inv, len(soma_w)) / soma_w[:, None]
    W = gpu.somar_por_grupo(gpu.para_gpu(V, m.float64) * w[:, None], inv, len(soma_w))
    W /= m.maximum(m.linalg.norm(W, axis=1, keepdims=True), 1e-9)
    return Q.astype(m.float32), soma_w.astype(m.float32), W.astype(m.float32)


def suavizar_passada(P1, V1, raio, k, max_desloc, min_vizinhos):
    """Uma passada de projecao no plano local, na GPU. P1/V1 cupy float32. Devolve (saida, normal, ok, deslocamento)."""
    m = gpu.xp()
    n = len(P1)
    sig = raio / 2
    g = gpu.Grade(P1, raio)
    saida = m.empty_like(P1)
    normal = m.empty_like(P1)
    ok = m.zeros(n, bool)
    desl = m.zeros(n, m.float32)
    for a, b in gpu.blocos(n, bytes_por_item=k * 150):
        Pc = P1[a:b].astype(m.float64)
        d, idx = g.knn(Pc, k, gpu=True)
        val = idx >= 0
        idx = m.where(val, idx, 0)
        w = m.where(val, m.exp(-(m.minimum(d, raio) ** 2) / (2 * sig**2)), 0.0)
        Q = P1[idx].astype(m.float64)
        sw = m.maximum(w.sum(1), 1e-12)
        cen = (w[..., None] * Q).sum(1) / sw[:, None]
        D = Q - cen[:, None, :]
        cov = m.einsum("nk,nki,nkj->nij", w, D, D) / sw[:, None, None]
        _, nrm = pca3_gpu.menor_autopar(cov)
        # orienta pela visada
        nrm = m.where((m.einsum("ij,ij->i", nrm, V1[a:b].astype(m.float64)) < 0)[:, None], -nrm, nrm)
        dist = m.clip(m.einsum("ij,ij->i", Pc - cen, nrm), -max_desloc, max_desloc)
        saida[a:b] = (Pc - dist[:, None] * nrm).astype(m.float32)
        normal[a:b] = nrm.astype(m.float32)
        ok[a:b] = val.sum(1) >= min_vizinhos
        desl[a:b] = m.abs(dist)
    return saida, normal, ok, desl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pasta")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.6)
    ap.add_argument("--raio", type=float, default=1.8)
    ap.add_argument("--k", type=int, default=28, help="vizinhos (exatos, no maximo 64)")
    ap.add_argument("--max-desloc", type=float, default=0.5)
    ap.add_argument("--min-vizinhos", type=int, default=6)
    ap.add_argument("--passadas", type=int, default=1)
    a = ap.parse_args()
    if not 1 <= a.k <= 64:
        ap.error("--k deve estar entre 1 e 64 (kNN exato na GPU)")
    baixa_prioridade()
    m = gpu.xp()
    os.makedirs(a.saida, exist_ok=True)
    t0 = time.time()
    P = np.load(os.path.join(a.pasta, "nuvem.npy"))
    c = np.load(os.path.join(a.pasta, "contagem.npy"))
    V = np.load(os.path.join(a.pasta, "visada.npy"))
    n0 = len(P)
    P1, W1, V1 = reagrupar(P, c, V, a.voxel)
    print(f"{n0:,} pontos (0,4 mm) -> {len(P1):,} voxels de {a.voxel:g} mm ({time.time() - t0:.0f}s)", flush=True)
    del P, c, V
    for passada in range(a.passadas):
        n = len(P1)
        saida, normal, ok, desl = suavizar_passada(P1, V1, a.raio, a.k, a.max_desloc, a.min_vizinhos)
        print(
            f"  passada {passada + 1}: {int(ok.sum()):,} de {n:,} mantidos ({int((~ok).sum()):,} isolados) | deslocamento mediano {float(m.median(desl[ok])) * 1000:.0f} um, p90 {float(m.percentile(desl[ok], 90)) * 1000:.0f} um ({time.time() - t0:.0f}s)",
            flush=True,
        )
        P1, V1, W1 = saida[ok], V1[ok], W1[ok]
        N1 = normal[ok]
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(P1).astype(np.float64)))
    pc.normals = o3d.utility.Vector3dVector(gpu.para_cpu(N1).astype(np.float64))
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_refinada.ply"), pc, write_ascii=False)
    json.dump(
        dict(pontos=int(len(P1)), voxel=a.voxel, raio=a.raio, max_desloc=a.max_desloc, passadas=a.passadas),
        open(os.path.join(a.saida, "relatorio.json"), "w"),
    )
    print(f"nuvem refinada: {len(P1):,} pontos com normais | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
