"""Orienta para FORA as normais da nuvem nos pontos visiveis de fora da peca (corrige trechos "de verso" na malha).

    python pipeline/orientar_normais.py <nuvem.ply (com normais)> --saida <nuvem_orientada.ply> [--celula 1.0] [--tol 3.0] [--cone 65]

As normais da fusao apontam para o lado de onde a camera viu a superficie: onde so um scan visto POR DENTRO mediu a parede,
o Poisson gera uma folha de verso (amarelo no MeshInspector, branco no 3Dconnexion). Aqui, de varias direcoes de fora (cone ao
redor do eixo menor da peca, PCA) um z-buffer ortografico marca os pontos a ate --tol mm do mais alto da celula: esses
sao visiveis de fora e tem a normal virada para o observador. O resto fica como estava.
Os z-buffers rodam na GPU (gpu.py); na CPU ficam so a leitura e a gravacao do PLY (Open3D).
"""

import argparse

import gpu
import numpy as np
import open3d as o3d


def _maximo_por_grupo(inv, valores, n):
    """Maximo de `valores` por grupo `inv` (GPU)."""
    m = gpu.xp()
    mx = m.full(n, -1e9, valores.dtype)
    m.maximum.at(mx, inv, valores)
    return mx


def direcoes(cone):
    """Direcoes de observacao: eixo menor e aneis de 8 azimutes a cada 20 graus ate `cone` graus."""
    dirs = [np.array([0, 0, 1.0])]
    for ang in np.radians(np.arange(25, cone + 1, 20)):
        for az in np.radians(np.arange(0, 360, 45)):
            dirs.append(np.array([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az), np.cos(ang)]))
    return dirs


def orientar(P, N, celula=1.0, tol=0.8, cone=65.0, log=None):
    """(normais orientadas (n, 3) float64 numpy, visto (n,) bool, virou (n,) bool). Mesma regra do script original:
    um ponto visivel de alguma direcao com normal para dentro so e virado se NENHUMA direcao que o ve a aprova."""
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    Ng = gpu.para_gpu(N, m.float64)
    amostra = Pg[::50]
    c = amostra.mean(0)
    cov = gpu.para_cpu(m.cov((amostra - c).T))  # 3x3: so o autovetor fica na CPU
    _, E = np.linalg.eigh(cov)
    V = gpu.para_gpu(E[:, ::-1].copy(), m.float64)
    Q = ((Pg - c) @ V).astype(m.float32)
    Nq = (Ng @ V).astype(m.float32)
    del Pg
    dirs = direcoes(cone)
    visto = m.zeros(len(Q), bool)
    bom = m.zeros(len(Q), bool)
    for k, dr in enumerate(dirs):
        u = np.cross(dr, [0, 0, 1.0] if abs(dr[2]) < 0.99 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(dr, u)
        drg, ug, vg = (gpu.para_gpu(x, m.float32) for x in (dr, u, v))
        s = Q @ drg
        gx = m.floor((Q @ ug) / celula).astype(m.int64)
        gy = m.floor((Q @ vg) / celula).astype(m.int64)
        key = (gx - gx.min()) * 100003 + (gy - gy.min())
        _, inv, cnt = gpu.agrupar(key)
        mx = _maximo_por_grupo(inv, s, len(cnt))
        vis = s >= mx[inv] - tol
        face = Nq @ drg
        ruim = vis & (face < 0)  # visivel de fora, normal apontando para dentro
        visto |= vis
        bom |= vis & (face > 0)
        if log:
            log(
                f"  direcao {k + 1}/{len(dirs)}: visiveis {float(visto.mean()) * 100:.1f}% | normais p/ dentro nesta direcao {float(ruim.mean()) * 100:.2f}%"
            )
    flip = visto & ~bom
    Nq = m.where(flip[:, None], -Nq, Nq)
    saida = gpu.para_cpu(Nq.astype(m.float64) @ V.T)
    return saida, gpu.para_cpu(visto), gpu.para_cpu(flip)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nuvem")
    ap.add_argument("--saida", required=True)
    ap.add_argument(
        "--so-visiveis",
        action="store_true",
        help="grava so os pontos visiveis de fora (pele externa), descartando a parede interna",
    )
    ap.add_argument("--celula", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.8)
    ap.add_argument("--cone", type=float, default=65.0)
    a = ap.parse_args()
    gpu.xp()
    pc = o3d.io.read_point_cloud(a.nuvem)
    P = np.asarray(pc.points)
    N = np.asarray(pc.normals)
    Nn, visto, flip = orientar(P, N, a.celula, a.tol, a.cone, log=lambda t: print(t, flush=True))
    print(
        f"visiveis de fora: {visto.mean() * 100:.1f}% | normais viradas p/ fora: {flip.sum():,} ({flip.mean() * 100:.2f}%)"
    )
    pc.normals = o3d.utility.Vector3dVector(Nn)
    if a.so_visiveis:
        pc = pc.select_by_index(np.flatnonzero(visto).tolist())
        print(f"so pele externa: {len(pc.points):,} pontos")
    o3d.io.write_point_cloud(a.saida, pc, write_ascii=False)


if __name__ == "__main__":
    main()
