"""
Limpeza de scans Revo Scan (fuse.ply) sem CloudCompare, via Open3D.

Por scan: remove o plano da mesa (RANSAC, iterativo) -> DBSCAN mantem o maior
componente (a peca) -> SOR. Salva um PLY limpo por scan e um resumo.
"""

import argparse
import os

import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d


def clean(pcd, plane_thresh, plane_passes, eps_mult, sor_nn, sor_std):
    """Mesa (RANSAC iterativo), maior componente DBSCAN e filtro estatistico: tudo na GPU (geom_gpu/gpu)."""
    xp = gpu.xp()
    P = gpu.para_gpu(np.asarray(pcd.points), xp.float64)
    idx = xp.arange(len(P))
    nn = float(xp.median(gpu.knn(P, P, 2)[0][:, 1]))  # espacamento mediano
    for i in range(plane_passes):
        plane, inl = gg.ransac_plano_open3d(P, plane_thresh, 2000)
        # so remove planos grandes (mesa), nao faces planas da peca
        if len(inl) < 0.25 * len(P):
            break
        print(f"  plano {i}: {len(inl)} pts removidos")
        manter = xp.ones(len(P), bool)
        manter[gpu.para_gpu(inl)] = False
        P, idx = P[manter], idx[manter]
    k = gpu.para_gpu(gg.dbscan_maior(P, nn * eps_mult, 10))
    P, idx = P[k], idx[k]
    k = gpu.remover_soltos(P, k=sor_nn - 1, desvios=sor_std)  # SOR do Open3D: media dos sor_nn vizinhos (com o proprio)
    return pcd.select_by_index(gpu.para_cpu(idx[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plane-thresh", type=float, default=1.5)
    ap.add_argument("--plane-passes", type=int, default=2)
    ap.add_argument("--eps-mult", type=float, default=4.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for s in a.scans:
        name = os.path.basename(os.path.dirname(os.path.abspath(s)))
        print(name)
        p = o3d.io.read_point_cloud(s)
        print(f"  entrada {len(p.points)} pts")
        c = clean(p, a.plane_thresh, a.plane_passes, a.eps_mult, 20, 2.0)
        print(f"  saida {len(c.points)} pts, extent {c.get_axis_aligned_bounding_box().get_extent()}")
        o3d.io.write_point_cloud(os.path.join(a.out, f"{name}_clean.ply"), c)


if __name__ == "__main__":
    main()
