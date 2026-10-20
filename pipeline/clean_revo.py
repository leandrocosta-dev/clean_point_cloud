"""
Limpeza de scans Revo Scan (fuse.ply) sem CloudCompare, via Open3D.

Por scan: remove o plano da mesa (RANSAC, iterativo) -> DBSCAN mantem o maior
componente (a peca) -> SOR. Salva um PLY limpo por scan e um resumo.
"""

import argparse
import os

import numpy as np
import open3d as o3d


def clean(pcd, plane_thresh, plane_passes, eps_mult, sor_nn, sor_std):
    nn = np.median(np.asarray(pcd.compute_nearest_neighbor_distance()))
    for i in range(plane_passes):
        plane, inl = pcd.segment_plane(plane_thresh, 3, 2000)
        # so remove planos grandes (mesa), nao faces planas da peca
        if len(inl) < 0.25 * len(pcd.points):
            break
        print(f"  plano {i}: {len(inl)} pts removidos")
        pcd = pcd.select_by_index(inl, invert=True)
    lab = np.array(pcd.cluster_dbscan(eps=nn * eps_mult, min_points=10))
    keep = np.argmax(np.bincount(lab[lab >= 0]))
    pcd = pcd.select_by_index(np.where(lab == keep)[0])
    pcd, _ = pcd.remove_statistical_outlier(sor_nn, sor_std)
    return pcd


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
