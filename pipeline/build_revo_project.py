"""
Gera um projeto Revo Scan (.revo) completo e limpo a partir de um projeto
original, sem tocar no original.

Copia a pasta inteira (frames.dataset, params, poses...) e, em cada scan,
substitui fuse.ply, fuse_mesh.ply (quando existem) e raw_preview.ply por
versoes sem mesa/ruido, no mesmo layout binario do Revo Scan, e atualiza as
contagens no property.rvproj. O raw_preview.ply e o que o Revo mostra ao abrir
um scan, por isso tambem e limpo (inclusive em scans sem fusao).
"""

import argparse
import json
import os
import shutil

import numpy as np
import open3d as o3d
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def read_ply(path, has_faces):
    with open(path, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end]
    n = int(header.split(b"element vertex ")[1].split()[0])
    vert = np.frombuffer(raw, "<f4", n * 6, end).reshape(n, 6)
    faces = None
    if has_faces:
        nf = int(header.split(b"element face ")[1].split()[0])
        rec = np.dtype([("n", "u1"), ("i", "<u4", 3)])
        faces = np.frombuffer(raw, rec, nf, end + n * 24)["i"]
    return vert, faces


def write_ply(path, vert, faces=None):
    h = ["ply", "format binary_little_endian 1.0", "comment created by revopoint 3D",
         f"element vertex {len(vert)}"]
    h += [f"property float {p}" for p in ("x", "y", "z", "nx", "ny", "nz")]
    if faces is not None:
        h += [f"element face {len(faces)}", "property list uchar uint vertex_indices"]
    h.append("end_header")
    with open(path, "wb") as f:
        f.write(("\n".join(h) + "\n").encode())
        f.write(np.ascontiguousarray(vert, "<f4").tobytes())
        if faces is not None:
            rec = np.empty(len(faces), np.dtype([("n", "u1"), ("i", "<u4", 3)]))
            rec["n"] = 3
            rec["i"] = faces
            f.write(rec.tobytes())


def to_pcd(xyz):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return p


def part_indices(vert, plane_thresh, passes, eps_mult):
    """Indices (no array original) dos pontos da peca."""
    idx = np.arange(len(vert))
    pcd = to_pcd(vert[:, :3])
    nn = np.median(np.asarray(pcd.compute_nearest_neighbor_distance()))
    for i in range(passes):
        _, inl = pcd.segment_plane(plane_thresh, 3, 2000)
        if len(inl) < 0.25 * len(pcd.points):
            break
        print(f"    plano {i}: {len(inl)} pts removidos")
        keep = np.setdiff1d(np.arange(len(pcd.points)), inl)
        pcd, idx = pcd.select_by_index(keep), idx[keep]
    lab = np.array(pcd.cluster_dbscan(eps=nn * eps_mult, min_points=10))
    k = np.where(lab == np.argmax(np.bincount(lab[lab >= 0])))[0]
    pcd, idx = pcd.select_by_index(k), idx[k]
    _, k = pcd.remove_statistical_outlier(20, 2.0)
    return idx[np.asarray(k)], nn


def clean_mesh(mverts, faces, kept_xyz, radius):
    d = np.asarray(to_pcd(mverts[:, :3]).compute_point_cloud_distance(to_pcd(kept_xyz)))
    ok = d < radius
    f = faces[ok[faces].all(axis=1)]
    # maior componente conexa da malha
    n = len(mverts)
    g = coo_matrix((np.ones(len(f) * 2), (f[:, 0].tolist() + f[:, 1].tolist(),
                                          f[:, 1].tolist() + f[:, 2].tolist())), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    used = np.unique(f)
    big = np.argmax(np.bincount(lab[used]))
    f = f[lab[f[:, 0]] == big]
    used = np.unique(f)
    remap = np.full(n, -1, np.int64)
    remap[used] = np.arange(len(used))
    return mverts[used], remap[f].astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_dir", help="pasta do projeto (contem o .revo e data/)")
    ap.add_argument("out_dir", help="pasta nova do projeto limpo (nao pode existir)")
    ap.add_argument("--plane-thresh", type=float, default=1.5)
    ap.add_argument("--plane-passes", type=int, default=2)
    ap.add_argument("--eps-mult", type=float, default=4.0)
    ap.add_argument("--mesh-radius-mult", type=float, default=4.0)
    a = ap.parse_args()

    print("copiando projeto...")
    shutil.copytree(a.project_dir, a.out_dir)
    data = os.path.join(a.out_dir, "data")
    for scan in sorted(os.listdir(data)):
        sd = os.path.join(data, scan)
        rv = os.path.join(sd, "property.rvproj")
        fuse = os.path.join(sd, "fuse.ply")
        print(scan)
        raw = os.path.join(sd, "raw_preview.ply")
        if os.path.isfile(raw):
            rv_, _ = read_ply(raw, False)
            k_, _ = part_indices(rv_, a.plane_thresh, a.plane_passes, a.eps_mult)
            write_ply(raw, rv_[k_])
            print(f"  raw_preview.ply: {len(rv_)} -> {len(k_)}")
        if not os.path.isfile(fuse):
            continue
        vert, _ = read_ply(fuse, False)
        keep, nn = part_indices(vert, a.plane_thresh, a.plane_passes, a.eps_mult)
        cv = vert[keep]
        write_ply(fuse, cv)
        print(f"  fuse.ply: {len(vert)} -> {len(cv)}")
        prop = json.load(open(rv, encoding="utf-8"))
        prop["point_count"] = len(cv)
        mesh = os.path.join(sd, "fuse_mesh.ply")
        if os.path.isfile(mesh):
            mv, mf = read_ply(mesh, True)
            nv, nf = clean_mesh(mv, mf, cv[:, :3], nn * a.mesh_radius_mult)
            write_ply(mesh, nv, nf)
            print(f"  fuse_mesh.ply: {len(mv)} v / {len(mf)} f -> {len(nv)} v / {len(nf)} f")
            prop["vertex_count"], prop["face_count"] = len(nv), len(nf)
        json.dump(prop, open(rv, "w", encoding="utf-8"), indent=4)


if __name__ == "__main__":
    main()
