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

import geom_gpu as gg
import gpu
import numpy as np


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
    h = ["ply", "format binary_little_endian 1.0", "comment created by revopoint 3D", f"element vertex {len(vert)}"]
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


def part_indices(vert, plane_thresh, passes, eps_mult):
    """Indices (no array original) dos pontos da peca; mesa, DBSCAN e filtro estatistico na GPU."""
    xp = gpu.xp()
    P = gpu.para_gpu(vert[:, :3], xp.float64)
    idx = xp.arange(len(P))
    nn = float(xp.median(gpu.knn(P, P, 2)[0][:, 1]))  # espacamento mediano
    for i in range(passes):
        _, inl = gg.ransac_plano_open3d(P, plane_thresh, 2000)
        if len(inl) < 0.25 * len(P):
            break
        print(f"    plano {i}: {len(inl)} pts removidos")
        manter = xp.ones(len(P), bool)
        manter[gpu.para_gpu(inl)] = False
        P, idx = P[manter], idx[manter]
    k = gpu.para_gpu(gg.dbscan_maior(P, nn * eps_mult, 10))
    P, idx = P[k], idx[k]
    k = gpu.remover_soltos(P, k=19, desvios=2.0)  # SOR(20 vizinhos, 2 desvios)
    return gpu.para_cpu(idx[k]), nn


def clean_mesh(mverts, faces, kept_xyz, radius):
    """Mantem triangulos com os 3 vertices a < radius da nuvem mantida e so a maior componente (vertices ligados)."""
    xp = gpu.xp()
    d = gg.distancia_ate(kept_xyz, mverts[:, :3], saida_gpu=True)
    ok = d < radius
    fg = gpu.para_gpu(faces).astype(xp.int64)
    f = fg[ok[fg].all(axis=1)]
    # maior componente conexa da malha (grafo de vertices; arestas f0-f1 e f1-f2)
    n = len(mverts)
    lab = gg.rotulos_por_arestas(n, xp.concatenate([f[:, 0], f[:, 1]]), xp.concatenate([f[:, 1], f[:, 2]]))
    used = xp.unique(f)
    big = int(xp.argmax(xp.bincount(lab[used], minlength=n)))
    f = f[lab[f[:, 0]] == big]
    used = xp.unique(f)
    remap = xp.full(n, -1, xp.int64)
    remap[used] = xp.arange(len(used))
    return mverts[gpu.para_cpu(used)], gpu.para_cpu(remap[f]).astype(np.uint32)


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
