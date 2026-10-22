"""Remove da nuvem/malha componentes ISOLADAS e pequenas (restos de suporte flutuando longe da peca).

    python pipeline/limpar_fragmentos.py <nuvem_ref.ply> --aplicar arq1.ply arq2.ply ... [--voxel 3] [--ponte 9] [--max-pct 3]

Define os componentes na nuvem de referencia (voxels de `--voxel` mm, fresta de ate `--ponte` mm ainda liga);
remove os que tem menos de `--max-pct`% do maior. Aplica o mesmo corte (por posicao) a nuvens e malhas listadas.
"""
import argparse, os, sys
import numpy as np
import open3d as o3d
from scipy import ndimage
ap = argparse.ArgumentParser(); ap.add_argument("ref"); ap.add_argument("--aplicar", nargs="+", required=True)
ap.add_argument("--voxel", type=float, default=3.0); ap.add_argument("--ponte", type=float, default=9.0); ap.add_argument("--max-pct", type=float, default=3.0)
a = ap.parse_args()
P = np.asarray(o3d.io.read_point_cloud(a.ref).points); mn = P.min(0) - 20
k = np.floor((P - mn) / a.voxel).astype(int); G = np.zeros(k.max(0) + 4, bool); G[tuple((k + 1).T)] = True
it = max(1, int(round(a.ponte / a.voxel / 2)))
lab, n = ndimage.label(ndimage.binary_dilation(G, structure=np.ones((3, 3, 3)), iterations=it), structure=np.ones((3, 3, 3)))
tam = np.bincount(lab[tuple((k + 1).T)], minlength=n + 1); tam[0] = 0
principal = int(np.argmax(tam)); remover = {i for i in range(1, n + 1) if tam[i] > 0 and tam[i] < a.max_pct / 100 * tam[principal]}
print(f"componentes: {int((tam>0).sum())} | removidos: {len(remover)} com {int(sum(tam[i] for i in remover)):,} pontos de ref ({sum(tam[i] for i in remover)/tam.sum()*100:.2f}%)")
def ids_de(X):
    q = np.floor((X - mn) / a.voxel).astype(int) + 1
    ok = np.all((q >= 0) & (q < np.array(G.shape)), axis=1); r = np.zeros(len(X), int); r[ok] = lab[tuple(q[ok].T)]; return r
for f in a.aplicar:
    if f.endswith(".ply"):
        try:
            M = o3d.io.read_triangle_mesh(f)
            if len(M.triangles) > 0:
                r = ids_de(np.asarray(M.vertices)); m = np.isin(r, list(remover)); M.remove_vertices_by_mask(m); M.remove_unreferenced_vertices()
                o3d.io.write_triangle_mesh(f, M, write_vertex_normals=False); print(f"  malha {os.path.basename(f)}: -{int(m.sum()):,} vertices -> {len(M.triangles):,} triangulos"); continue
        except Exception: pass
        pc = o3d.io.read_point_cloud(f); r = ids_de(np.asarray(pc.points)); m = np.isin(r, list(remover)); pc = pc.select_by_index(np.flatnonzero(~m).tolist())
        o3d.io.write_point_cloud(f, pc, write_ascii=False); print(f"  nuvem {os.path.basename(f)}: -{int(m.sum()):,} pontos -> {len(pc.points):,}")
