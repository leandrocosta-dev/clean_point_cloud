"""Remove da nuvem/malha componentes ISOLADAS e pequenas (restos de suporte flutuando longe da peca).

    python pipeline/limpar_fragmentos.py <nuvem_ref.ply> --aplicar arq1.ply arq2.ply ... [--voxel 3] [--ponte 9] [--max-pct 3]

Define os componentes na nuvem de referencia (voxels de `--voxel` mm, fresta de ate `--ponte` mm ainda liga);
remove os que tem menos de `--max-pct`% do maior. Aplica o mesmo corte (por posicao) a nuvens e malhas listadas.
Grade 3D, dilatacao, rotulagem e consulta por posicao na GPU (cupyx.scipy.ndimage); na CPU ficam so a leitura/gravacao
dos PLY e a remocao de vertices da malha (Open3D, sem equivalente na GPU).
"""

import argparse
import os

import geom_gpu
import gpu
import numpy as np
import open3d as o3d


def componentes_grade(P, voxel, ponte, pad=20.0):
    """(origem, rotulos da grade (cupy int32), tamanhos por rotulo (cupy), rotulo de cada ponto de P, forma da grade).
    Voxels de `voxel` mm ocupados por P, dilatados para fechar frestas de ate `ponte` mm, rotulados (26-conectividade)."""
    m = gpu.xp()
    ndi = geom_gpu.ndi()
    P = gpu.para_gpu(P, m.float64)
    mn = P.min(0) - pad
    k = m.floor((P - mn) / voxel).astype(m.int64)
    G = m.zeros(tuple(int(x) for x in (k.max(0) + 4)), bool)
    G[k[:, 0] + 1, k[:, 1] + 1, k[:, 2] + 1] = True
    it = max(1, int(round(ponte / voxel / 2)))
    est = m.ones((3, 3, 3), bool)
    lab, n = ndi.label(ndi.binary_dilation(G, structure=est, iterations=it, brute_force=True), structure=est)
    tam = m.bincount(lab[k[:, 0] + 1, k[:, 1] + 1, k[:, 2] + 1].astype(m.int64), minlength=int(n) + 1)
    tam[0] = 0
    return mn, lab, tam, lab[k[:, 0] + 1, k[:, 1] + 1, k[:, 2] + 1], G.shape


def ids_de(X, mn, lab, voxel, forma):
    """Rotulo (0 fora da grade ou fora de qualquer componente) da celula de cada posicao de X."""
    m = gpu.xp()
    X = gpu.para_gpu(X, m.float64)
    q = m.floor((X - mn) / voxel).astype(m.int64) + 1
    ok = m.all((q >= 0) & (q < gpu.para_gpu(np.array(forma))), axis=1)
    r = m.zeros(len(X), m.int64)
    qq = q[ok]
    r[ok] = lab[qq[:, 0], qq[:, 1], qq[:, 2]]
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--aplicar", nargs="+", required=True)
    ap.add_argument("--voxel", type=float, default=3.0)
    ap.add_argument("--ponte", type=float, default=9.0)
    ap.add_argument("--max-pct", type=float, default=3.0)
    a = ap.parse_args()
    m = gpu.xp()
    P = np.asarray(o3d.io.read_point_cloud(a.ref).points)
    mn, lab, tam, _, forma = componentes_grade(P, a.voxel, a.ponte)
    principal = int(tam.argmax())
    lim = a.max_pct / 100 * float(tam[principal])
    remover = m.flatnonzero((tam > 0) & (tam < lim))
    remover = remover[remover > 0]
    removidos = int(tam[remover].sum())
    total = int(tam.sum())
    print(
        f"componentes: {int((tam > 0).sum())} | removidos: {len(remover)} com {removidos:,} pontos de ref ({removidos / total * 100:.2f}%)"
    )

    def marcar(X):
        return gpu.para_cpu(m.isin(ids_de(X, mn, lab, a.voxel, forma), remover))

    for f in a.aplicar:
        if f.endswith(".ply"):
            try:
                M = o3d.io.read_triangle_mesh(f)
                if len(M.triangles) > 0:
                    msk = marcar(np.asarray(M.vertices))
                    M.remove_vertices_by_mask(msk)
                    M.remove_unreferenced_vertices()
                    o3d.io.write_triangle_mesh(f, M, write_vertex_normals=False)
                    print(
                        f"  malha {os.path.basename(f)}: -{int(msk.sum()):,} vertices -> {len(M.triangles):,} triangulos"
                    )
                    continue
            except Exception:
                pass
            pc = o3d.io.read_point_cloud(f)
            msk = marcar(np.asarray(pc.points))
            pc = pc.select_by_index(np.flatnonzero(~msk).tolist())
            o3d.io.write_point_cloud(f, pc, write_ascii=False)
            print(f"  nuvem {os.path.basename(f)}: -{int(msk.sum()):,} pontos -> {len(pc.points):,}")


if __name__ == "__main__":
    main()
