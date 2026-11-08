"""Une a nuvem fundida do Revo (superficie nitida) com a fusao propria refinada (completa, mais ruidosa).

    python pipeline/uniao_revo_propria.py <fundida_revo.ply> <nuvem_refinada.ply> --saida <pasta> [--dist 1.2] [--min-componente 3000]

Regra: onde o Revo tem superficie, fica a do Revo. Dos pontos da fusao propria, entram so os que ficam a mais de
`--dist` mm de qualquer ponto do Revo (regioes que o Revo descartou por baixa confianca), sem os pedacos pequenos
(componentes de voxels de 2 mm com menos de `--min-componente` pontos). O referencial e o do Revo (ICP rigido leve).
Cada ponto e colorido pela fonte: cinza = Revo, laranja = acrescentado.
ICP (ponto-plano, Tukey 0,5 mm), distancias e componentes na GPU (gpu.py, geom_gpu.py); na CPU ficam so a leitura e a
gravacao dos PLY (Open3D).
"""

import argparse
import json
import os
import time

import geom_gpu
import gpu
import numpy as np
import open3d as o3d
from recursos import baixa_prioridade


def icp_leve(propria, revo, dist=4.0, voxel_revo=1.0, voxel_propria=1.5, iteracoes=30):
    """ICP rigido leve propria -> Revo na GPU: T 4x4 (numpy). Mesmos parametros do caminho antigo (Open3D)."""
    m = gpu.xp()
    rr, _ = gpu.reduzir_voxel(gpu.para_gpu(revo, m.float64), voxel_revo)
    ss, _ = gpu.reduzir_voxel(gpu.para_gpu(propria, m.float64), voxel_propria)
    nr = geom_gpu.normais_hibrido(rr, 3.0, max_nn=30, saida_gpu=True)
    ok = gpu.xp().isfinite(nr).all(1)
    T, _ = gpu.icp_ponto_plano(ss, rr[ok], nr[ok], np.eye(4), dist_max=dist, iteracoes=iteracoes, perda="tukey", c=0.5)
    return T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("revo")
    ap.add_argument("propria")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--dist", type=float, default=1.2)
    ap.add_argument("--min-componente", type=int, default=3000)
    a = ap.parse_args()
    baixa_prioridade()
    m = gpu.xp()
    os.makedirs(a.saida, exist_ok=True)
    t0 = time.time()
    R = o3d.io.read_point_cloud(a.revo)
    PR = np.asarray(R.points)
    NR = np.asarray(R.normals)
    S = o3d.io.read_point_cloud(a.propria)
    PS = np.asarray(S.points)
    NS = np.asarray(S.normals)
    print(f"Revo {len(PR):,} pontos | propria refinada {len(PS):,}", flush=True)
    # ICP rigido leve: propria -> Revo
    T = icp_leve(PS, PR)
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"referencial: propria -> Revo desloca {np.round(T[:3, 3], 2)} mm, {ang:.3f} graus", flush=True)
    PS = geom_gpu.transformar(PS, T, saida_gpu=True)
    NS = gpu.para_gpu(NS, m.float64) @ gpu.para_gpu(T[:3, :3].T.copy())
    d, _ = gpu.vizinho_mais_proximo(PR, PS)
    nova = d > a.dist
    print(
        f"da propria, {int(nova.sum()):,} pontos ({float(nova.mean()) * 100:.0f}%) estao a > {a.dist} mm do Revo",
        flush=True,
    )
    X = PS[nova]
    Nn = NS[nova]
    # componentes conexas (voxels de 2 mm): tira pedacos pequenos
    rot, tam = gpu.componentes_conexos_voxel(X, 2.0, 26)
    bom = tam[rot] >= a.min_componente
    print(
        f"componentes: {len(tam):,}; removidos {int((~bom).sum()):,} pontos de pedacos < {a.min_componente} pontos; ficam {int(bom.sum()):,}",
        flush=True,
    )
    X, Nn = gpu.para_cpu(X[bom]), gpu.para_cpu(Nn[bom])
    P = np.vstack([PR, X])
    N = np.vstack([NR, Nn])
    cor = np.vstack([np.tile([0.78, 0.78, 0.78], (len(PR), 1)), np.tile([1.0, 0.45, 0.0], (len(X), 1))])
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    pc.normals = o3d.utility.Vector3dVector(N)
    pc.colors = o3d.utility.Vector3dVector(cor)
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_completa.ply"), pc, write_ascii=False)
    np.save(os.path.join(a.saida, "fonte.npy"), np.r_[np.zeros(len(PR), np.uint8), np.ones(len(X), np.uint8)])
    json.dump(
        dict(
            revo=int(len(PR)), acrescentados=int(len(X)), dist_mm=a.dist, ICP_mm=T[:3, 3].tolist(), ICP_graus=float(ang)
        ),
        open(os.path.join(a.saida, "relatorio.json"), "w"),
    )
    print(f"nuvem completa: {len(P):,} pontos (Revo {len(PR):,} + acrescentados {len(X):,}) | {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
