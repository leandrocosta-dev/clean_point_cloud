"""Reproduz por codigo a vista de um visualizador de malha: o que um observador de tras da peca ve.

    python pipeline/vista_malha.py <malha.ply> --saida vista.png [--res 1.0] [--eixo-pca 2] [--ref nuvem_para_pca.ply]
Raios ortograficos ao longo do eixo menor da peca (PCA) atingem a malha: azul = face de frente, amarelo = VERSO (o que o
MeshInspector pinta de amarelo e o 3Dconnexion de branco), branco = nada (furo de verdade). Imprime as fracoes.
"""

import argparse

import matplotlib
import numpy as np
import open3d as o3d

matplotlib.use("Agg")
import gpu
import matplotlib.pyplot as plt

# GPU: PCA da malha, projecao dos vertices, grade de raios e classificacao frente/verso. CPU: o Open3D RaycastingScene
# (intersecao raio-triangulo, so existe na CPU) e a imagem matplotlib.


def vista(M, res=1.0, c=None, V=None, lado=1):
    m = gpu.xp()
    Vt = gpu.para_gpu(np.asarray(M.vertices))
    if V is None:
        c = Vt.mean(0)
        w, E = np.linalg.eigh(gpu.para_cpu(m.cov((Vt[::7] - c).T)))
        V = E[:, ::-1]
    else:
        c = gpu.para_gpu(c)
    V = gpu.para_gpu(V)
    Q = (Vt - c) @ V
    xs = np.arange(float(Q[:, 0].min()), float(Q[:, 0].max()), res)
    ys = np.arange(float(Q[:, 1].min()), float(Q[:, 1].max()), res)
    gx, gy = np.meshgrid(xs, ys)
    z0 = float(Q[:, 2].max()) + 50 if lado > 0 else float(Q[:, 2].min()) - 50
    o = m.stack([gpu.para_gpu(gx.ravel()), gpu.para_gpu(gy.ravel()), m.full(gx.size, z0)], 1) @ V.T + c
    d = m.tile(V[:, 2] * (-lado), (len(o), 1))
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(M))  # CPU
    r = sc.cast_rays(o3d.core.Tensor(gpu.para_cpu(m.hstack([o, d])).astype(np.float32)))
    hit = gpu.para_gpu(np.isfinite(r["t_hit"].numpy()))
    n = gpu.para_gpu(r["primitive_normals"].numpy())
    front = hit & (m.einsum("ij,ij->i", n, d) < 0)
    hit, front = gpu.para_cpu(hit), gpu.para_cpu(front)
    img = np.zeros(gx.shape + (3,))
    h = hit.reshape(gx.shape)
    f = front.reshape(gx.shape)
    img[:] = 1
    img[h & f] = (0.1, 0.2, 0.9)
    img[h & ~f] = (0.85, 0.85, 0.1)
    return img, hit, front, (xs, ys)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("malha")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--res", type=float, default=1.0)
    ap.add_argument("--lado", type=int, default=1)
    a = ap.parse_args()
    M = o3d.io.read_triangle_mesh(a.malha)
    M.compute_triangle_normals()
    img, hit, front, (xs, ys) = vista(M, a.res, lado=a.lado)
    print(
        f"cobertura da vista: {hit.mean() * 100:.1f}% da caixa | dos pixels com malha: frente {front.sum() / max(hit.sum(), 1) * 100:.2f}%, verso {(hit & ~front).sum() / max(hit.sum(), 1) * 100:.2f}%"
    )
    plt.figure(figsize=(18, 7), dpi=70)
    plt.imshow(img, origin="lower")
    plt.axis("off")
    plt.savefig(a.saida, bbox_inches="tight")
