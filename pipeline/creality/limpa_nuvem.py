import geom_gpu as gg
import gpu
import numpy as np
import open3d as o3d

R = r"C:\Users\pc\Downloads\Teste Gatinho\sGtXXt9_DOkFjr_xK2vpu_o0vNul"


def carregar(nome):
    p = o3d.io.read_point_cloud(R + "\\" + nome)
    p.points = o3d.utility.Vector3dVector(
        gpu.para_cpu(gpu.para_gpu(np.asarray(p.points), np.float64) * 1000.0)
    )  # m -> mm (GPU)
    return p


def objeto(p, ref_lado=None, dist=1.0):
    P = np.asarray(p.points)
    C, _ = gpu.reduzir_voxel(P, 0.5)
    pl, _ = gg.ransac_plano_open3d(
        C, 0.4, 3000
    )  # plano da mesa por RANSAC na GPU (n unitaria, orientada para o lado dos objetos)
    n = pl[:3]
    d = float(pl[3])
    s = gpu.para_cpu(gpu.para_gpu(P, np.float64) @ gpu.para_gpu(n) + d)
    if ref_lado is None:
        ref_lado = -1 if np.median(s) > 0 else 1  # decide below
    return n, d, s


def maior_componente(o, eps=0.8, min_pts=8):
    """Nuvem Open3D so com o maior cluster DBSCAN de `o` (GPU)."""
    return o.select_by_index(np.where(gg.dbscan_maior(np.asarray(o.points), eps, min_pts))[0])


def ruido(p, idx=None, k=40, raio=1.0, N=20000, seed=0):
    P = np.asarray(p.points)
    rng = np.random.default_rng(seed)
    W, C = gg.espalhamento_local(P, P[rng.choice(len(P), N, replace=False)], raio)  # covariancia local na GPU
    ok = C >= 15
    out = np.sqrt(np.maximum(W[ok, 0], 0))
    return np.median(out), np.mean(out)


if __name__ == "__main__":
    for nome in ("pc_before.ply", "pc_after.ply"):
        p = carregar(nome)
        n, d, s = objeto(p)
        # lado do gato = o lado que contem a maior massa de pontos fora do plano; usar Fused como referencia
        F = o3d.io.read_triangle_mesh(R + r"\result\Fused.ply")
        sf = gpu.para_cpu(gpu.para_gpu(np.asarray(F.vertices), np.float64) @ gpu.para_gpu(n) + d)
        lado = np.sign(np.median(sf))
        m = (s * lado) > 1.0
        o = p.select_by_index(np.where(m)[0])
        O = maior_componente(o)
        print(nome, len(O.points), "ruido mediana/media (mm)", ruido(O))
