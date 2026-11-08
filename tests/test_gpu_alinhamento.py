"""Grupo 2 (alinhamento) na GPU: registro_gpu, alinhar_direto, consenso_visibilidade, aprender e scripts de cobertura.

Cada teste compara a GPU com uma referencia numpy/scipy escrita aqui (o que o codigo antigo calculava).
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from tests.gpu_utils import CUDA_DISPONIVEL

PIPE = Path(__file__).resolve().parents[1] / "pipeline"
import alinhar_direto as AD
import alinhar_segmentos as AS
import gpu
import registro_gpu as RG


def ondulada(passo=0.8, lado=100.0, ruido=0.05, semente=1):
    rng = np.random.default_rng(semente)
    g = np.arange(0, lado, passo)
    x, y = np.meshgrid(g, g)
    z = 3 * np.sin(x / 9.0) * np.cos(y / 11.0)
    return np.c_[x.ravel(), y.ravel(), z.ravel()] + rng.normal(0, ruido, (x.size, 3))


def pose(graus, t):
    a = np.radians(graus)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class RegistroGpuTest(unittest.TestCase):
    def test_voxel_igual_ao_centroide_numpy(self):
        P = ondulada()
        n = RG.nuvem_voxel(P, 1.0, normais=False)
        o = P.min(0) - 0.5
        k = np.floor((P - o) / 1.0).astype(np.int64)
        u, inv = np.unique(k, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        ref = np.stack([np.bincount(inv, P[:, c]) for c in range(3)], 1) / np.bincount(inv)[:, None]
        self.assertEqual(len(n), len(ref))
        d, _ = cKDTree(ref).query(gpu.para_cpu(n.pts))
        self.assertLess(d.max(), 1e-9)

    def test_normais_de_plano_e_orientacao_para_o_centroide(self):
        g = np.arange(0, 30, 0.7)
        x, y = np.meshgrid(g, g)
        P = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        n = RG.nuvem_voxel(P, 1.0)
        N = gpu.para_cpu(n.normais)
        np.testing.assert_allclose(np.abs(N[:, 2]), 1.0, atol=1e-9)
        self.assertEqual(len(set(np.sign(N[:, 2]))), 1)  # todas com o mesmo sinal (orientadas igual)

    def test_encaixe_igual_ao_kdtree(self):
        P = ondulada()
        A = RG.nuvem_voxel(P, 1.0, normais=False)
        B = RG.nuvem_voxel(P[::2] + [0.2, 0.0, 0.1], 1.0, normais=False)
        T = pose(0.3, [0.1, 0.0, 0.0])
        a, b = gpu.para_cpu(A.pts), gpu.para_cpu(B.pts) @ T[:3, :3].T + T[:3, 3]
        d, _ = cKDTree(a).query(b)
        m = d <= 0.3
        fit, rmse, n = RG.avaliar_encaixe(B, A, 0.3, T)
        self.assertEqual(n, int(m.sum()))
        self.assertAlmostEqual(fit, m.mean(), places=12)
        self.assertAlmostEqual(rmse, float(np.sqrt((d[m] ** 2).mean())), places=6)

    def test_icp_recupera_pose(self):
        P = ondulada()
        T0 = pose(0.5, [0.4, -0.3, 0.2])
        A = RG.nuvem_voxel(P, 1.0)
        B = RG.nuvem_voxel(P @ T0[:3, :3].T + T0[:3, 3], 1.0)
        T, info = RG.icp(B, A, np.eye(4), 3.0, 60)
        D = T @ T0
        self.assertLess(np.linalg.norm(D[:3, 3]), 0.01)
        self.assertLess(np.abs(D[:3, :3] - np.eye(3)).max(), 1e-3)
        self.assertGreater(info["fitness"], 0.99)

    def test_icp_robusto_ignora_ponto_fora(self):
        P = ondulada()
        A = RG.nuvem_voxel(P, 1.0)
        Pb = np.vstack([P, np.c_[np.random.default_rng(3).uniform(0, 100, (2000, 2)), np.full(2000, 6.0)]])
        B = RG.nuvem_voxel(Pb, 1.0)
        T, _ = RG.icp(B, A, pose(0.1, [0.1, 0, 0]), 3.0, 40, robusto=0.6)
        self.assertLess(np.linalg.norm(T[:3, 3]), 0.15)

    def test_matriz_informacao_referencia_numpy(self):
        P = ondulada()
        A = RG.nuvem_voxel(P, 1.0)
        B = RG.nuvem_voxel(P[::2], 1.0)
        T = pose(0.2, [0.05, 0.0, 0.0])
        H = RG.matriz_informacao(B, A, 1.0, T)
        a = gpu.para_cpu(A.pts)
        b = gpu.para_cpu(B.pts) @ T[:3, :3].T + T[:3, 3]
        d, j = cKDTree(a).query(b)
        ref = np.zeros((6, 6))
        for q in a[j[d <= 1.0]]:
            x, y, z = q
            G = np.array([[0, z, -y, 1, 0, 0], [-z, 0, x, 0, 1, 0], [y, -x, 0, 0, 0, 1]])
            ref += G.T @ G
        np.testing.assert_allclose(H, ref, rtol=1e-9)
        np.testing.assert_allclose(H, H.T, rtol=1e-12)

    def test_residuo_plano_igual_ao_codigo_antigo(self):
        P = ondulada()
        A = RG.nuvem_voxel(P, 1.0)
        B = RG.nuvem_voxel(P[::2] + [0.0, 0.0, 0.15], 1.0, normais=False)
        T = pose(0.0, [0.1, 0.0, 0.0])
        r = RG.residuo_plano(A, B, T)
        a, N = gpu.para_cpu(A.pts), gpu.para_cpu(A.normais)
        Bt = gpu.para_cpu(B.pts) @ T[:3, :3].T + T[:3, 3]
        d, idx = cKDTree(a).query(Bt)
        m = d < 2.0
        Bm = Bt[m]
        res = np.einsum("ij,ij->i", Bm - a[idx[m]], N[idx[m]])
        k = np.floor(Bm / 30.0).astype(np.int64)
        u, inv = np.unique(k, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        cnt = np.bincount(inv)
        ordem = np.argsort(inv, kind="stable")
        vies = [abs(float(np.median(res[g]))) for g, c in zip(np.split(ordem, np.cumsum(cnt)[:-1]), cnt) if c >= 100]
        self.assertAlmostEqual(r[0], float(m.mean()), places=12)
        self.assertAlmostEqual(r[1], float(np.median(vies)), places=9)
        self.assertAlmostEqual(r[2], float(np.percentile(vies, 90)), places=9)
        self.assertAlmostEqual(r[3], float(np.median(np.abs(res))), places=9)
        self.assertAlmostEqual(r[1], 0.137, delta=0.01)  # desvio imposto de 0,15 mm em z

    def test_residuo_plano_aceita_pointcloud_do_open3d(self):
        import open3d as o3d

        P = ondulada()
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(1.0)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3.0, max_nn=40))
        b = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[::2])).voxel_down_sample(1.0)
        r = AS.residuo_plano(p, b, np.eye(4))
        self.assertGreater(r[0], 0.95)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class AlinharDiretoGpuTest(unittest.TestCase):
    def test_custo_e_residuo_iguais_a_referencia_numpy(self):
        X = ondulada(passo=1.5, lado=90.0)
        with tempfile.TemporaryDirectory() as tmp:
            nos = []
            for i in range(2):
                arq = os.path.join(tmp, f"n{i}.npy")
                np.save(arq, X)
                nos.append(dict(scan=f"s{i}", seg=0, k0=0, k1=0, arquivo=arq))
            P, N, arv = AD.carregar(nos)
            T = [np.eye(4), pose(0.3, [0.3, -0.2, 0.1])]
            sel = np.arange(0, len(X), 2)
            t = AD._termos_pares(P, N, arv, T, [(0, 1)], [sel], 2.0, 0.5)
            # referencia: o que o codigo antigo (cKDTree + numpy) calculava com as mesmas normais
            p0, n0, p1 = gpu.para_cpu(P[0]), gpu.para_cpu(N[0]), gpu.para_cpu(P[1])
            Rr = np.linalg.inv(T[0]) @ T[1]
            q = p1[sel] @ Rr[:3, :3].T + Rr[:3, 3]
            d, idx = cKDTree(p0).query(q)
            m = d < 2.0
            r = np.einsum("ij,ij->i", q[m] - p0[idx[m]], n0[idx[m]])
            c = 0.5
            rho = c * c / 6.0
            u = (r / c) ** 2
            dentro = np.abs(r) < c
            custo = rho * (~m).sum() + np.sum(np.where(dentro, rho * (1 - (1 - u) ** 3), rho))
            w = np.where(dentro, (1 - u) ** 2, 0.0)
            self.assertAlmostEqual(t["custo"], float(custo), places=6)
            self.assertAlmostEqual(t["n_efetivo"], float(w.sum()), places=6)
            self.assertAlmostEqual(t["rms"], float(np.sqrt((w * r * r).sum() / w.sum())), places=9)
            self.assertEqual(t["H"].shape, (12, 12))
            np.testing.assert_allclose(t["H"], t["H"].T, atol=1e-6 * np.abs(t["H"]).max())


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class ConsensoGpuTest(unittest.TestCase):
    def test_vista_e_votacao_na_gpu(self):
        import consenso_visibilidade as CV

        D = np.full((40, 40), 50.0, np.float32)
        D[10:12, :] = np.nan
        v = CV.Vista(D, (50.0, 50.0, 20.0, 20.0), np.eye(4), "B")
        self.assertTrue(gpu._eh_gpu(v.D) and gpu._eh_gpu(v.Dmin) and gpu._eh_gpu(v.val9))
        self.assertFalse(bool(v.val9[10, 20]))  # janela 3x3 com buraco nao e valida
        self.assertTrue(bool(v.val9[25, 20]))
        k = CV.chaves_voxel(np.array([[-0.1, 0, 0], [0.1, 0, 0], [0.4, 0, 0]]), 0.5)
        self.assertIsInstance(k, np.ndarray)  # numpy entra, numpy sai
        self.assertEqual(k[1], k[2])
        self.assertNotEqual(k[0], k[1])
        kg = CV.chaves_voxel(gpu.para_gpu(np.array([[-0.1, 0, 0], [0.1, 0, 0]])), 0.5)
        self.assertTrue(gpu._eh_gpu(kg))

    def test_sem_opcao_motor_cpu(self):
        r = subprocess.run(
            [sys.executable, str(PIPE / "consenso_visibilidade.py"), "--help"], capture_output=True, text=True
        )
        self.assertEqual(r.returncode, 0)
        self.assertNotIn("--motor", r.stdout)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class AprenderGpuTest(unittest.TestCase):
    def test_metricas_iguais_ao_kdtree(self):
        import aprender

        rng = np.random.default_rng(0)
        E = rng.uniform(0, 20, (20000, 3))
        F = E[rng.choice(len(E), 3000, replace=False)] + rng.normal(0, 0.1, (3000, 3))
        keep = rng.uniform(size=len(E)) > 0.3
        rec, lixo, kf = aprender.metricas(F, E, keep)
        d_t = cKDTree(F).query(E)[0]
        d, _ = cKDTree(E).query(F)
        cob_t = d < 0.6
        cob_k = cKDTree(E[keep]).query(F)[0] < 0.6
        self.assertAlmostEqual(rec, cob_k.sum() / cob_t.sum(), places=12)
        self.assertAlmostEqual(lixo, float((d_t[keep] > 3.0).mean()), places=12)
        self.assertAlmostEqual(kf, float(keep.mean()), places=12)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class UniaoScansGpuTest(unittest.TestCase):
    def test_plano_dominante_e_pca(self):
        import uniao_scans as U

        rng = np.random.default_rng(0)
        plano = np.c_[rng.uniform(0, 60, (30000, 2)), rng.normal(0, 0.02, 30000)]
        resto = rng.uniform(0, 60, (3000, 3))
        n, d = U._plano_dominante(np.vstack([plano, resto]))
        self.assertAlmostEqual(abs(n[2]), 1.0, places=2)
        self.assertAlmostEqual(d, 0.0, delta=0.05)
        c, V = U._pca(np.c_[rng.uniform(0, 100, 5000), rng.uniform(0, 20, 5000), rng.uniform(0, 5, 5000)])
        self.assertAlmostEqual(abs(V[0, 0]), 1.0, places=1)
        self.assertGreater(np.linalg.det(V), 0)


def _cena_segmentos(pasta):
    """3 scans em x = [0,40], [30,70] e [0,70] (plano de 1 mm de passo, 3 de largura) como segmentos de um quadro."""
    ind, poses = {}, {}
    faixas = {"A": (0, 40), "B": (30, 70), "C": (0, 70)}
    for g, (x0, x1) in faixas.items():
        x, y = np.meshgrid(np.arange(x0, x1, 1.0), np.arange(0, 3, 1.0))
        P = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)].astype(np.float32)
        arq = os.path.join(pasta, f"{g}_000.npy")
        np.save(arq, P)
        ind[g] = [dict(scan=g, seg=0, k0=0, k1=0, arquivo=arq, n=len(P), n_brutos=len(P))]
        poses[g] = [dict(k0=0, k1=0, seg=0, T=np.eye(4).tolist())]
    with open(os.path.join(pasta, "segmentos_indice.json"), "w") as f:
        json.dump(ind, f)
    with open(os.path.join(pasta, "segmentos.json"), "w") as f:
        json.dump(dict(referencia="C", segmentos=poses), f)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class CoberturaGpuTest(unittest.TestCase):
    def test_selecionar_segmentos_escolhe_o_minimo(self):
        with tempfile.TemporaryDirectory() as tmp:
            _cena_segmentos(tmp)
            sel = os.path.join(tmp, "sel.json")
            r = subprocess.run(
                [
                    sys.executable,
                    str(PIPE / "selecionar_segmentos.py"),
                    tmp,
                    os.path.join(tmp, "segmentos.json"),
                    "--saida",
                    sel,
                    "--redundancia",
                    "1",
                    "--apoio-min",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            with open(sel) as f:
                j = json.load(f)
            self.assertEqual(j["por_scan"], {"C": [0]})  # C sozinho cobre tudo
            self.assertEqual(j["deficit_restante_pct"], 0.0)
            r = subprocess.run(
                [
                    sys.executable,
                    str(PIPE / "selecionar_segmentos.py"),
                    tmp,
                    os.path.join(tmp, "segmentos.json"),
                    "--saida",
                    sel,
                    "--redundancia",
                    "2",
                    "--apoio-min",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            with open(sel) as f:
                self.assertEqual(len(json.load(f)["por_scan"]), 3)  # redundancia 2 pede os tres

    def test_cobertura_ordem_gulosa(self):
        with tempfile.TemporaryDirectory() as tmp:
            _cena_segmentos(tmp)
            saida = os.path.join(tmp, "cob")
            r = subprocess.run(
                [
                    sys.executable,
                    str(PIPE / "cobertura.py"),
                    tmp,
                    os.path.join(tmp, "segmentos.json"),
                    "--saida",
                    saida,
                    "--min-voxels",
                    "1",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            with open(os.path.join(saida, "cobertura.json")) as f:
                j = json.load(f)
            self.assertEqual(j["ordem"][0], "C")
            self.assertAlmostEqual(j["passos"][0]["acumulado_pct"], 100.0, places=6)


if __name__ == "__main__":
    unittest.main()
