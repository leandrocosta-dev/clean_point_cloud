"""Camada GPU (pipeline/gpu.py) contra referencias scipy/numpy em dados sinteticos."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / 'pipeline'))
sys.path.insert(0, str(RAIZ / 'pipeline' / 'revo_frames'))
import gpu  # noqa: E402

try:
    gpu.xp()
    TEM_GPU = gpu.em_gpu()
except gpu.ErroGPU:
    TEM_GPU = False


def _nuvem(n=40000, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 40, (n, 2))
    z = 2 * np.sin(xy[:, 0] / 5) + 0.02 * rng.standard_normal(n)
    return np.column_stack([xy, z])


def _particao_igual(a, b):
    """Dois rotulamentos sao a mesma particao (iguais a menos do nome dos rotulos)."""
    pares = np.unique(np.column_stack([a, b]), axis=0)
    return len(pares) == len(np.unique(a)) == len(np.unique(b))


@unittest.skipUnless(TEM_GPU, 'CUDA/CuPy ausente')
class VizinhosTest(unittest.TestCase):
    def test_knn_exato_igual_ao_ckdtree(self):
        P = _nuvem()
        Q = _nuvem(5000, 1)
        for k in (1, 8, 16, 33, 64):
            d, j = gpu.knn(P, Q, k, raio_max=3.0)
            dr, jr = cKDTree(P).query(Q, k, distance_upper_bound=3.0)
            dr, jr = dr.reshape(len(Q), k), jr.reshape(len(Q), k)
            achou = np.isfinite(dr)
            np.testing.assert_allclose(d[achou], dr[achou], atol=2e-4)
            self.assertTrue(np.all(j[~achou] == -1) and np.all(np.isinf(d[~achou])))
            self.assertGreater((j[achou] == jr[achou]).mean(), 0.999)

    def test_knn_sem_raio_acha_sempre_os_k(self):
        P = _nuvem(20000)
        Q = np.vstack([_nuvem(3000, 2), [[500.0, 500.0, 500.0]]])       # uma consulta muito longe
        d, j = gpu.knn(P, Q, 12)
        dr, jr = cKDTree(P).query(Q, 12)
        self.assertTrue(np.all(j >= 0))
        np.testing.assert_allclose(d, dr, rtol=1e-4, atol=2e-4)
        self.assertGreater((j == jr).mean(), 0.999)

    def test_vizinho_mais_proximo_com_e_sem_limite(self):
        P = _nuvem()
        Q = _nuvem(6000, 3) + [0, 0, 0.3]
        Q[:100] += [100, 0, 0]                                           # fora do alcance
        d, j = gpu.vizinho_mais_proximo(P, Q, dist_max=0.4)
        dr, jr = cKDTree(P).query(Q, 1, distance_upper_bound=0.4)
        achou = np.isfinite(dr)
        self.assertTrue(np.all(j[~achou] == -1) and np.all(np.isinf(d[~achou])))
        self.assertTrue(achou.sum() > 1000 and (~achou).sum() >= 100)
        np.testing.assert_allclose(d[achou], dr[achou], atol=2e-4)
        self.assertGreater((j[achou] == jr[achou]).mean(), 0.999)
        d2, j2 = gpu.vizinho_mais_proximo(P, Q)                          # sem limite
        dr2, jr2 = cKDTree(P).query(Q)
        np.testing.assert_allclose(d2, dr2, rtol=1e-4, atol=2e-4)
        self.assertGreater((j2 == jr2).mean(), 0.999)

    def test_contar_no_raio(self):
        P = _nuvem()
        Q = _nuvem(3000, 4)
        n = gpu.contar_no_raio(P, Q, 0.8)
        ref = cKDTree(P).query_ball_point(Q, 0.8, return_length=True)
        self.assertLess(np.abs(n - ref).max(), 2)                        # bordas de float32 no raio exato

    def test_normais_pca_k_e_raio_contra_numpy(self):
        P = _nuvem(30000)
        Q = P[:2000]
        Nk = gpu.normais_pca(P, k=20, consultas=Q, orientar_para=[0, 0, 100])
        _, j = cKDTree(P).query(Q, 20)
        Y = P[j] - P[j].mean(1, keepdims=True)
        v = np.linalg.eigh(np.einsum("nki,nkj->nij", Y, Y))[1][:, :, 0]
        v *= np.sign(v[:, 2])[:, None]
        np.testing.assert_allclose(np.abs((Nk * v).sum(1)), 1, atol=1e-5)
        self.assertTrue(np.all(Nk[:, 2] > 0))                            # orientadas para a camera acima
        Nr, curv = gpu.normais_pca(P, raio=1.2, consultas=Q, orientar_para=[0, 0, 100], com_curvatura=True)
        listas = cKDTree(P).query_ball_point(Q, 1.2)
        vr = []
        for lst in listas:
            y = P[lst] - P[lst].mean(0)
            vv = np.linalg.eigh(y.T @ y)[1][:, 0]
            vr.append(vv * np.sign(vv[2]))
        self.assertGreater(np.abs((Nr * np.array(vr)).sum(1)).min(), 0.999)
        self.assertTrue(np.all(curv < 0.05))

    def test_normais_plano_exato_e_isolado_nan(self):
        rng = np.random.default_rng(0)
        P = np.column_stack([rng.uniform(0, 10, (5000, 2)), np.zeros(5000)])
        N = gpu.normais_pca(P, raio=1.0, orientar_para=[5, 5, 10])
        np.testing.assert_allclose(N[:, 2], 1.0, atol=1e-5)
        P2 = np.vstack([P, [[100, 100, 100.0]]])
        N2 = gpu.normais_pca(P2, raio=1.0)
        self.assertTrue(np.all(np.isnan(N2[-1])))

    def test_remover_soltos(self):
        rng = np.random.default_rng(5)
        P = np.column_stack([rng.uniform(0, 20, (20000, 2)), 0.01 * rng.standard_normal(20000)])
        soltos = rng.uniform([0, 0, 5], [20, 20, 30], (50, 3))
        X = np.vstack([P, soltos])
        manter = gpu.remover_soltos(X, k=12, desvios=2.0)
        self.assertEqual(manter.dtype, bool)
        self.assertFalse(manter[-50:].any())
        self.assertGreater(manter[:-50].mean(), 0.9)
        m2 = gpu.remover_soltos(X, raio=1.0, min_vizinhos=5)
        self.assertFalse(m2[-50:].any())
        self.assertGreater(m2[:-50].mean(), 0.99)


@unittest.skipUnless(TEM_GPU, 'CUDA/CuPy ausente')
class AgrupamentoTest(unittest.TestCase):
    def test_reduzir_voxel_igual_numpy(self):
        P = _nuvem(50000)
        W = np.random.default_rng(1).uniform(0.5, 2, len(P))
        for pesos, min_obs in ((None, 1), (W, 3)):
            C, cont = gpu.reduzir_voxel(P, 0.5, pesos=pesos, min_obs=min_obs)
            o = P.min(0)
            idx = np.floor((P - o) / 0.5).astype(np.int64)
            ch = (idx[:, 0] << 42) | (idx[:, 1] << 21) | idx[:, 2]
            u, inv, cnt = np.unique(ch, return_inverse=True, return_counts=True)
            w = np.ones(len(P)) if pesos is None else pesos
            ref = np.stack([np.bincount(inv, weights=w * P[:, c]) for c in range(3)], 1) / np.bincount(inv, weights=w)[:, None]
            ok = cnt >= min_obs
            np.testing.assert_allclose(C, ref[ok], atol=1e-9)
            np.testing.assert_array_equal(cont, cnt[ok])

    def test_agrupar_somar_e_chaves(self):
        rng = np.random.default_rng(2)
        k = rng.integers(0, 50, 10000)
        u, inv, cnt = gpu.agrupar(k)
        ur, invr, cr = np.unique(k, return_inverse=True, return_counts=True)
        np.testing.assert_array_equal(u, ur)
        np.testing.assert_array_equal(inv, invr)
        np.testing.assert_array_equal(cnt, cr)
        V = rng.standard_normal((10000, 3))
        S = gpu.somar_por_grupo(V, inv)
        ref = np.stack([np.bincount(invr, weights=V[:, c]) for c in range(3)], 1)
        np.testing.assert_allclose(S, ref, atol=1e-9)
        np.testing.assert_allclose(gpu.somar_por_grupo(V[:, 0], inv, 50), ref[:, 0], atol=1e-9)
        P = _nuvem(1000)
        o = P.min(0)
        ch = gpu.chaves_voxel(P, 0.5, o)
        c = gpu.decodificar_voxel(ch, 0.5, o)
        self.assertLessEqual(np.abs(c - P).max(), 0.25 + 1e-9)
        with self.assertRaises(ValueError):
            gpu.chaves_voxel(P, 1e-6)

    def test_entrada_cupy_sai_cupy(self):
        import cupy as cp
        P = cp.asarray(_nuvem(2000))
        C, n = gpu.reduzir_voxel(P, 0.5)
        self.assertIsInstance(C, cp.ndarray)
        d, j = gpu.vizinho_mais_proximo(P, P[:10], 0.2)
        self.assertIsInstance(j, cp.ndarray)
        self.assertIsInstance(gpu.reduzir_voxel(cp.asnumpy(P), 0.5)[0], np.ndarray)

    def test_blocos_cobrem_tudo(self):
        fatias = list(gpu.blocos(1000, bytes_por_item=1 << 40))          # nada cabe: 1 item por bloco
        self.assertEqual(len(fatias), 1000)
        fatias = list(gpu.blocos(10_000_000, bytes_por_item=100))
        self.assertEqual(fatias[0][0], 0)
        self.assertEqual(fatias[-1][1], 10_000_000)
        self.assertTrue(all(b == a2 for (_, b), (a2, _) in zip(fatias, fatias[1:])))


@unittest.skipUnless(TEM_GPU, 'CUDA/CuPy ausente')
class GeometriaTest(unittest.TestCase):
    def test_ransac_acha_plano_conhecido_com_outliers(self):
        rng = np.random.default_rng(7)
        n0 = np.array([0.2, -0.3, 0.9])
        n0 /= np.linalg.norm(n0)
        d0 = -12.0
        uv = rng.uniform(-80, 80, (30000, 2))
        base = np.cross(n0, [1, 0, 0])
        base /= np.linalg.norm(base)
        base2 = np.cross(n0, base)
        centro = -d0 * n0                                                   # ponto do plano: n0 . c + d0 = 0
        plano = centro + uv[:, :1] * base + uv[:, 1:] * base2 + 0.05 * rng.standard_normal((30000, 1)) * n0
        objetos = centro + rng.uniform(-80, 80, (12000, 1)) * base + rng.uniform(-80, 80, (12000, 1)) * base2             + rng.uniform(5, 40, (12000, 1)) * n0                           # todos do lado + do plano
        P = np.vstack([plano, objetos])
        n, d, frac = gpu.ransac_plano(P, limiar=0.3, iteracoes=500, semente=1)
        self.assertGreater(float(n @ n0), 0.9999)                          # orientada para o lado dos objetos
        self.assertAlmostEqual(d, d0, delta=0.05)
        self.assertGreater(frac, 0.6)
        import revo_projeto as R
        nr, dr, fr = R.plano_ransac(P, 0.3, 500)
        self.assertGreater(float(n @ nr), 0.9999)
        self.assertAlmostEqual(d, dr, delta=0.05)

    def test_ransac_orienta_para_o_lado_dos_objetos(self):
        rng = np.random.default_rng(8)
        mesa = np.column_stack([rng.uniform(-50, 50, (8000, 2)), 0.02 * rng.standard_normal(8000)])
        peca = rng.uniform([-10, -10, 3], [10, 10, 20], (2000, 3))
        n, d, _ = gpu.ransac_plano(np.vstack([mesa, peca]), 0.3, 300)
        self.assertGreater(n[2], 0.999)
        self.assertAlmostEqual(d, 0.0, delta=0.05)
        n, d, _ = gpu.ransac_plano(np.vstack([mesa, peca]) * [1, 1, -1], 0.3, 300)
        self.assertLess(n[2], -0.999)

    @staticmethod
    def _cena(n=15000, seed=0):
        """Canto de caixa: tres planos ortogonais de 40 x 40 mm. Normais exatas; ICP bem condicionado e sem vies de
        amostragem (num plano o residuo ponto-plano e zero para qualquer vizinho)."""
        rng = np.random.default_rng(seed)
        a = rng.uniform(0, 40, (3 * n, 2))
        pts = np.vstack([np.column_stack([a[:n], np.zeros(n)]),
                         np.column_stack([a[n:2 * n, 0], np.zeros(n), a[n:2 * n, 1]]),
                         np.column_stack([np.zeros(n), a[2 * n:]])])
        nrm = np.vstack([np.tile(e, (n, 1)) for e in np.eye(3)[[2, 1, 0]]])
        return pts, nrm

    def test_icp_recupera_transformacao_conhecida(self):
        alvo, nrm = self._cena()
        rng = np.random.default_rng(3)
        sel = rng.choice(len(alvo), 30000, replace=False)
        ang = np.deg2rad([0.5, -0.4, 0.6])
        from gpu import _rotacao
        Rv = _rotacao(ang)
        tv = np.array([0.25, -0.20, 0.20])
        Tverd = np.eye(4)
        Tverd[:3, :3], Tverd[:3, 3] = Rv, tv                              # origem -> alvo
        origem = (alvo[sel] - tv) @ Rv                                    # inverso: alvo = Rv @ origem + tv
        # 10% de pontos espurios a 1,25 mm da superficie (dentro de dist_max, fora da escala c do Tukey)
        longe_da_quina = np.nonzero(((alvo == 0) | ((alvo > 5) & (alvo < 35))).all(1))[0]
        k = rng.choice(longe_da_quina, 3000, replace=False)
        lixo = (alvo[k] + 1.25 * nrm[k] - tv) @ Rv
        origem_ruidosa = np.vstack([origem, lixo])
        T, info = gpu.icp_ponto_plano(origem_ruidosa, alvo, nrm, np.eye(4), dist_max=1.8, iteracoes=60,
                                      perda="tukey", c=1.0)
        self.assertLess(np.abs(T - Tverd).max(), 1e-4, (T, info))
        self.assertGreater(info["fitness"], 0.8)
        T2, info2 = gpu.icp_ponto_plano(origem, alvo, nrm, None, dist_max=1.8, iteracoes=60, perda="l2")
        self.assertLess(np.abs(T2 - Tverd).max(), 1e-4)
        self.assertTrue(info2["convergiu"])

    def test_componentes_iguais_ao_scipy_ndimage(self):
        rng = np.random.default_rng(4)
        blocos_ = [rng.uniform(c, c + 6, (3000, 3)) for c in (0, 8.0, 20.0)]     # 3 nuvens separadas
        blocos_.append(np.array([[3.0, 3.0, 40.0]]))                              # um ponto isolado
        P = np.vstack(blocos_)
        voxel = 0.8
        for conect, estrutura in ((26, np.ones((3, 3, 3))), (6, ndimage.generate_binary_structure(3, 1)),
                                  (18, ndimage.generate_binary_structure(3, 2))):
            rot, tam = gpu.componentes_conexos_voxel(P, voxel, conect)
            idx = np.floor((P - P.min(0)) / voxel).astype(np.int64)
            g = np.zeros(idx.max(0) + 1, bool)
            g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
            lab, nlab = ndimage.label(g, structure=estrutura)
            ref = lab[idx[:, 0], idx[:, 1], idx[:, 2]]
            self.assertEqual(len(tam), nlab)
            self.assertTrue(_particao_igual(rot, ref))
            self.assertEqual(int(tam.sum()), len(P))
            self.assertTrue(np.all(np.diff(tam) <= 0))
            self.assertEqual(int((rot == 0).sum()), int(tam[0]))

    def test_componentes_aleatorio_denso_igual_ao_scipy(self):
        rng = np.random.default_rng(6)
        P = rng.uniform(0, 30, (6000, 3))
        rot, tam = gpu.componentes_conexos_voxel(P, 1.0)
        idx = np.floor((P - P.min(0)) / 1.0).astype(np.int64)
        g = np.zeros(idx.max(0) + 1, bool)
        g[idx[:, 0], idx[:, 1], idx[:, 2]] = True
        lab, nlab = ndimage.label(g, structure=np.ones((3, 3, 3)))
        self.assertEqual(len(tam), nlab)
        self.assertTrue(_particao_igual(rot, lab[idx[:, 0], idx[:, 1], idx[:, 2]]))


class AmbienteTest(unittest.TestCase):
    """Comportamento de xp() sem CUDA / com fallback de depuracao e ausencia do aviso de CUDA_PATH."""

    def _py(self, codigo, **env):
        e = {k: v for k, v in os.environ.items() if k not in ("CUDA_PATH", "REVO_PERMITIR_CPU")}
        e.update(env)
        return subprocess.run([sys.executable, "-W", "error::UserWarning", "-c", codigo], cwd=str(RAIZ / "tests"), env=e,
                              capture_output=True, text=True, timeout=300)

    def test_sem_cuda_levanta_erro_gpu(self):
        r = self._py("import gpu\ntry:\n gpu.xp()\nexcept gpu.ErroGPU as e:\n print('ERRO', e)",
                     CUDA_VISIBLE_DEVICES="-1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ERRO", r.stdout)
        self.assertIn("REVO_PERMITIR_CPU", r.stdout)

    def test_fallback_cpu_com_variavel(self):
        codigo = ("import warnings, numpy as np, gpu\n"
                  "warnings.simplefilter('ignore')\n"
                  "assert gpu.xp() is np and not gpu.em_gpu()\n"
                  "P = np.random.default_rng(0).uniform(0, 10, (3000, 3))\n"
                  "d, j = gpu.knn(P, P, 4, 1.0); assert (j[:, 0] == np.arange(3000)).all()\n"
                  "d, j = gpu.vizinho_mais_proximo(P, P[:5] + 0.01, 0.5); assert (j >= 0).all()\n"
                  "C, n = gpu.reduzir_voxel(P, 1.0); assert n.sum() == 3000\n"
                  "r, t = gpu.componentes_conexos_voxel(P, 1.0); assert t.sum() == 3000\n"
                  "N = gpu.normais_pca(P, k=10); assert np.isfinite(N).all()\n"
                  "print('OK')")
        r = self._py(codigo, REVO_PERMITIR_CPU="1", CUDA_VISIBLE_DEVICES="-1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("OK", r.stdout)

    @unittest.skipUnless(TEM_GPU, 'CUDA/CuPy ausente')
    def test_cuda_path_definido_sem_aviso_do_cupy(self):
        r = self._py("import gpu, os\ngpu.xp()\nprint(os.environ['CUDA_PATH'])")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(Path(r.stdout.strip()).exists())


if __name__ == "__main__":
    unittest.main()
