"""Regressoes reais de cache, agrupamento e contrato de orientacao da fusao GPU.

python -m unittest tests.test_fusao_regressoes -v
"""

import json
import tempfile
import unittest
from pathlib import Path

import gpu
import numpy as np
from comum import sha256

from tests.gpu_utils import CUDA_DISPONIVEL


def orientar_cpu_original(P, N, celula=1.0, tol=0.8, cone=65.0):
    """Referencia numpy da regra original (HEAD), independente dos kernels e funcoes de producao.

    O cone segue o eixo PCA menor e cobre so um hemisferio. A visibilidade de um ponto
    depende do z-buffer; uma normal so vira se nenhuma direcao visivel a aprovar.
    """
    c = P[::50].mean(0)
    _, E = np.linalg.eigh(np.cov((P[::50] - c).T))
    V = E[:, ::-1]
    Q = ((P - c) @ V).astype(np.float32)
    Nq = (N @ V).astype(np.float32)
    dirs = [np.array([0, 0, 1.0])]
    for ang in np.radians(np.arange(25, cone + 1, 20)):
        for az in np.radians(np.arange(0, 360, 45)):
            dirs.append(np.array([np.sin(ang) * np.cos(az), np.sin(ang) * np.sin(az), np.cos(ang)]))
    visto = np.zeros(len(P), bool)
    bom = np.zeros(len(P), bool)
    for dr in dirs:
        dr = dr.astype(np.float32)
        u = np.cross(dr, [0, 0, 1.0] if abs(dr[2]) < 0.99 else [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(dr, u)
        s = Q @ dr
        gx = np.floor((Q @ u) / celula).astype(np.int64)
        gy = np.floor((Q @ v) / celula).astype(np.int64)
        key = (gx - gx.min()) * 100003 + (gy - gy.min())
        _, inv = np.unique(key, return_inverse=True)
        inv = inv.ravel()
        mx = np.full(inv.max() + 1, -1e9, np.float32)
        np.maximum.at(mx, inv, s)
        vis = s >= mx[inv] - tol
        visto |= vis
        bom |= vis & (Nq @ dr > 0)
    flip = visto & ~bom
    Nq[flip] *= -1
    return (Nq @ V.T).astype(np.float64), visto, flip


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class CachePontosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scan = self.root / "scan"
        (self.scan / "cache").mkdir(parents=True)
        (self.scan / "cache/frames.dataset").write_bytes(b"entrada intocada")
        (self.scan / "raw_preview.ply").write_bytes(b"preview intocado")
        self.raiz = self.root / "cache"
        self.K = (100.0, 101.0, 8.0, 8.0)
        self.chamadas = 0

    def gerar(self):
        self.chamadas += 1
        return gpu.para_gpu(np.arange(30.0).reshape(10, 3)), np.zeros((2, 3), np.float32), np.array([4, 6])

    def pontos(self, K=None):
        from cache_observacoes import pontos_scan

        return pontos_scan(self.scan, self.K if K is None else K, self.gerar, self.raiz)

    def test_cria_reusa_e_preserva_entrada(self):
        assinatura = sha256(self.scan / "cache/frames.dataset")
        a, b = self.pontos(), self.pontos()
        self.assertEqual((a[4], b[4], self.chamadas), (False, True, 1))
        np.testing.assert_array_equal(gpu.para_cpu(a[0]), gpu.para_cpu(b[0]))
        np.testing.assert_array_equal(b[2], [4, 6])
        self.assertEqual(sha256(self.scan / "cache/frames.dataset"), assinatura)
        self.assertEqual((self.scan / "raw_preview.ply").read_bytes(), b"preview intocado")

    def test_invalida_calibracao_dataset_e_artefato(self):
        self.pontos()
        self.assertFalse(self.pontos((100.1, 101.0, 8.0, 8.0))[4])
        (self.scan / "cache/frames.dataset").write_bytes(b"entrada nova de teste")
        self.assertFalse(self.pontos()[4])
        meta = self.pontos()[3]
        pasta = next(
            p.parent
            for p in (self.raiz / "pontos").glob("*/meta.json")
            if json.loads(p.read_text())["componentes"] == meta["componentes"]
        )
        np.save(pasta / "pontos.npy", np.full((10, 3), 999.0))
        r = self.pontos()
        self.assertFalse(r[4])
        self.assertEqual(self.chamadas, 4)
        np.testing.assert_array_equal(gpu.para_cpu(r[0]), np.arange(30.0).reshape(10, 3))

    def test_reconstroi_entrada_completa_mas_metadados_invalidos(self):
        self.pontos()
        arq = next((self.raiz / "pontos").glob("*/meta.json"))
        meta = json.loads(arq.read_text())
        meta["quadros"] = 3
        arq.write_text(json.dumps(meta))
        r = self.pontos()
        self.assertEqual((r[4], self.chamadas), (False, 2))
        self.assertEqual(r[3]["quadros"], 2)
        self.assertTrue(self.pontos()[4])

    def test_reconstroi_arquivos_ausentes(self):
        self.pontos()
        next((self.raiz / "pontos").glob("*/quadros.npz")).unlink()
        r = self.pontos()
        self.assertEqual((r[4], self.chamadas), (False, 2))

    def test_rejeita_contagem_incompativel_sem_publicar(self):
        from cache_observacoes import pontos_scan

        with self.assertRaisesRegex(ValueError, "inconsistentes"):
            pontos_scan(self.scan, self.K, lambda: (np.zeros((5, 3)), np.zeros((1, 3)), np.array([6])), self.raiz)
        self.assertFalse(list(self.raiz.glob("pontos/*/meta.json")))


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class AgrupamentoFusaoTest(unittest.TestCase):
    def test_grupos_numpy_e_cupy_com_limites_e_coordenadas_negativas(self):
        from fundir_nitido import grupos

        X = np.array([[-0.4, -0.2, 0], [-0.2, 0.0, 0.2], [0.0, 0.2, 0.4], [-0.4, -0.2, 0]])
        _, ref_i, ref_n = np.unique(np.floor(X / 0.2).astype(np.int64), axis=0, return_inverse=True, return_counts=True)
        for dados in (X, gpu.para_gpu(X)):
            inv, n = grupos(dados, 0.2)
            np.testing.assert_array_equal(gpu.para_cpu(inv), ref_i.ravel())
            np.testing.assert_array_equal(gpu.para_cpu(n), ref_n)


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class ConectividadeRaioTest(unittest.TestCase):
    def comparar(self, P, raio):
        from conectividade_gpu import componentes_por_raio
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        pares = cKDTree(P).query_pairs(raio, output_type="ndarray")
        _, ref = connected_components(
            coo_matrix((np.ones(len(pares)), pares.T), shape=(len(P), len(P))), directed=False
        )
        rot, tam = componentes_por_raio(P, raio)
        np.testing.assert_array_equal(gpu.para_cpu(rot), ref)
        np.testing.assert_array_equal(gpu.para_cpu(tam), np.bincount(ref))
        return gpu.para_cpu(rot), gpu.para_cpu(tam)

    def test_ponte_esparsa_entre_blocos_com_mais_de_64_vizinhos(self):
        from limpar_artefatos import componentes_por_raio
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(22)
        A = np.r_[rng.normal(0, 0.002, (200, 3)) + [-0.02, 0, 0], [[0.05, 0, 0]]]
        B = np.r_[rng.normal(0, 0.002, (200, 3)) + [1.11, 0, 0], [[1.04, 0, 0]]]
        P = np.r_[A, B, [[10, 0, 0]]]
        rng.shuffle(P)
        rot, tam = self.comparar(P, 1.0)
        self.assertEqual(sorted(tam.tolist()), [1, 402])
        # Demonstra a falha da aproximacao anterior: todos os k64 de cada extremo da ponte ficam no seu bloco.
        d, viz = cKDTree(P).query(P, k=64, distance_upper_bound=1.0)
        origem = np.broadcast_to(np.arange(len(P))[:, None], viz.shape)
        ok = np.isfinite(d)
        nc, _ = connected_components(
            coo_matrix((np.ones(ok.sum()), (origem[ok], viz[ok])), shape=(len(P), len(P))), directed=False
        )
        self.assertEqual(nc, 3)
        for k in (1, 64):
            rk, tk = componentes_por_raio(P, 1.0, k=k)
            np.testing.assert_array_equal(gpu.para_cpu(rk), rot)
            np.testing.assert_array_equal(gpu.para_cpu(tk), tam)

    def test_grafo_aleatorio_e_duplicatas_igual_scipy(self):
        rng = np.random.default_rng(23)
        P = rng.uniform(-3, 3, (2000, 3))
        P = np.r_[P, P[::50]]
        self.comparar(P, 0.4)

    def test_limiar_inclusivo_singleton_e_vazio(self):
        from conectividade_gpu import componentes_por_raio

        P = np.array([[0, 0, 0], [1, 0, 0], [np.nextafter(2.0, np.inf), 0, 0], [-1, 0, 0], [0, 0, 0]])
        self.comparar(P, 1.0)
        self.comparar(np.array([[4.0, -2, 0]]), 1.0)
        rot, tam = componentes_por_raio(np.empty((0, 3)), 1.0)
        self.assertEqual((len(rot), len(tam)), (0, 0))

    def test_cadeia_longa_independe_da_ordem_das_entradas(self):
        P = np.c_[np.arange(5000) * 0.4, np.zeros((5000, 2))]
        rng = np.random.default_rng(24)
        for _ in range(3):
            rng.shuffle(P)
            rot, tam = self.comparar(P, 0.41)
            np.testing.assert_array_equal(rot, np.zeros(len(P)))
            np.testing.assert_array_equal(tam, [len(P)])


if __name__ == "__main__":
    unittest.main()
