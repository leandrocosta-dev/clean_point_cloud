"""Consenso por visibilidade: espaco livre remove, falta de cobertura e oclusao preservam."""

import unittest

import numpy as np
from consenso_visibilidade import Vista, avaliar_candidatos, chaves_voxel, decidir

from tests.gpu_utils import CUDA_DISPONIVEL

K = (200.0, 200.0, 100.0, 100.0)


def camera(x, y, altura=100.0, plano=0.0, buraco=None):
    """Camera olhando para -z (eixo optico para baixo) vendo o plano z = plano."""
    T = np.eye(4)
    T[:3, 3] = (x, y, altura)
    D = np.full((200, 200), altura - plano, np.float32)
    if buraco is not None:
        D[buraco] = np.nan
    return T, D


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class VisibilidadeTest(unittest.TestCase):
    def vistas(self, buraco=None):
        out = []
        for scan, xs in (("B", (0.0, 2.0)), ("C", (5.0, 7.0))):
            for x in xs:
                T, D = camera(x, 0.0, buraco=buraco)
                out.append(Vista(D, K, T, scan))
        T, D = camera(0.0, 0.0)
        out.append(Vista(D, K, T, "A"))  # vista do proprio scan nunca vota
        return out

    def test_fiapo_atravessado_e_removido_superficie_confirmada(self):
        C = np.array(
            [
                [0.0, 0.0, 20.0],  # fiapo flutuando: B e C veem o plano atras dele
                [1.0, 1.0, 0.0],
            ]
        )  # superficie real: B e C confirmam
        v = avaliar_candidatos(C, "A", self.vistas(), tol=0.5, min_livre_quadros=2)
        self.assertEqual(v["outros_scans"], 2)
        np.testing.assert_array_equal(v["contradizem"], [2, 0])
        np.testing.assert_array_equal(v["confirmam"], [0, 2])
        np.testing.assert_array_equal(decidir(v, 2), [1, 2])

    def test_sem_cobertura_fica_inconclusivo(self):
        C = np.array([[500.0, 0.0, 0.0]])  # fora do campo de visao dos outros scans
        v = avaliar_candidatos(C, "A", self.vistas())
        self.assertEqual(int(v["cobrem"][0]), 0)
        self.assertEqual(int(decidir(v, 1)[0]), 0)

    def test_oclusao_fica_inconclusiva(self):
        C = np.array([[0.0, 0.0, -10.0]])  # atras do plano visto pelos outros: ocluido, nao livre
        v = avaliar_candidatos(C, "A", self.vistas())
        self.assertEqual(int(v["contradizem"][0]), 0)
        self.assertEqual(int(v["confirmam"][0]), 0)
        self.assertEqual(int(decidir(v, 1)[0]), 0)

    def test_espaco_livre_exige_janela_3x3_valida(self):
        # pixel vizinho sem dado ao redor da projecao do fiapo: nao conta como espaco livre
        buraco = (slice(99, 100), slice(None))
        C = np.array([[0.0, 0.0, 20.0]])
        v = avaliar_candidatos(C, "A", self.vistas(buraco=buraco), min_livre_quadros=1)
        self.assertEqual(int(v["contradizem"][0]), 0)

    def test_um_quadro_so_nao_basta_com_min_livre_2(self):
        T, D = camera(0.0, 0.0)
        vs = [Vista(D, K, T, "B")]
        v = avaliar_candidatos(np.array([[0.0, 0.0, 20.0]]), "A", vs, min_livre_quadros=2)
        self.assertEqual(int(v["contradizem"][0]), 0)

    def test_chaves_voxel_negativas_e_distintas(self):
        P = np.array([[-0.1, 0, 0], [0.1, 0, 0], [0.4, 0, 0], [-300, 200, -50]])
        k = chaves_voxel(P, 0.5)
        self.assertEqual(len(set(k.tolist())), 3)
        self.assertEqual(k[1], k[2])

    def test_cuda_corresponde_cpu(self):
        import cupy as cp
        from consenso_visibilidade import _para_xp

        rng = np.random.default_rng(0)
        C = np.c_[rng.uniform(-20, 20, (500, 2)), rng.uniform(-5, 30, 500)]
        a = avaliar_candidatos(C, "A", self.vistas())
        b = avaliar_candidatos(C, "A", _para_xp(self.vistas(), cp), xp=cp)
        for chave in ("confirmam", "contradizem", "cobrem"):
            np.testing.assert_array_equal(a[chave], b[chave])


if __name__ == "__main__":
    unittest.main()
