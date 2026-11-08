"""Casos adversariais para selecao conservadora de qualidade geometrica 3D."""

import unittest

import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL

if CUDA_DISPONIVEL:
    import selecao_qualidade as SQ


def _med(
    pontos,
    ruido=0.02,
    quadros=5,
    incidencia=0.9,
    normais=None,
    confiavel=True,
    rastreio=False,
    temporal=True,
    detalhe=False,
):
    n = len(pontos)
    return {
        "pontos": np.asarray(pontos, np.float64),
        "normais": np.tile([0.0, 0.0, 1.0], (n, 1)) if normais is None else np.asarray(normais, np.float64),
        "ruido_mm": np.full(n, ruido),
        "quadros": np.full(n, quadros),
        "incidencia": np.full(n, incidencia),
        "normal_confiavel": np.full(n, confiavel),
        "rastreio_ruim": np.full(n, rastreio),
        "evidencia_temporal": np.ones(n, bool),
        "consistencia_temporal": np.full(n, temporal),
        "detalhe_protegido": np.full(n, detalhe),
    }


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class SelecaoQualidadeTest(unittest.TestCase):
    def _rodar(self, medicoes):
        return SQ.selecionar_regioes(medicoes)

    def test_ruidoso_sobreposto_remove_somente_candidato(self):
        x, y = np.meshgrid(np.arange(5) * 0.1, np.arange(5) * 0.1)
        p = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        r = self._rodar({"ruim": _med(p, ruido=0.08, temporal=False), "bom": _med(p, ruido=0.02)})
        self.assertEqual(int(r["remover"]["ruim"].sum().get()), len(p))
        self.assertEqual(int(r["remover"]["bom"].sum().get()), 0)
        self.assertTrue(all(x == "bom" for x in r["referencia_scan"]["ruim"]))

    def test_detalhe_ressaltado_unico_preservado_com_conflito(self):
        x, y = np.meshgrid(np.arange(5) * 0.1, np.arange(5) * 0.1)
        base = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        relevo = base.copy()
        relevo[:, 2] = 0.08
        r = self._rodar({"detalhe": _med(relevo, ruido=0.08), "plano": _med(base, ruido=0.01)})
        self.assertEqual(int(r["remover"]["detalhe"].sum().get()), 0)
        self.assertGreater(r["relatorio"]["scans"]["detalhe"]["pendencias"]["conflito_geometrico"], 0)

    def test_regiao_exclusiva_preservada_e_pendente(self):
        a = np.array([[0.0, 0.0, 0.0], [10.0, 0, 0]])
        b = a[:1]
        r = self._rodar({"candidato": _med(a, ruido=0.08), "referencia": _med(b, ruido=0.01)})
        self.assertFalse(bool(r["remover"]["candidato"][1].get()))
        self.assertGreater(r["relatorio"]["scans"]["candidato"]["pendencias"]["suspeito_sem_cobertura"], 0)

    def test_parede_oposta_nao_cobre_pela_normal_com_sinal(self):
        p = np.array([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], float)
        q = p.copy()
        q[:, 2] = 0.01
        normals = np.tile([0, 0, -1.0], (len(q), 1))
        r = self._rodar({"a": _med(p, ruido=0.08), "b": _med(q, ruido=0.01, normais=normals)})
        self.assertEqual(int(r["remover"]["a"].sum().get()), 0)

    def test_empate_sem_quadros_incidente_rasante_e_normal_ambigua_preservam(self):
        p = np.array([[0, 0, 0], [0.1, 0, 0]], float)
        casos = [
            ("empate", _med(p, ruido=0.02), _med(p, ruido=0.02)),
            ("quadros", _med(p, ruido=0.08), _med(p, ruido=0.01, quadros=1)),
            ("rasante", _med(p, ruido=0.08), _med(p, ruido=0.01, incidencia=0.1)),
            ("normal", _med(p, ruido=0.08), _med(p, ruido=0.01, confiavel=False)),
        ]
        for nome, a, b in casos:
            with self.subTest(nome=nome):
                r = self._rodar({"a": a, "b": b})
                self.assertEqual(int(r["remover"]["a"].sum().get()), 0)

    def test_rastreio_sem_ref_segura_preserva_e_pende(self):
        p = np.array([[0, 0, 0], [2, 0, 0]], float)
        r = self._rodar({"a": _med(p, rastreio=True), "b": _med(p[:1], confiavel=False)})
        self.assertEqual(int(r["remover"]["a"].sum().get()), 0)
        self.assertGreater(r["relatorio"]["scans"]["a"]["pendencias"]["suspeito_sem_cobertura"], 0)

    def test_ruido_sem_evidencia_temporal_nao_autoriza_descarte(self):
        p = np.array([[0, 0, 0], [0.1, 0, 0]], float)
        ruim = _med(p, ruido=0.08)
        ruim.pop("evidencia_temporal")
        r = self._rodar({"ruim": ruim, "bom": _med(p, ruido=0.01)})
        self.assertEqual(int(r["remover"]["ruim"].sum().get()), 0)
        self.assertGreater(r["relatorio"]["scans"]["ruim"]["pendencias"]["suspeito_sem_prova_temporal"], 0)

    def test_normal_ambigua_na_fonte_ruim_preserva_e_pende(self):
        p = np.array([[0, 0, 0], [0.1, 0, 0]], float)
        r = self._rodar({"ruim": _med(p, ruido=0.08, confiavel=False), "bom": _med(p, ruido=0.01)})
        self.assertEqual(int(r["remover"]["ruim"].sum().get()), 0)
        self.assertGreater(r["relatorio"]["scans"]["ruim"]["pendencias"]["suspeito_sem_cobertura"], 0)


if __name__ == "__main__":
    unittest.main()
