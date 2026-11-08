"""Planos de apoio e remapeamento: recusam erros antes de abrir os frames."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from apoio_validado import carregar_planos, validar_perda


class ApoioValidadoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.arquivo = self.root / "planos.json"

    def escrever(self, doc):
        self.arquivo.write_text(json.dumps(doc), encoding="utf-8")
        return carregar_planos(self.arquivo, self.root)

    def test_remapeia_avaliacao_e_normaliza_tambem_deslocamento(self):
        (self.root / "identificadores.json").write_text('{"antigo": "novo"}')
        r = self.escrever({"scans": {"antigo": {"plano": {"n": [0, 0, 2], "d": -4}}}})
        np.testing.assert_array_equal(r["novo"]["n"], [0, 0, 1])
        self.assertEqual(r["novo"]["d"], -2)

    def test_rejeita_plano_invalido_ou_que_atravessa_peca(self):
        for plano in (
            {"n": [0, 0, 0], "d": 0},
            {"n": [1, 2], "d": 0},
            {"n": [0, 0, 1], "d": float("nan")},
            {"n": [0, 0, 1], "d": 0, "dispersao_mad_mm": 0.5},
            {"n": [0, 0, 1], "d": 0, "selecao_abaixo_menos_05_pct": 6},
        ):
            with self.subTest(plano=plano), self.assertRaises(ValueError):
                self.escrever({"scan": plano})

    def test_rejeita_planos_duplicados_apos_remapeamento(self):
        (self.root / "identificadores.json").write_text('{"antigo": "novo"}')
        plano = {"n": [0, 0, 1], "d": 0}
        with self.assertRaisesRegex(ValueError, "duplicado"):
            self.escrever({"antigo": plano, "novo": plano})

    def test_perda_excessiva_ou_corte_vazio_nao_passa(self):
        self.assertAlmostEqual(validar_perda(1000, 980, 0.3), 0.02)
        for antes, depois in ((1000, 560), (1000, 0), (0, 0), (10, 11)):
            with self.subTest(antes=antes, depois=depois), self.assertRaises(ValueError):
                validar_perda(antes, depois, 0.3)


if __name__ == "__main__":
    unittest.main()
