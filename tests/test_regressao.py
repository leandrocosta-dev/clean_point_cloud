"""Banco de regressao (pipeline/regressao.py): base, ok, regressao, entrada mudada e comando falho."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import regressao as R  # noqa: E402


class RegressaoTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        self.rodadas, self.base = os.path.join(d, "rodadas"), os.path.join(d, "base")
        self.entrada = os.path.join(d, "entrada.txt")
        Path(self.entrada).write_text("x", encoding="utf-8")
        self.valor = os.path.join(d, "valor.txt")
        Path(self.valor).write_text("1.0", encoding="utf-8")
        codigo = ("import json,sys;v=float(open(sys.argv[1]).read());"
                  "json.dump({'r':{'v':v,'nome':'abc','n':3}},open(sys.argv[2],'w'))")
        self.caso = dict(nome="sintetico", entradas={"E": {"caminho": self.entrada}, "V": {"caminho": self.valor, "verificar": False}},
                         comandos=[["{PY}", "-c", codigo, "{V}", "{SAIDA}/m.json"]],
                         metricas=[{"tipo": "json", "arquivo": "{SAIDA}/m.json",
                                    "campos": {"v": "r.v", "nome": "r.nome", "n": "r.n"},
                                    "contem": {"tem_abc": {"chave": "r.nome", "texto": "ab"}}}],
                         limites={"tem_abc": {"igual": True}, "n": {"min": 1}},
                         tolerancias={"v": {"rel": 0.05}})

    def tearDown(self):
        self.tmp.cleanup()

    def rodar(self, registrar=False):
        return R.rodar_caso(self.caso, registrar, raiz_rodadas=self.rodadas, base_dir=self.base, log=lambda *a: None)

    def test_ciclo_completo(self):
        r, cod = self.rodar()
        self.assertEqual((r["status"], cod), ("sem_base", 4))
        r, cod = self.rodar(registrar=True)
        self.assertEqual((r["status"], cod), ("base_registrada", 0))
        self.assertEqual(json.load(open(os.path.join(self.base, "sintetico.json")))["valores"]["v"], 1.0)
        Path(self.valor).write_text("1.04", encoding="utf-8")     # dentro de 5%
        r, cod = self.rodar()
        self.assertEqual((r["status"], cod), ("ok", 0))
        Path(self.valor).write_text("1.2", encoding="utf-8")      # fora
        r, cod = self.rodar()
        self.assertEqual((r["status"], cod), ("regrediu", 1))
        self.assertTrue(any(p.startswith("v:") for p in r["problemas"]))
        Path(self.entrada).write_text("mudou", encoding="utf-8")
        r, cod = self.rodar()
        self.assertEqual((r["status"], cod), ("entrada_mudou", 3))
        self.assertEqual(len(os.listdir(self.rodadas)), len(set(os.listdir(self.rodadas))))

    def test_comando_falho(self):
        self.caso["comandos"] = [["{PY}", "-c", "import sys; sys.exit(5)"]]
        r, cod = self.rodar(registrar=True)
        self.assertEqual((r["status"], cod), ("comando_falhou", 2))
        self.assertFalse(os.path.exists(os.path.join(self.base, "sintetico.json")))

    def test_comparar_regras(self):
        self.assertEqual(R.comparar({"a": 2}, {"a": 2}, {}, {}), [])
        self.assertTrue(R.comparar({"a": 3}, {"a": 2}, {}, {}))                       # inteiro: igual por padrao
        self.assertEqual(R.comparar({"a": 1.01}, {"a": 1.0}, {}, {}), [])            # float: rel 2%
        self.assertTrue(R.comparar({"a": 1.1}, {"a": 1.0}, {}, {}))
        self.assertEqual(R.comparar({"a": 9.0}, {"a": 1.0}, {"a": {"ignorar": True}}, {}), [])
        self.assertEqual(R.comparar({"p": None}, None, {}, {"p": {"igual": None}}), [])
        self.assertTrue(R.comparar({}, None, {}, {"p": {"igual": None}}))             # ausente != None
        self.assertTrue(R.comparar({"q": 0.5}, None, {}, {"q": {"min": 0.9}}))
        self.assertTrue(R.comparar({}, {"a": 1}, {}, {}))                             # metrica sumiu

    def test_casos_do_repositorio_validos(self):
        casos = json.load(open(R.CASOS, encoding="utf-8"))["casos"]
        nomes = [c["nome"] for c in casos]
        self.assertEqual(len(nomes), len(set(nomes)))
        for c in casos:
            self.assertTrue(c.get("metricas"), c["nome"])
            for spec in c["metricas"]:
                self.assertIn(spec["tipo"], ("nuvem", "malha", "json"))


if __name__ == "__main__":
    unittest.main()
