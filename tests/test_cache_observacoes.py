"""Cache versionado de observacoes: reuso, invalidacao por conteudo e compatibilidade com origem/."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cache_observacoes import chave, obter, vincular
from revo_projeto import SIG_REVO, SIG_SQL

from tests.gpu_utils import CUDA_DISPONIVEL
from tests.test_precisao import criar_scan

CAL = {"0": dict(K=[100, 100, 8, 8], erro_preview_mm=0.0)}


class CacheTest(unittest.TestCase):
    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_reusa_e_invalida_quando_quadros_mudam(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = root / "scan"
            criar_scan(scan)
            antes = (scan / "cache/frames.dataset").read_bytes()
            p1, m1, r1 = obter(scan, root / "cache", CAL)
            p2, m2, r2 = obter(scan, root / "cache", CAL)
            self.assertFalse(r1)
            self.assertTrue(r2)
            self.assertEqual(p1, p2)
            self.assertEqual(m1["observacoes"], 5 * 256)
            self.assertEqual((scan / "cache/frames.dataset").read_bytes(), antes)  # origem intacta
            # muda uma profundidade: chave nova, sem reutilizar a entrada antiga
            arq = scan / "cache/frames.dataset"
            with arq.open("r+b") as f:
                f.write(SIG_SQL)
            cn = sqlite3.connect(arq)
            d = bytearray(cn.execute("select data from FrameRGBD where id=1").fetchone()[0])
            d[-4:] = np.float32(90).tobytes()
            cn.execute("update FrameRGBD set data=? where id=1", (bytes(d),))
            cn.commit()
            cn.close()
            with arq.open("r+b") as f:
                f.write(SIG_REVO)
            p3, _, r3 = obter(scan, root / "cache", CAL)
            self.assertNotEqual(p3, p1)
            self.assertFalse(r3)

    def test_calibracao_e_versao_entram_na_chave(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "scan"
            criar_scan(scan)
            k1, _ = chave(scan, CAL, versao=1)
            k2, _ = chave(scan, CAL, versao=2)
            k3, _ = chave(scan, {"0": dict(K=[101, 100, 8, 8], erro_preview_mm=0.0)}, versao=1)
            self.assertEqual(len({k1, k2, k3}), 3)

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_entrada_corrompida_e_refeita(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = root / "scan"
            criar_scan(scan)
            p1, _, _ = obter(scan, root / "cache", CAL)
            meta = json.loads((p1 / "observacoes.json").read_text())
            meta["observacoes"] = 1
            (p1 / "observacoes.json").write_text(json.dumps(meta))
            p2, m2, reuso = obter(scan, root / "cache", CAL)
            self.assertFalse(reuso)
            self.assertEqual(m2["observacoes"], 5 * 256)

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_vincular_mantem_formato_de_origem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = root / "scan"
            criar_scan(scan)
            p, _, _ = obter(scan, root / "cache", CAL)
            vincular(p, root / "fusao/origem")
            a = np.load(root / "fusao/origem/observacoes.npy")
            b = np.load(p / "observacoes.npy")
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
