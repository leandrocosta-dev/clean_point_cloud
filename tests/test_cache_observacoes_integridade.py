"""Integridade de cache: hashes, componentes de calibração e mmap Windows."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from cache_observacoes import _rmtree_dentro, _valida, chave, obter
from comum import sha256

from tests.gpu_utils import CUDA_DISPONIVEL
from tests.test_precisao import criar_scan

CAL = {"0": dict(K=[100, 100, 8, 8], erro_preview_mm=0.0)}


class CacheObservacoesIntegridadeTest(unittest.TestCase):
    def test_rmtree_nao_sai_da_raiz(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            fora = Path(tmp) / "fora"
            root.mkdir()
            fora.mkdir()
            (fora / "manter.txt").write_text("nao apagar", encoding="utf-8")
            with self.assertRaises(ValueError):
                _rmtree_dentro(root, fora)
            self.assertTrue((fora / "manter.txt").is_file())

    def test_param_bin_altera_chave_auto_e_fixo_nao_depende_dele(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan = Path(tmp) / "scan"
            criar_scan(scan)
            k_vazio, c_vazio = chave(scan)
            self.assertEqual(c_vazio["parametros_sha256"], {})
            param = scan / "param"
            param.mkdir()
            arquivo = param / "Pl.bin"
            arquivo.write_bytes(b"param-1")
            k1, c1 = chave(scan)
            self.assertEqual(c1["parametros_sha256"], {"param/Pl.bin": sha256(arquivo)})
            arquivo.write_bytes(b"param-2")
            k2, _ = chave(scan)
            self.assertNotEqual(k1, k2)
            self.assertNotEqual(k_vazio, k1)
            kf1, _ = chave(scan, CAL)
            arquivo.write_bytes(b"param-3")
            kf2, _ = chave(scan, CAL)
            self.assertEqual(kf1, kf2)

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_entrada_legada_sem_sha_refaz_e_fecha_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = root / "scan"
            criar_scan(scan)
            original = sha256(scan / "cache" / "frames.dataset")
            pasta, meta, reused = obter(scan, root / "cache", CAL)
            self.assertFalse(reused)
            self.assertIn("arquivo_observacoes_sha256", meta)
            self.assertFalse(isinstance(meta, np.memmap))
            old = json.loads((pasta / "observacoes.json").read_text(encoding="utf-8"))
            old.pop("arquivo_observacoes_sha256")
            old.pop("versao_cache")
            old.pop("componentes_cache")
            (pasta / "observacoes.json").write_text(json.dumps(old), encoding="utf-8")
            pasta2, meta2, reused2 = obter(scan, root / "cache", CAL)
            self.assertEqual(pasta2, pasta)
            self.assertFalse(reused2)
            self.assertEqual(meta2["arquivo_observacoes_sha256"], sha256(pasta2 / "observacoes.npy"))
            self.assertEqual(sha256(scan / "cache" / "frames.dataset"), original)
            self.assertTrue((pasta2 / "observacoes.npy").is_file())  # rename/rmtree funcionou com mmap fechado

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_corrupcao_de_um_byte_mesma_forma_e_refeita(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = root / "scan"
            criar_scan(scan)
            original = sha256(scan / "cache" / "frames.dataset")
            pasta, _, _ = obter(scan, root / "cache", CAL)
            arq = pasta / "observacoes.npy"
            obs = np.load(arq, mmap_mode="r", allow_pickle=False)
            shape, dtype = obs.shape, obs.dtype
            obs._mmap.close()
            with arq.open("r+b") as f:
                f.seek(-1, 2)
                byte = f.read(1)
                f.seek(-1, 2)
                f.write(bytes([byte[0] ^ 1]))
            damaged = np.load(arq, mmap_mode="r", allow_pickle=False)
            self.assertEqual((damaged.shape, damaged.dtype), (shape, dtype))
            damaged._mmap.close()
            good, meta, reused = obter(scan, root / "cache", CAL)
            self.assertFalse(reused)
            self.assertEqual(meta["arquivo_observacoes_sha256"], sha256(good / "observacoes.npy"))
            self.assertEqual(sha256(scan / "cache" / "frames.dataset"), original)

    @unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
    def test_validacao_rejeita_hash_de_componentes_diferente(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scan = root / "scan"
            criar_scan(scan)
            pasta, _, _ = obter(scan, root / "cache", CAL)
            _, comp = chave(scan, CAL)
            comp_alterado = dict(comp, parametros_sha256={"param/Pl.bin": "0" * 64})
            valido, retorno = _valida(pasta, comp_alterado)
            self.assertFalse(valido)
            self.assertIsNone(retorno)


if __name__ == "__main__":
    unittest.main()
