"""Mede ruído temporal separadamente de curvatura/detalhe persistente (sintético, sem projeto real)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gpu
import numpy as np

from tests.gpu_utils import CUDA_DISPONIVEL


@unittest.skipUnless(CUDA_DISPONIVEL, "CUDA/CuPy ausente")
class MedicoesQualidadeTest(unittest.TestCase):
    def _superficie(self, jitter=False, relevo=True):
        import medicoes_qualidade_3d as mq

        x, y = np.meshgrid(np.linspace(-1, 1, 25), np.linspace(-1, 1, 25))
        z = 0.25 * np.exp(-((x**2 + y**2) / 0.18)) if relevo else np.zeros_like(x)
        p = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
        vis = np.tile([0.0, 0.0, 1.0], (len(p), 1))
        repeticoes, frames = [], []
        deslocamentos = [-0.14, -0.04, 0.05, 0.13] if jitter else [0.0] * 4
        for frame, dz in enumerate(deslocamentos):
            repeticoes.append(p + [0, 0, dz])
            frames.append(np.full(len(p), frame, dtype=np.int32))
        raw = np.concatenate(repeticoes)
        raw_v = np.tile(vis, (4, 1))
        raw_frame = np.concatenate(frames)
        xp = gpu.xp()
        args = [gpu.para_gpu(a, xp.float64) for a in (p, vis, raw, raw_v)]
        p_original_gpu = args[0].copy()
        normal, ruido, espacial, confiavel, consistente, evidencia, detalhe, suporte, suspeito = (
            mq._medir_normais_ruido(
                *args,
                gpu.para_gpu(raw_frame, xp.int32),
                0.35,
            )
        )
        mask = gpu.para_cpu(confiavel)
        return {
            "ruido": gpu.para_cpu(ruido)[mask],
            "pontos": gpu.para_cpu(args[0])[mask],
            "tangential_delta": gpu.para_cpu(args[0][:, :2] - p_original_gpu[:, :2])[mask],
            "espacial": gpu.para_cpu(espacial)[mask],
            "consistente": gpu.para_cpu(consistente)[mask],
            "evidencia": gpu.para_cpu(evidencia)[mask],
            "quadros": gpu.para_cpu(suporte)[mask],
            "detalhe": gpu.para_cpu(detalhe)[mask],
            "suspeito": gpu.para_cpu(suspeito)[mask],
            "normal": gpu.para_cpu(normal)[mask],
        }

    def test_relevo_repetido_e_detalhe_nao_vira_ruido_temporal(self):
        m = self._superficie(jitter=False)
        self.assertGreater(len(m["ruido"]), 0)
        self.assertLess(float(np.nanmedian(m["ruido"])), 0.01)
        self.assertTrue(np.mean(m["consistente"]) > 0.8)
        np.testing.assert_allclose(m["tangential_delta"], 0, atol=1e-12)

    def test_jitter_independente_por_quadro_eleva_ruido_temporal(self):
        m = self._superficie(jitter=True, relevo=False)
        self.assertGreater(float(np.nanmedian(m["ruido"])), 0.05)
        self.assertTrue(np.mean(m["evidencia"]) > 0.8)
        self.assertTrue(np.mean(m["suspeito"]) > 0.5)

    def test_preparo_e_seletor_removem_jitter_mas_preservam_relevo_sem_ref(self):
        import medicoes_qualidade_3d as mq
        import selecao_qualidade as sq

        x, y = np.meshgrid(np.linspace(-1, 1, 51), np.linspace(-1, 1, 51))
        relevo = 0.08 * np.exp(-((x**2 + y**2) / 0.0016))
        base = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        p_ruidoso = base.copy()
        p_ruidoso[:, 2] = relevo.ravel()
        vis = np.tile([0.0, 0.0, 1.0], (len(base), 1)).astype(np.float32)
        cache_por_guid = {}
        root = Path(tempfile.mkdtemp(prefix="medicoes_qualidade_fixture_"))
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        for guid, superficie, jitter in (("ruidoso", p_ruidoso, True), ("referencia", base, False)):
            scan = root / "data" / guid
            (scan / "cache").mkdir(parents=True)
            (scan / "cache" / "frames.dataset").write_bytes(b"dataset_fixture")
            (scan / "raw_preview.ply").write_bytes(b"preview_fixture")
            pasta = root / "cache" / guid
            pasta.mkdir(parents=True)
            (pasta / "observacoes.json").write_text("{}", encoding="utf-8")
            deslocamentos = [-0.14, -0.04, 0.05, 0.13] if jitter else [0.0] * 4
            arrays = []
            frames_meta = []
            for frame, dz in enumerate(deslocamentos):
                a = len(arrays) * len(base)
                atual = superficie.copy()
                atual[:, 2] += dz
                arrays.append(atual)
                frames_meta.append({"indice": frame, "inicio": a, "fim": a + len(base)})
            raw_p = np.concatenate(arrays)
            obs = np.empty(len(raw_p), dtype=[("p", "<f4", 3), ("v", "<f4", 3), ("quadro", "<i4"), ("pixel", "<i4")])
            obs["p"] = raw_p
            obs["v"] = np.tile(vis, (4, 1))
            obs["quadro"] = np.repeat(np.arange(4, dtype=np.int32), len(base))
            obs["pixel"] = np.tile(np.arange(len(base), dtype=np.int32), 4)
            np.save(pasta / "observacoes.npy", obs)
            cache_por_guid[guid] = (pasta, {"quadros": frames_meta, "calibracao": {}})

        def obter_fake(scan, cache, calibracao):
            return cache_por_guid[Path(scan).name][0], cache_por_guid[Path(scan).name][1], True

        poses = {g: np.eye(4) for g in cache_por_guid}
        with patch.object(mq, "obter", side_effect=obter_fake):
            preparado = mq.preparar_medicoes(root, poses, voxel=0.2, raio=0.35)
        for g in poses:
            self.assertTrue(np.mean(gpu.para_cpu(preparado["medicoes"][g]["evidencia_temporal"])) > 0.8)
        noisy = preparado["medicoes"]["ruidoso"]
        self.assertFalse(bool(gpu.para_cpu(noisy["consistencia_temporal"]).mean() > 0.8))
        selecionado = sq.selecionar_regioes(preparado["medicoes"])
        remover = gpu.para_cpu(selecionado["remover"]["ruidoso"])
        self.assertGreater(int(remover.sum()), 0)
        pontos_mensurados = gpu.para_cpu(noisy["pontos"])
        centro_relevo = np.linalg.norm(pontos_mensurados[:, :2], axis=1) < 0.12
        self.assertTrue(bool(centro_relevo.any()))
        detalhe = gpu.para_cpu(noisy["detalhe_protegido"])
        self.assertTrue(bool(detalhe[centro_relevo].any()))
        self.assertFalse(bool(remover[centro_relevo].any()))


if __name__ == "__main__":
    unittest.main()
