"""A deteccao de CUDA pode pular hardware ausente, mas nao erros reais do runtime."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tests import gpu_utils


class ErroCudaFixture(RuntimeError):
    def __init__(self, status: int):
        self.status = status
        super().__init__(str(status))


class DisponibilidadeCudaTest(unittest.TestCase):
    def setUp(self):
        gpu_utils.cuda_disponivel.cache_clear()
        self.addCleanup(gpu_utils.cuda_disponivel.cache_clear)

    def cupy_falso(self, **kwargs):
        runtime = SimpleNamespace(getDeviceCount=Mock(**kwargs), CUDARuntimeError=ErroCudaFixture)
        return SimpleNamespace(cuda=SimpleNamespace(runtime=runtime))

    def test_modulo_ausente_ou_zero_dispositivos(self):
        with patch.dict("sys.modules", {"cupy": None}):
            self.assertFalse(gpu_utils.cuda_disponivel())
        gpu_utils.cuda_disponivel.cache_clear()
        with patch.dict("sys.modules", {"cupy": self.cupy_falso(return_value=0)}):
            self.assertFalse(gpu_utils.cuda_disponivel())

    def test_driver_ou_dispositivo_ausente(self):
        for status in (35, 100):
            gpu_utils.cuda_disponivel.cache_clear()
            with self.subTest(status=status):
                with patch.dict("sys.modules", {"cupy": self.cupy_falso(side_effect=ErroCudaFixture(status))}):
                    self.assertFalse(gpu_utils.cuda_disponivel())

    def test_runtime_quebrado_nao_vira_skip(self):
        for erro in (ErroCudaFixture(700), RuntimeError("runtime quebrado")):
            gpu_utils.cuda_disponivel.cache_clear()
            with self.subTest(erro=erro):
                with patch.dict("sys.modules", {"cupy": self.cupy_falso(side_effect=erro)}):
                    with self.assertRaises(type(erro)) as cm:
                        gpu_utils.cuda_disponivel()
                    self.assertIs(cm.exception, erro)


if __name__ == "__main__":
    unittest.main()
