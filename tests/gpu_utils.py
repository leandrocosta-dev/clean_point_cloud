"""Detecta CUDA para os testes sem esconder erros numericos ou de inicializacao do pipeline."""

from functools import lru_cache


@lru_cache(maxsize=1)
def cuda_disponivel() -> bool:
    """Pula somente quando CuPy, uma placa ou o driver CUDA estao ausentes.

    Nao chama/captura gpu.xp: uma falha NVRTC, de kernel ou calculo continua sendo erro no teste.
    """
    try:
        import cupy as cp
    except ModuleNotFoundError as exc:
        if exc.name == "cupy":
            return False
        raise
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status in (35, 100):  # cudaErrorInsufficientDriver / cudaErrorNoDevice
            return False
        raise


CUDA_DISPONIVEL = cuda_disponivel()
