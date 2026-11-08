"""Verifica runtime CUDA, compilacao de kernel e cuBLAS sem alterar scans.

    python pipeline/verificar_cuda.py
"""
import json
import os
from pathlib import Path
import sys


def main():
    runtime = Path(sys.prefix) / 'Lib/site-packages/nvidia/cuda_runtime'
    if runtime.exists() and not Path(os.environ.get('CUDA_PATH', '__ausente__')).exists():
        os.environ['CUDA_PATH'] = str(runtime)
    import cupy as cp
    import numpy as np
    import open3d as o3d

    a = cp.arange(256, dtype=cp.float64).reshape(16, 16)
    # Kernel JIT (NVRTC), reducao e produto matricial (cuBLAS).
    b = cp.asnumpy((a @ a.T) + cp.sin(a))
    x = np.arange(256, dtype=np.float64).reshape(16, 16)
    erro = float(np.max(np.abs(b - ((x @ x.T) + np.sin(x)))))
    np.testing.assert_allclose(b, (x @ x.T) + np.sin(x), rtol=1e-12)
    props = cp.cuda.runtime.getDeviceProperties(0)
    nome = props['name']
    if isinstance(nome, bytes):
        nome = nome.decode()
    print(json.dumps(dict(cupy=cp.__version__, cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                          gpu=nome, memoria_total_bytes=props['totalGlobalMem'],
                          kernel_e_cublas='ok', erro_max=erro,
                          open3d=o3d.__version__, open3d_cuda=o3d.core.cuda.is_available()), indent=2))


if __name__ == '__main__':
    main()
