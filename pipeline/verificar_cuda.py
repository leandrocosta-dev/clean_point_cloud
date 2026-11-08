"""Verifica runtime CUDA, compilacao de kernel e cuBLAS sem alterar scans.

python pipeline/verificar_cuda.py
"""

import json

import gpu


def main():
    cp = gpu.xp()  # configura CUDA_PATH (runtime pip) e levanta ErroGPU se nao houver CUDA/CuPy
    import numpy as np
    import open3d as o3d

    a = cp.arange(256, dtype=cp.float64).reshape(16, 16)
    # Kernel JIT (NVRTC), reducao e produto matricial (cuBLAS).
    b = cp.asnumpy((a @ a.T) + cp.sin(a))
    x = np.arange(256, dtype=np.float64).reshape(16, 16)
    erro = float(np.max(np.abs(b - ((x @ x.T) + np.sin(x)))))
    np.testing.assert_allclose(b, (x @ x.T) + np.sin(x), rtol=1e-12)
    # camada de vizinhanca (kernels proprios): vizinho mais proximo de uma grade regular contra ela mesma deve dar distancia 0
    g = np.stack(np.meshgrid(*[np.arange(8.0)] * 3), -1).reshape(-1, 3)
    d, j = gpu.vizinho_mais_proximo(g, g, dist_max=0.5)
    assert float(np.abs(d).max()) == 0.0 and (j == np.arange(len(g))).all(), "vizinho_mais_proximo falhou"
    props = cp.cuda.runtime.getDeviceProperties(0)
    nome = props["name"]
    if isinstance(nome, bytes):
        nome = nome.decode()
    print(
        json.dumps(
            dict(
                cupy=cp.__version__,
                cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                gpu=nome,
                memoria_total_bytes=props["totalGlobalMem"],
                kernel_e_cublas="ok",
                vizinhanca_gpu="ok",
                erro_max=erro,
                open3d=o3d.__version__,
                open3d_cuda=o3d.core.cuda.is_available(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
