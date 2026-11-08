"""Menor autovalor/autovetor de matrizes simetricas 3x3 em lote, em forma fechada (CuPy), sem `linalg.eigh`.

    import pca3_gpu
    lam, v = pca3_gpu.menor_autopar(M)      # M (n, 3, 3); lam (n,) = menor autovalor, v (n, 3) = autovetor unitario

`cupy.linalg.eigh` em lote e lento (~1,5 us por matriz, 2 s por milhao); a forma fechada (Smith 1961 para os autovalores,
produto vetorial das linhas de M - lam*I para o autovetor) leva milissegundos e, em covariancias de planos (menor
autovalor muito menor que os outros), concorda com `eigh` ate ~1e-9 no autovetor (testado em tests/test_gpu_fusao.py).
Matriz quase isotropica (autovalores iguais) nao tem normal definida: devolve (0, 0, 1). O sinal do autovetor e arbitrario.
"""

import gpu


def menor_autopar(M):
    m = gpu.xp()
    M = gpu.para_gpu(M, m.float64)
    a00, a11, a22 = M[:, 0, 0], M[:, 1, 1], M[:, 2, 2]
    a01, a02, a12 = M[:, 0, 1], M[:, 0, 2], M[:, 1, 2]
    q = (a00 + a11 + a22) / 3.0
    b0, b1, b2 = a00 - q, a11 - q, a22 - q
    p2 = b0 * b0 + b1 * b1 + b2 * b2 + 2.0 * (a01 * a01 + a02 * a02 + a12 * a12)
    p = m.sqrt(p2 / 6.0)
    pp = m.maximum(p, 1e-300)
    c00, c11, c22, c01, c02, c12 = b0 / pp, b1 / pp, b2 / pp, a01 / pp, a02 / pp, a12 / pp
    det = c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) + c02 * (c01 * c12 - c11 * c02)
    r = m.clip(det / 2.0, -1.0, 1.0)
    phi = m.arccos(r) / 3.0
    lam = q + 2.0 * p * m.cos(phi + 2.0 * m.pi / 3.0)  # menor autovalor
    # autovetor: coluna de espaco nulo de (M - lam I) = melhor produto vetorial entre duas linhas
    r0 = m.stack([a00 - lam, a01, a02], axis=1)
    r1 = m.stack([a01, a11 - lam, a12], axis=1)
    r2 = m.stack([a02, a12, a22 - lam], axis=1)
    c = m.stack([m.cross(r0, r1), m.cross(r0, r2), m.cross(r1, r2)], axis=1)  # (n, 3, 3)
    nc = m.linalg.norm(c, axis=2)
    k = nc.argmax(1)
    v = c[m.arange(len(c)), k]
    nv = nc[m.arange(len(c)), k]
    escala = m.maximum(m.abs(M).max(axis=(1, 2)), 1e-300)
    degenerado = nv <= 1e-12 * escala * escala
    v = v / m.maximum(nv, 1e-300)[:, None]
    v = m.where(degenerado[:, None], m.asarray([0.0, 0.0, 1.0]), v)
    return lam, v
