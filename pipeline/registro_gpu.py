"""Registro de nuvens na GPU (substitui voxel_down_sample, estimate_normals, registration_icp, evaluate_registration
e get_information_matrix do Open3D, e os cKDTree dos scripts de alinhamento). Usa so a camada `gpu.py`.

    from registro_gpu import Nuvem, nuvem_voxel, icp, avaliar_encaixe, matriz_informacao, residuo_plano
    pa = nuvem_voxel(P, 1.0)                         # centroides por voxel + normais PCA (raio = 3 x voxel), cupy float64
    T, info = icp(pb, pa, T0, dist_max=2.0, iteracoes=60)         # ponto-plano, T leva pb -> pa
    fit, rmse, n = avaliar_encaixe(pb, pa, 0.3, T)   # = evaluate_registration (fracao, RMS ponto-a-ponto, pares)
    H = matriz_informacao(pb, pa, 1.0, T)            # 6x6 para as arestas do PoseGraph (CPU, Open3D)

Diferencas medidas para o Open3D (ver tests/test_gpu_alinhamento.py): normais por TODOS os pontos do raio (o Open3D
usa no maximo `max_nn`) e orientadas para o centroide da nuvem (o sinal do Open3D e arbitrario; o vies por celula usa
o sinal). Voxel com origem min - voxel/2, como o Open3D.
"""

import gpu
import numpy as np


class Nuvem:
    """Pontos (n, 3) float64 e normais (n, 3) opcionais, em cupy (numpy so no modo REVO_PERMITIR_CPU)."""

    __slots__ = ("pts", "normais")

    def __init__(self, pts, normais=None):
        self.pts = pts
        self.normais = normais

    def __len__(self):
        return len(self.pts)

    def transformada(self, T):
        xp = gpu.xp()
        R, t = xp.asarray(T[:3, :3]), xp.asarray(T[:3, 3])
        n = None if self.normais is None else self.normais @ R.T
        return Nuvem(self.pts @ R.T + t, n)

    def amostra(self, n, semente=0):
        """No maximo n pontos escolhidos ao acaso (mesmas normais)."""
        if len(self) <= n:
            return self
        idx = gpu.para_gpu(np.random.default_rng(semente).choice(len(self), n, replace=False))
        return Nuvem(self.pts[idx], None if self.normais is None else self.normais[idx])

    def para_o3d(self):
        """PointCloud do Open3D (so para FPFH + RANSAC, que nao tem equivalente na GPU)."""
        import open3d as o3d

        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gpu.para_cpu(self.pts)))
        if self.normais is not None:
            p.normals = o3d.utility.Vector3dVector(gpu.para_cpu(self.normais))
        return p


def transformar(P, T):
    """P @ R.T + t na GPU; devolve no mesmo lado da entrada (numpy -> numpy)."""
    xp = gpu.xp()
    saida_gpu = gpu._eh_gpu(P)
    Pg = gpu.para_gpu(P, xp.float64)
    out = Pg @ xp.asarray(np.asarray(T)[:3, :3].T) + xp.asarray(np.asarray(T)[:3, 3])
    return out if saida_gpu else gpu.para_cpu(out)


def normais_nuvem(C, raio):
    """Normais PCA dos pontos a <= raio, orientadas para o centroide. Devolve (normais, validos)."""
    xp = gpu.xp()
    N = gpu.normais_pca(C, raio=raio, orientar_para=C.mean(0))
    return N, xp.isfinite(N[:, 0])


def nuvem_voxel(P, voxel, normais=True, raio_normal=None):
    """Centroides de voxel (origem min - voxel/2, como o Open3D) e, se `normais`, normais PCA no raio `raio_normal`
    (padrao 3 x voxel); pontos sem normal (< 3 vizinhos) saem da nuvem."""
    xp = gpu.xp()
    Pg = gpu.para_gpu(P, xp.float64)
    if len(Pg) == 0:
        return Nuvem(Pg, Pg.copy() if normais else None)
    C, _ = gpu.reduzir_voxel(Pg, voxel, origem=Pg.min(0) - 0.5 * voxel)
    if not normais:
        return Nuvem(C)
    N, ok = normais_nuvem(C, raio_normal if raio_normal is not None else 3.0 * voxel)
    return Nuvem(C[ok], N[ok])


def avaliar_encaixe(src, tgt, dist_max, T=None):
    """Como `evaluate_registration`: (fitness, rmse, pares). fitness = fracao dos pontos de `src` (levados por T) com
    vizinho em `tgt` a <= dist_max; rmse = RMS dessas distancias (ponto a ponto)."""
    xp = gpu.xp()
    S = src.pts if isinstance(src, Nuvem) else gpu.para_gpu(src, xp.float64)
    A = tgt.pts if isinstance(tgt, Nuvem) else gpu.para_gpu(tgt, xp.float64)
    if T is not None:
        S = transformar(S, T)
    d, j = gpu.vizinho_mais_proximo(A, S, dist_max=dist_max)
    ok = j >= 0
    n = int(ok.sum())
    if n == 0:
        return 0.0, 0.0, 0
    return n / max(len(S), 1), float(xp.sqrt((d[ok] ** 2).mean())), n


def fracao_proxima(alvo, consultas, dist_max):
    """Fracao das `consultas` com algum ponto de `alvo` a <= dist_max (cobertura, sobreposicao)."""
    A = alvo.pts if isinstance(alvo, Nuvem) else alvo
    Q = consultas.pts if isinstance(consultas, Nuvem) else consultas
    if len(Q) == 0 or len(A) == 0:
        return 0.0
    _, j = gpu.vizinho_mais_proximo(A, Q, dist_max=dist_max)
    return float((j >= 0).mean())


def icp(src, tgt, T0, dist_max, iteracoes, robusto=None):
    """ICP ponto-plano de `src` em `tgt` (precisa de normais) com a pose inicial T0. `robusto` = None (minimos
    quadrados, como o Open3D padrao) ou c em mm (Tukey, como TukeyLoss(k=c)). Devolve (T, info de gpu.icp_ponto_plano)."""
    if tgt.normais is None:
        raise ValueError("o alvo do ICP precisa de normais")
    perda, c = ("l2", None) if robusto is None else ("tukey", float(robusto))
    return gpu.icp_ponto_plano(
        src.pts, tgt.pts, tgt.normais, T0=T0, dist_max=dist_max, iteracoes=iteracoes, perda=perda, c=c, tol=1e-6
    )


def matriz_informacao(src, tgt, dist_max, T):
    """Matriz 6x6 (numpy) de `get_information_matrix_from_point_clouds`: soma de G^T G sobre os pares a <= dist_max,
    com G = [-[q]x | I] nos pontos q do alvo (rotacao primeiro, translacao depois)."""
    xp = gpu.xp()
    p = transformar(src.pts, T)
    _, j = gpu.vizinho_mais_proximo(tgt.pts, p, dist_max=dist_max)
    q = tgt.pts[j[j >= 0]]
    H = xp.zeros((6, 6), xp.float64)
    if len(q):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        G = xp.zeros((len(q), 3, 6), xp.float64)
        G[:, 0, 1], G[:, 0, 2], G[:, 1, 0], G[:, 1, 2], G[:, 2, 0], G[:, 2, 1] = z, -y, -z, x, y, -x
        G[:, 0, 3] = G[:, 1, 4] = G[:, 2, 5] = 1.0
        H = xp.einsum("nai,naj->ij", G, G)
    return gpu.para_cpu(H)


def _mediana_por_grupo(r, inv, cnt):
    """Mediana de `r` em cada grupo `inv` (cnt = tamanho de cada grupo), tudo na GPU."""
    xp = gpu.xp()
    o1 = xp.argsort(r)
    o2 = xp.argsort(inv[o1], kind="stable")
    rs = r[o1[o2]]  # ordenado por grupo e, dentro dele, por valor
    ini = xp.cumsum(cnt) - cnt
    return 0.5 * (rs[ini + (cnt - 1) // 2] + rs[ini + cnt // 2])


def residuo_plano(pa, pb, T, tol=2.0, cel=30.0, min_n=100):
    """Medida de ALINHAMENTO insensivel ao ruido: (cobertura, vies mediano |.| por celula de `cel` mm, p90 dos vieses,
    mediana de |r|). r = distancia de cada ponto de pb (levado por T) ao plano local do vizinho em pa; a MEDIANA COM
    SINAL por celula so sobra se as nuvens estiverem deslocadas. Celulas na grade absoluta de `cel` mm."""
    xp = gpu.xp()
    B = transformar(pb.pts, T)
    d, idx = gpu.vizinho_mais_proximo(pa.pts, B, dist_max=tol)
    m = idx >= 0
    if int(m.sum()) < 200:
        return 0.0, None, None, None
    Bm, j = B[m], idx[m]
    r = ((Bm - pa.pts[j]) * pa.normais[j]).sum(1)
    cobertura = float(m.mean())
    mediana_abs = float(xp.median(xp.abs(r)))
    _, inv, cnt = gpu.agrupar(gpu.chaves_voxel(Bm, cel, xp.floor(Bm.min(0) / cel) * cel))
    med = xp.abs(_mediana_por_grupo(r, inv, cnt))
    vies = gpu.para_cpu(med[cnt >= min_n])
    if not len(vies):
        return cobertura, None, None, mediana_abs
    return cobertura, float(np.median(vies)), float(np.percentile(vies, 90)), mediana_abs
