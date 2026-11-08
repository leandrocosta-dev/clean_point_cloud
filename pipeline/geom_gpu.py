"""Operacoes geometricas na GPU que faltavam na camada `gpu.py` (substituem chamadas do Open3D/scipy nos scripts de
formatos antigos, Creality e utilitarios): normais "hibridas", DBSCAN, componentes de malha, amostragem de malha,
distancia nuvem-nuvem, ICP encadeado e filtros de imagem/grade 3D (cupyx.scipy.ndimage).

    import geom_gpu as gg
    N = gg.normais_hibrido(P, raio=2.4, max_nn=30)        # como KDTreeSearchParamHybrid(radius, max_nn)
    mantem = gg.dbscan_maior(P, eps=0.8, min_pts=8)       # mascara do maior cluster DBSCAN
    rot, tam = gg.componentes_triangulos(F, len(V))       # como cluster_connected_triangles()
    pts, nor = gg.amostrar_malha(V, F, 1_000_000)         # como sample_points_uniformly(use_triangle_normal=True)
    d = gg.distancia_ate(alvo, consultas)                 # como compute_point_cloud_distance

Tudo aceita numpy ou cupy e devolve numpy (use `saida_gpu=True` onde existir para encadear sem voltar a CPU).
Fica na CPU (sem equivalente na GPU): Open3D FPFH / RANSAC por features, TSDF (VoxelBlockGrid), Poisson, edicao de
topologia de malha do Open3D (remove_*_by_mask) e MeshLib; os calculos numericos ao redor passam por aqui e por `gpu.py`.
"""

import gpu
import numpy as np


def _para_saida(x, saida_gpu):
    return gpu.para_gpu(x) if saida_gpu else gpu.para_cpu(x)


# ---------------------------------------------------------------- normais
def normais_hibrido(P, raio, max_nn=30, orientar_para=None, saida_gpu=False):
    """Normais PCA com os `max_nn` vizinhos mais proximos DENTRO de `raio` (como KDTreeSearchParamHybrid do Open3D).
    Vizinhanca com < 3 pontos: normal NaN. `orientar_para`: ver gpu.normais_pca."""
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    N = m.full((len(Pg), 3), m.nan, m.float64)
    k = int(max_nn)
    for a, b in gpu.blocos(len(Pg), bytes_por_item=k * 16 + 400):
        d, j = gpu.knn(Pg, Pg[a:b], k, raio_max=raio)
        valido = j >= 0
        Y = Pg[m.maximum(j, 0)]
        cont = valido.sum(1)
        w = valido[:, :, None].astype(m.float64)
        med = (Y * w).sum(1) / m.maximum(cont, 1)[:, None]
        Y = (Y - med[:, None, :]) * w
        M = m.einsum("nki,nkj->nij", Y, Y)
        ok = cont >= 3
        _, v = m.linalg.eigh(m.where(ok[:, None, None], M, m.eye(3)))
        N[a:b] = m.where(ok[:, None], v[:, :, 0], m.nan)
    if orientar_para is not None:
        o = gpu.para_gpu(orientar_para, m.float64)
        direcao = (o - Pg) if o.ndim == 1 else o
        N = m.where(((N * direcao).sum(1) < 0)[:, None], -N, N)
    return _para_saida(N, saida_gpu)


# ---------------------------------------------------------------- componentes
def _componentes_arestas(n, a, b):
    """Rotulo (menor indice do componente) de cada um de `n` nos, dadas as arestas (a, b); tudo em cupy.
    Propagacao do menor rotulo com salto de ponteiros (cada passada reduz o diametro; converge em O(log) tipico)."""
    m = gpu.xp()
    lab = m.arange(n, dtype=m.int64)
    if len(a) == 0:
        return lab
    for _ in range(10000):
        la, lb = lab[a], lab[b]
        menor = m.minimum(la, lb)
        novo = lab.copy()
        m.minimum.at(novo, la, menor)  # liga raizes (hooking)
        m.minimum.at(novo, lb, menor)
        for _ in range(8):  # salto de ponteiros
            novo = novo[novo]
        if bool((novo == lab).all()):
            break
        lab = novo
    return lab


def rotulos_por_arestas(n, a, b):
    """Rotulo (menor indice do componente, cupy int64) de cada um de `n` nos dadas as arestas (a, b) (numpy ou cupy)."""
    m = gpu.xp()
    return _componentes_arestas(int(n), gpu.para_gpu(a, m.int64), gpu.para_gpu(b, m.int64))


def _compactar_rotulos(raiz):
    """(rotulos 0..c-1 ordenados do maior para o menor componente, tamanhos)."""
    m = gpu.xp()
    _, inv = m.unique(raiz, return_inverse=True)
    inv = inv.reshape(-1)
    cont = m.bincount(inv)
    ordem = m.argsort(-cont, kind="stable")
    posto = m.empty(len(ordem), m.int64)
    posto[ordem] = m.arange(len(ordem))
    return posto[inv], cont[ordem]


def componentes_triangulos(F, n_vertices, saida_gpu=False):
    """(rotulo por triangulo, tamanhos): componentes de triangulos que compartilham aresta (como
    `cluster_connected_triangles`). Rotulo 0 = maior componente."""
    m = gpu.xp()
    F = gpu.para_gpu(F, m.int64)
    nt = len(F)
    if nt == 0:
        return _para_saida(m.zeros(0, m.int64), saida_gpu), _para_saida(m.zeros(0, m.int64), saida_gpu)
    e = m.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = m.sort(e, axis=1)
    chave = e[:, 0] * int(n_vertices) + e[:, 1]
    tri = m.tile(m.arange(nt, dtype=m.int64), 3)
    ordem = m.argsort(chave, kind="stable")
    ks, ts = chave[ordem], tri[ordem]
    igual = ks[1:] == ks[:-1]
    a, b = ts[:-1][igual], ts[1:][igual]
    rot, tam = _compactar_rotulos(_componentes_arestas(nt, a, b))
    return _para_saida(rot, saida_gpu), _para_saida(tam, saida_gpu)


def remover_triangulos_fora_do_maior(F, n_vertices, frac_min=None, min_tri=None):
    """Mascara bool (True = triangulo mantido): so componentes com >= frac_min * maior e/ou >= min_tri triangulos."""
    rot, tam = componentes_triangulos(F, n_vertices, saida_gpu=True)
    m = gpu.xp()
    if len(tam) == 0:
        return gpu.para_cpu(m.zeros(0, bool))
    ok = m.ones(len(tam), bool)
    if frac_min is not None:
        ok &= tam >= frac_min * tam.max()
    if min_tri is not None:
        ok &= tam >= min_tri
    return gpu.para_cpu(ok[rot])


def dbscan(P, eps, min_pts=10, max_viz=64):
    """Rotulos DBSCAN (n,) int64 numpy, -1 = ruido. Nucleo = >= min_pts pontos a <= eps (inclui o proprio, como Open3D).
    Nucleos ligados por vizinhanca eps (grafo dos `max_viz` mais proximos de cada nucleo dentro de eps: em nuvens de
    superficie o grafo kNN conecta o mesmo que o grafo de raio); bordas ficam com o cluster do nucleo mais proximo.
    Rotulo 0 = maior cluster."""
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    n = len(Pg)
    cont = gpu.contar_no_raio(Pg, Pg, eps)
    nucleo = cont >= int(min_pts)
    rotulo = m.full(n, -1, m.int64)
    idc = m.nonzero(nucleo)[0]
    nc = len(idc)
    if nc == 0:
        return gpu.para_cpu(rotulo)
    C = Pg[idc]
    k = int(min(max_viz, nc))
    J = m.empty((nc, k), m.int32)
    for a, b in gpu.blocos(nc, bytes_por_item=k * 16 + 400):
        d, j = gpu.knn(C, C[a:b], k, raio_max=eps)
        J[a:b] = j.astype(m.int32)
    ai = m.repeat(m.arange(nc, dtype=m.int64), k)
    bj = J.reshape(-1).astype(m.int64)
    ok = bj >= 0
    raiz = _componentes_arestas(nc, ai[ok], bj[ok])
    del J, ai, bj, ok
    rot_c, tam = _compactar_rotulos(raiz)
    rotulo[idc] = rot_c
    resto = m.nonzero(~nucleo)[0]
    if len(resto):
        d, j = gpu.vizinho_mais_proximo(C, Pg[resto], dist_max=float(eps))
        casou = j >= 0
        rotulo[resto[casou]] = rot_c[j[casou]]
    # ordena de novo pelo tamanho final (bordas contam), 0 = maior
    pos = rotulo >= 0
    if bool(pos.any()):
        cont_f = m.bincount(rotulo[pos])
        ordem = m.argsort(-cont_f, kind="stable")
        posto = m.empty(len(ordem), m.int64)
        posto[ordem] = m.arange(len(ordem))
        rotulo[pos] = posto[rotulo[pos]]
    return gpu.para_cpu(rotulo)


def dbscan_maior(P, eps, min_pts=10):
    """Mascara bool numpy do maior cluster DBSCAN (ver `dbscan`)."""
    rot = dbscan(P, eps, min_pts)
    return rot == 0


class AcumuladorVoxel:
    """Decimacao por voxel em fluxo (quadro a quadro, memoria limitada): soma coordenadas (e colunas extras) por voxel
    numa grade de origem FIXA, de modo que o resultado e identico a reduzir a nuvem inteira de uma vez.

        acc = AcumuladorVoxel(0.4)
        for quadro in quadros: acc.add(P_quadro, extra=centro_da_camera_repetido)
        C, cont, extra_medio = acc.resultado()      # centroides (m, 3), pontos por voxel, media das colunas extras

    A origem fixa (-2^20 * voxel) cobre +-2^20 voxels em cada eixo; ValueError se algum ponto sair disso."""

    def __init__(self, voxel, pendente_max=4_000_000):
        self.voxel = float(voxel)
        self.origem = np.full(3, -(2**20) * self.voxel)
        self.pendente_max = int(pendente_max)
        self._lotes, self._n = [], 0

    def add(self, P, extra=None):
        m = gpu.xp()
        Pg = gpu.para_gpu(P, m.float64)
        if not len(Pg):
            return
        cols = Pg if extra is None else m.concatenate([Pg, gpu.para_gpu(extra, m.float64)], axis=1)
        u, inv, cnt = gpu.agrupar(gpu.chaves_voxel(Pg, self.voxel, self.origem))
        S = gpu.somar_por_grupo(cols, inv, len(u))
        self._lotes.append((u, S, cnt))
        self._n += len(u)
        if self._n > self.pendente_max:
            self._consolidar()

    def _consolidar(self):
        m = gpu.xp()
        if len(self._lotes) <= 1:
            return
        k = m.concatenate([l[0] for l in self._lotes])
        S = m.concatenate([l[1] for l in self._lotes])
        c = m.concatenate([l[2] for l in self._lotes])
        u, inv, _ = gpu.agrupar(k)
        S = gpu.somar_por_grupo(S, inv, len(u))
        c = gpu.somar_por_grupo(c.astype(m.float64), inv, len(u)).astype(m.int64)
        self._lotes, self._n = [(u, S, c)], len(u)

    def resultado(self, saida_gpu=False):
        """(centroides (m, 3), contagens (m,), media das colunas extras (m, e) ou None), ordenados pela chave do voxel."""
        gpu.xp()
        self._consolidar()
        if not self._lotes:
            return None
        u, S, c = self._lotes[0]
        med = S / c[:, None]
        C, E = med[:, :3], (med[:, 3:] if med.shape[1] > 3 else None)
        return (_para_saida(C, saida_gpu), _para_saida(c, saida_gpu), None if E is None else _para_saida(E, saida_gpu))


# ---------------------------------------------------------------- amostragem e distancias
def alinhar_sinal(N, referencia, saida_gpu=False):
    """Inverte as normais N que apontam contra `referencia` (mesmo efeito de estimar normais num PointCloud que ja tem
    normais no Open3D); linhas NaN de N recebem a referencia."""
    m = gpu.xp()
    Ng = gpu.para_gpu(N, m.float64)
    R = gpu.para_gpu(referencia, m.float64)
    Ng = m.where(m.isnan(Ng[:, :1]), R, Ng)
    Ng = m.where(((Ng * R).sum(1) < 0)[:, None], -Ng, Ng)
    return _para_saida(Ng, saida_gpu)


def area_malha(V, F):
    """Area da superficie (float) da malha, na GPU."""
    m = gpu.xp()
    V = gpu.para_gpu(V, m.float64)
    F = gpu.para_gpu(F, m.int64)
    return float(0.5 * m.linalg.norm(m.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum())


def espalhamento_local(P, Q, raio):
    """(autovalores (nq, 3) crescentes da covariancia populacional dos pontos de P a <= raio de cada consulta, contagem (nq,))
    numpy. Soma exata (todos os vizinhos, nao so os k mais proximos) em coordenadas relativas a consulta (float32 seguro).
    sqrt(autovalor[0]) = desvio fora do plano local (rugosidade)."""
    m = gpu.xp()
    g = gpu.Grade(gpu.para_gpu(P, m.float64), raio)
    Qg = gpu.para_gpu(Q, m.float64)
    W = np.full((len(Qg), 3), np.nan)
    C = np.zeros(len(Qg), np.int64)
    for a, b in gpu.blocos(len(Qg), bytes_por_item=10 * 8 + 9 * 8 * 3):
        S = g.somas(Qg[a:b], gpu=True).astype(m.float64)
        s0 = S[:, 0]
        med = S[:, 1:4] / m.maximum(s0, 1e-30)[:, None]
        M = m.empty((b - a, 3, 3), m.float64)
        M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = S[:, 4], S[:, 5], S[:, 6]
        M[:, 0, 1] = M[:, 1, 0] = S[:, 7]
        M[:, 0, 2] = M[:, 2, 0] = S[:, 8]
        M[:, 1, 2] = M[:, 2, 1] = S[:, 9]
        M = M / m.maximum(s0, 1)[:, None, None] - med[:, :, None] * med[:, None, :]
        ok = s0 >= 1
        w = m.linalg.eigvalsh(m.where(ok[:, None, None], M, m.eye(3)))
        W[a:b] = gpu.para_cpu(m.where(ok[:, None], w, m.nan))
        C[a:b] = gpu.para_cpu(s0.astype(m.int64))
    return W, C


def centros_triangulos(V, F, saida_gpu=False):
    """Centroide (n, 3) de cada triangulo."""
    m = gpu.xp()
    return _para_saida(gpu.para_gpu(V, m.float64)[gpu.para_gpu(F, m.int64)].mean(1), saida_gpu)


def normais_triangulos(V, F, saida_gpu=False):
    """Normal unitaria (n, 3) de cada triangulo (regra da mao direita, como `compute_triangle_normals`)."""
    m = gpu.xp()
    V = gpu.para_gpu(V, m.float64)
    F = gpu.para_gpu(F, m.int64)
    c = m.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    return _para_saida(c / m.maximum(m.linalg.norm(c, axis=1, keepdims=True), 1e-300), saida_gpu)


def normais_vertices(V, F, saida_gpu=False):
    """Normais de vertice (media das normais dos triangulos ponderada pela area, normalizada), como `compute_vertex_normals`."""
    m = gpu.xp()
    V = gpu.para_gpu(V, m.float64)
    F = gpu.para_gpu(F, m.int64)
    c = m.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])  # modulo = 2 x area: ja pondera
    S = m.zeros((len(V), 3), m.float64)
    for k in range(3):
        for dim in range(3):
            S[:, dim] += m.bincount(F[:, k], weights=c[:, dim], minlength=len(V))
    n = S / m.maximum(m.linalg.norm(S, axis=1, keepdims=True), 1e-300)
    return _para_saida(n, saida_gpu)


def quantil(x, q):
    """Quantil(es) `q` (0..1) de x na GPU, como np.quantile; devolve float/numpy."""
    m = gpu.xp()
    r = m.quantile(gpu.para_gpu(x, m.float64), q)
    return float(r) if np.ndim(q) == 0 else gpu.para_cpu(r)


def campo_deriva(V, B, nB, celula=20.0, min_pts=30):
    """Campo SUAVE de correcao (deriva de baixa frequencia) de V em relacao a B: mediana por celula (`celula` mm) do
    deslocamento normal V->B (pares a < 4 mm), gaussiana (sigma 1 celula), preenchimento do vazio pelo vizinho mais proximo
    e interpolacao trilinear. Devolve f(X) -> deslocamento (n, 3) para pontos X (cupy) DENTRO da caixa de V (um
    ponto fora recebe o valor da borda). Tudo na GPU (cupyx.scipy.ndimage)."""
    m = gpu.xp()
    ndi = ndi_modulo()
    V = gpu.para_gpu(V, m.float64)
    B = gpu.para_gpu(B, m.float64)
    nB = gpu.para_gpu(nB, m.float64)
    d, i = gpu.vizinho_mais_proximo(B, V, dist_max=np.inf)
    ok = d < 4.0
    Vm, im = V[ok], i[ok]
    vec = ((B[im] - Vm) * nB[im]).sum(1)[:, None] * nB[im]
    lo = V.min(0) - celula
    n = tuple(int(x) for x in m.ceil((V.max(0) + celula - lo) / celula).astype(m.int64) + 1)
    idx = m.floor((Vm - lo) / celula).astype(m.int64)
    flat = (
        m.ravel_multi_index(tuple(idx.T), n)
        if hasattr(m, "ravel_multi_index")
        else (idx[:, 0] * n[1] + idx[:, 1]) * n[2] + idx[:, 2]
    )
    S = m.zeros((*n, 3))
    N = m.zeros(n)
    if len(flat):
        u, med, cont = mediana_por_grupo(flat, vec)
        u, med, cont = gpu.para_gpu(u), gpu.para_gpu(med), gpu.para_gpu(cont)
        boa = cont >= min_pts
        ijk = m.unravel_index(u[boa], n)
        S[ijk] = med[boa]
        N[ijk] = 1
    Ss = m.stack([ndi.gaussian_filter(S[..., c], 1.0) for c in range(3)], -1)
    Ns = ndi.gaussian_filter(N, 1.0)
    bom = Ns > 0.05
    if not bool(bom.any()):
        return lambda X: m.zeros_like(gpu.para_gpu(X, m.float64))
    F = m.zeros_like(Ss)
    F[bom] = Ss[bom] / Ns[bom][:, None]
    _, perto = ndi.distance_transform_edt(
        ~bom, return_indices=True
    )  # fora da sobreposicao: vizinho preenchido mais proximo
    F = F[tuple(perto)]

    def f(X):
        coords = ((gpu.para_gpu(X, m.float64) - lo) / celula).T
        return m.stack([ndi.map_coordinates(F[..., c], coords, order=1, mode="nearest") for c in range(3)], 1)

    return f


def reduzir_voxel_normais(P, N, voxel, saida_gpu=False):
    """Decimacao por voxel como `voxel_down_sample` do Open3D: (centroides (m, 3), normais medias normalizadas (m, 3))."""
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    Ng = gpu.para_gpu(N, m.float64)
    o = Pg.min(0)
    u, inv, cnt = gpu.agrupar(gpu.chaves_voxel(Pg, voxel, o))
    inv = gpu.para_gpu(inv)
    C = gpu.somar_por_grupo(Pg - o, inv, len(u)) / gpu.para_gpu(cnt)[:, None] + o
    S = gpu.somar_por_grupo(Ng, inv, len(u))
    S = S / m.maximum(m.linalg.norm(S, axis=1, keepdims=True), 1e-300)
    return _para_saida(C, saida_gpu), _para_saida(S, saida_gpu)


def ler_nuvem(caminho, voxel=None, raio_normais=None, max_nn=30):
    """Le um PLY/npy de nuvem (E/S na CPU) e devolve (P, N) numpy; `voxel` decima na GPU; sem normais no arquivo
    (ou NaN) estima por PCA hibrida com `raio_normais` (padrao 5 x voxel, ou 3 mm)."""
    caminho = str(caminho)
    if caminho.lower().endswith(".npy"):
        P, N = np.load(caminho).astype(np.float64), None
    else:
        import open3d as o3d

        pc = o3d.io.read_point_cloud(caminho)
        P = np.asarray(pc.points, np.float64)
        N = np.asarray(pc.normals, np.float64) if pc.has_normals() else None
    if voxel is not None:
        if N is None:
            P = gpu.para_cpu(gpu.reduzir_voxel(P, voxel)[0])
        else:
            P, N = reduzir_voxel_normais(P, N, voxel)
    if N is None:
        N = normais_hibrido(P, raio_normais or (5 * voxel if voxel else 3.0), max_nn)
    ok = np.isfinite(N).all(1)  # pontos sem vizinhos suficientes (normal NaN) saem
    return (P, N) if ok.all() else (P[ok], N[ok])


def amostrar_malha(V, F, n, semente=0, saida_gpu=False):
    """(pontos (n, 3), normais (n, 3)) uniformes em area sobre a malha; normais = normal do triangulo.
    Como `sample_points_uniformly(n, use_triangle_normal=True)`, sorteio na GPU com `semente`."""
    m = gpu.xp()
    V = gpu.para_gpu(V, m.float64)
    F = gpu.para_gpu(F, m.int64)
    if len(F) == 0:
        vazio = m.zeros((0, 3), m.float64)
        return _para_saida(vazio, saida_gpu), _para_saida(vazio, saida_gpu)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = m.cross(b - a, c - a)
    area = 0.5 * m.linalg.norm(cr, axis=1)
    cdf = m.cumsum(area)
    rng = m.random.RandomState(int(semente))
    r = rng.random_sample(int(n)) * cdf[-1]
    t = m.minimum(m.searchsorted(cdf, r), len(F) - 1)
    u, v = rng.random_sample(int(n)), rng.random_sample(int(n))
    flip = (u + v) > 1.0
    u, v = m.where(flip, 1 - u, u), m.where(flip, 1 - v, v)
    P = a[t] + u[:, None] * (b[t] - a[t]) + v[:, None] * (c[t] - a[t])
    nt = cr / m.maximum(m.linalg.norm(cr, axis=1), 1e-300)[:, None]
    return _para_saida(P, saida_gpu), _para_saida(nt[t], saida_gpu)


def distancia_ate(alvo, consultas, dist_max=np.inf, saida_gpu=False):
    """Distancia de cada consulta ao ponto mais proximo de `alvo` (inf alem de `dist_max`).
    Substitui cKDTree(alvo).query(consultas) e compute_point_cloud_distance."""
    if len(alvo) == 0:  # sem alvo: tudo "infinito" (como cKDTree vazio)
        return _para_saida(gpu.para_gpu(np.full(len(consultas), np.inf)), saida_gpu)
    d, _ = gpu.vizinho_mais_proximo(alvo, consultas, dist_max=dist_max)
    return _para_saida(d, saida_gpu)


def vizinho_ate(alvo, consultas, dist_max=np.inf):
    """(dist, idx) numpy do vizinho mais proximo (idx = -1 alem de `dist_max`)."""
    d, j = gpu.vizinho_mais_proximo(alvo, consultas, dist_max=dist_max)
    return gpu.para_cpu(d), gpu.para_cpu(j)


def media_no_raio(alvo, valores, consultas, raio, k=64):
    """Media de `valores` (n, d) dos ate `k` vizinhos de `alvo` a <= raio de cada consulta (substitui
    query_ball_point + media); consultas sem vizinho ficam com NaN. `k` limita a vizinhanca (mais proximos primeiro)."""
    m = gpu.xp()
    V = gpu.para_gpu(valores, m.float64)
    Q = gpu.para_gpu(consultas, m.float64)
    out = m.full((len(Q), V.shape[1]), m.nan, m.float64)
    for a, b in gpu.blocos(len(Q), bytes_por_item=k * 16 + V.shape[1] * 8 * k):
        d, j = gpu.knn(alvo, Q[a:b], int(k), raio_max=raio)
        ok = j >= 0
        S = (V[m.maximum(j, 0)] * ok[:, :, None]).sum(1)
        c = ok.sum(1)
        out[a:b] = m.where((c > 0)[:, None], S / m.maximum(c, 1)[:, None], m.nan)
    return gpu.para_cpu(out)


def mediana_por_grupo(chaves, valores):
    """(chaves unicas ordenadas, mediana de `valores` (n, d) por chave, contagens). Mediana como np.median
    (media dos dois centrais quando a contagem e par), por coluna, na GPU."""
    m = gpu.xp()
    k = gpu.para_gpu(chaves).astype(m.int64)
    V = gpu.para_gpu(valores, m.float64)
    if V.ndim == 1:
        V = V[:, None]
    u, inv, cont = gpu.agrupar(k)
    inv = gpu.para_gpu(inv)
    cont = gpu.para_gpu(cont)
    ini = m.cumsum(cont) - cont
    lo = ini + (cont - 1) // 2
    hi = ini + cont // 2
    cols = []
    for c in range(V.shape[1]):
        ordem = m.lexsort(m.stack([V[:, c], inv]))  # por grupo, depois por valor
        s = V[ordem, c]
        cols.append(0.5 * (s[lo] + s[hi]))
    return gpu.para_cpu(u), gpu.para_cpu(m.stack(cols, axis=1)), gpu.para_cpu(cont)


# ---------------------------------------------------------------- registro
def icp_encadeado(origem, alvo, normais_alvo, T0, tolerancias, iteracoes=50, perda="l2"):
    """ICP ponto-plano em tolerancias decrescentes (cada uma parte da T da anterior). `origem` leva a `alvo`.
    Devolve (T 4x4, info da ultima etapa). `perda='l2'` reproduz o minimo quadrado do Open3D."""
    T = np.array(T0, np.float64)
    info = None
    for tol in tolerancias:
        T, info = gpu.icp_ponto_plano(
            origem, alvo, normais_alvo, T0=T, dist_max=float(tol), iteracoes=iteracoes, perda=perda
        )
    return T, info


def transformar(P, T, saida_gpu=False):
    """P @ R.T + t (numpy ou cupy)."""
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    T = np.asarray(T, np.float64)
    return _para_saida(Pg @ gpu.para_gpu(T[:3, :3].T) + gpu.para_gpu(T[:3, 3]), saida_gpu)


def fracao_ate(alvo, consultas, dist):
    """Fracao das consultas a < `dist` de algum ponto do alvo."""
    d, j = gpu.vizinho_mais_proximo(alvo, consultas, dist_max=float(dist))
    return float((j >= 0).mean()) if len(j) else 0.0


def ransac_plano_open3d(P, limiar=1.0, iteracoes=3000, semente=0):
    """Plano (a, b, c, d) com |n| = 1 e indices dos inliers (como Open3D `segment_plane`), RANSAC na GPU.
    O plano e refinado por PCA nos inliers (gpu.ransac_plano); o sinal de n segue o lado dos objetos."""
    n, d, _ = gpu.ransac_plano(P, limiar=limiar, iteracoes=iteracoes, semente=semente)
    m = gpu.xp()
    Pg = gpu.para_gpu(P, m.float64)
    h = Pg @ gpu.para_gpu(n) + float(d)
    inl = m.nonzero(m.abs(h) <= float(limiar))[0]
    return np.array([n[0], n[1], n[2], d]), gpu.para_cpu(inl)


# ---------------------------------------------------------------- imagens e grades (cupyx.scipy.ndimage)
def ndi_modulo():
    """Modulo de filtros n-dimensionais: cupyx.scipy.ndimage (GPU) ou scipy.ndimage (so com REVO_PERMITIR_CPU=1)."""
    if gpu.em_gpu():
        import cupyx.scipy.ndimage as modulo
    else:
        from scipy import ndimage as modulo
    return modulo


ndi = ndi_modulo


def dilatar(mascara, iteracoes):
    """binary_dilation com estrutura padrao (cruz) repetida `iteracoes` vezes (>= 1), na GPU. O cupyx exige brute_force=True
    para iterations > 1; scipy aceita o argumento. (scipy com iterations=0 repete ate encher: aqui 0 devolve a mascara.)"""
    if iteracoes < 1:
        return mascara
    return ndi_modulo().binary_dilation(mascara, iterations=int(iteracoes), brute_force=True)
