"""Fusao propria de alta nitidez para pecas pequenas com detalhes finos (2-4 scans ja limpos).

    python pipeline/fundir_nitido.py <projeto_limpo> --poses poses.json --saida PASTA [--voxel 0.1]

poses.json: {guid: 4x4} leva cada scan ao referencial comum (partida; e refinada aqui por ICP multi-vista).
1. Todos os pixels dos quadros, com UMA calibracao para todos os scans (foco gravado + centro ajustado).
2. Normal local por superficie de centroides de 0,2 mm, orientada para o lado visto pelas cameras.
3. Observacoes boas: nao rasantes (cos >= --cos-min) e perto do plano local (<= --dist-max). Viram uma grade fina
   de 0,05 mm com peso cos^2 e direcao de visada.
4. Amostras = centroides de --voxel mm com >= --min-obs observacoes, sem pontos soltos (--viz-min vizinhos a 0,3 mm).
5. Mean-shift ao longo da normal sobre a grade fina (gaussiana tangencial --sigma-t, faixa de altura decrescente):
   colapsa a espessura da superficie, que a media por voxel NAO colapsa (camadas de voxels empilhadas), sem puxar
   vizinhos do outro lado de arestas nem a face oposta. Depois, uma amostra por voxel de --voxel mm.
6. Apoio fraco: superficies com poucas observacoes (paredes de cavidade) entram em voxel de --voxel-fraco mm, longe
   (> --raio-fraco) da superficie principal e so se >= --scans-fraco scans independentes as confirmam.
7. Remove componentes soltos pequenos (--comp-min da peca).

Todo o calculo numerico roda na GPU (CuPy, `gpu.py`; sem CUDA o script para). Na CPU ficam so o SQLite e a decodificacao
das mascaras dos quadros, a calibracao (`RevoScan.calibrar`: Nelder-Mead com cKDTree, sem equivalente na GPU, mas
reaproveitada pelo cache) e a leitura/gravacao de PLY (Open3D). Leitura, calibracao e pontos 3D vao para
`cache_observacoes` (chave = sha256 do frames.dataset + K): a 2a execucao sobre a mesma entrada nao le nem recalibra
(`--sem-cache` desliga; `--cache-raiz` muda a pasta). A vizinhanca usa TODOS os pontos do raio (nao os k mais proximos).
Saida: nuvem_nitida.ply (com normais), nuvem_apoio_fraco.ply (todas as observacoes fracas confirmadas, mais ruidosas:
paredes de cavidade, sem o filtro de lamina), poses_finais.json, relatorio.json. Nao altera o projeto de entrada.
"""

import argparse
import json
import os
import sys
import tempfile
import time
import types

import numpy as np
import open3d as o3d

AQUI = os.path.dirname(os.path.abspath(__file__))
import geom_gpu  # noqa: E402
import gpu  # noqa: E402
import pca3_gpu  # noqa: E402
from cache_observacoes import calibracao_scan, pontos_gpu, pontos_scan  # noqa: E402
from recursos import baixa_prioridade  # noqa: E402
from revo_projeto import RevoScan  # noqa: E402

_GRADES = {}


def _grade(F, raio):
    """Grade GPU reaproveitada enquanto F (mesmo objeto) e o raio nao mudam (a grade fina F do mean-shift).
    Guarda a referencia a F: o id nunca e reutilizado por outro array enquanto a grade esta no cache."""
    chave = (id(F), len(F), float(raio))
    if chave not in _GRADES:
        _GRADES.clear()
        _GRADES[chave] = (F, gpu.Grade(F, raio))
    return _GRADES[chave][1]


def grupos(X, voxel):
    """(inverso, contagens) dos voxels de `voxel` mm de X (n, 3), na ordem lexicografica (x, y, z) das celulas."""
    m = gpu.xp()
    X = gpu.para_gpu(X, m.float64)
    _, inv, n = gpu.agrupar(m.floor(X / voxel).astype(m.int64))
    return inv, n


def mais_proximo(C, X, raio):
    """(d, idx) do vizinho mais proximo de cada X em C, so ate `raio` (alem disso d=inf e idx=-1)."""
    return gpu.vizinho_mais_proximo(C, X, raio)


def centroides(X, voxel, w=None, min_obs=1):
    """Centroide (ponderado por w) de cada voxel com >= min_obs pontos; (C, contagens, inverso de todos os voxels)."""
    m = gpu.xp()
    X = gpu.para_gpu(X, m.float64)
    inv, n = grupos(X, voxel)
    if w is None:
        soma, pesos = gpu.somar_por_grupo(X, inv, len(n)), n.astype(m.float64)
    else:
        w = gpu.para_gpu(w, m.float64)
        soma, pesos = gpu.somar_por_grupo(X * w[:, None], inv, len(n)), gpu.somar_por_grupo(w, inv, len(n))
    C = soma / pesos[:, None]
    manter = n >= min_obs
    return C[manter], n[manter], inv


def _pca(g, Q, sigma):
    """(s0, M, media) na GPU: M = sum w (y - media)(y - media)^T (mesma matriz de Grade.pca, sem sair do dispositivo)."""
    m = gpu.xp()
    S = g.somas(Q, sigma, gpu=True).astype(m.float64)
    s0 = S[:, 0]
    media = S[:, 1:4] / m.maximum(s0, 1e-30)[:, None]
    xx, yy, zz, xy, xz, yz = (S[:, c] for c in range(4, 10))
    M = m.empty((len(Q), 3, 3), m.float64)
    M[:, 0, 0], M[:, 1, 1], M[:, 2, 2] = xx, yy, zz
    M[:, 0, 1] = M[:, 1, 0] = xy
    M[:, 0, 2] = M[:, 2, 0] = xz
    M[:, 1, 2] = M[:, 2, 1] = yz
    M -= s0[:, None, None] * (media[:, :, None] * media[:, None, :])
    return s0, M, media


def normais_locais(C, ref, raio=0.5, sigma=0.25):
    """Normal do plano local (PCA ponderado por gaussiana sobre TODOS os pontos do raio), orientada para `ref`."""
    m = gpu.xp()
    C = gpu.para_gpu(C, m.float64)
    ref = gpu.para_gpu(ref, m.float64)
    g = gpu.Grade(C, raio)
    N = m.empty_like(C)
    for a, b in gpu.blocos(len(C), bytes_por_item=1500):
        _, M, _ = _pca(g, C[a:b], sigma)
        N[a:b] = pca3_gpu.menor_autopar(M)[1]
    return N * m.sign((N * ref).sum(1, keepdims=True) + 1e-12)


def carregar(pasta, scan, K, raiz_cache=None, usar_cache=True):
    """(P (n, 3) float64 GPU, centro da camera por quadro (float32), pontos por quadro, info do cache).
    `scan()` cria o RevoScan so se for preciso ler os quadros (cache sem acerto ou desligado)."""

    def gerar():
        sc = scan()
        ks = [k for k in range(len(sc.ids)) if sc.tem_prof[k]]
        pts = [pontos_gpu(sc, k, K)[0] for k in ks]
        cam = np.array([sc.T[k][:3, 3] for k in ks], np.float32).reshape(-1, 3)
        return gpu.xp().concatenate(pts), cam, np.array([len(p) for p in pts], np.int64)

    if not usar_cache:
        P, cam, cont = gerar()
        return P, cam, cont, dict(reutilizado=False)
    P, cam, cont, _, reuso = pontos_scan(pasta, K, gerar, raiz_cache)
    return P, cam, cont, dict(reutilizado=reuso)


def icp_multivista(nuv, T, normais=None):
    """ICP ponto-plano multivista na GPU (gpu.icp_ponto_plano, Tukey k=0,15 mm): cada scan (menos o primeiro, que e a
    referencia) contra a uniao dos outros nas poses atuais, com a mesma escala de tolerancias do caminho antigo (Open3D).
    nuv {guid: centroides (n, 3)}; normais {guid: (n, 3)} (calculadas aqui como o Open3D: ate 40 vizinhos em 0,8 mm, se omitidas)."""
    m = gpu.xp()
    guids = list(nuv)
    nuv = {g: gpu.para_gpu(c, m.float64) for g, c in nuv.items()}
    if normais is None:
        normais = {g: geom_gpu.normais_hibrido(c, 0.8, max_nn=40, saida_gpu=True) for g, c in nuv.items()}
    for g in guids:  # sem vizinhanca para o plano (normal NaN): nao serve de ponto nem de alvo
        ok = m.isfinite(normais[g]).all(1)
        nuv[g], normais[g] = nuv[g][ok], normais[g][ok]
    for tol in (1.0, 0.6, 0.4, 0.3, 0.25, 0.2, 0.2, 0.2):
        for g in guids[1:]:
            pts, nrm = [], []
            for o in guids:
                if o != g:
                    Rt = gpu.para_gpu(T[o][:3, :3].T.copy(), m.float64)
                    pts.append(nuv[o] @ Rt + gpu.para_gpu(T[o][:3, 3].copy(), m.float64))
                    nrm.append(normais[o] @ Rt)
            T[g], _ = gpu.icp_ponto_plano(
                nuv[g], m.concatenate(pts), m.concatenate(nrm), T[g], dist_max=tol, iteracoes=60, perda="tukey", c=0.15
            )
    return T


def sem_soltos(C, ref, viz_min, raio=0.3):
    n = gpu.contar_no_raio(C, C, raio)
    ok = n > viz_min
    return C[ok], ref[ok], int((~ok).sum())


def espessura_local(C, raio):
    """RMS (mm) do plano local e numero de vizinhos: superficie = fina; nuvem de multipercurso = grossa."""
    m = gpu.xp()
    C = gpu.para_gpu(C, m.float64)
    g = gpu.Grade(C, raio)
    e = m.empty(len(C), m.float64)
    nv = m.empty(len(C), m.int64)
    for a, b in gpu.blocos(len(C), bytes_por_item=1500):
        s0, M, _ = _pca(g, C[a:b], None)
        e[a:b] = m.sqrt(m.maximum(pca3_gpu.menor_autopar(M)[0], 0) / m.maximum(s0, 1))
        nv[a:b] = s0.astype(m.int64)
    return e, nv


def mean_shift(S, N, F, W, VF, sigma_t, sigma_h, raio=0.35, max_passo=0.1):
    """Leva cada amostra S ao longo da normal N ate a moda de altura da grade fina F (pesos W, visada VF)."""
    m = gpu.xp()
    S = gpu.para_gpu(S, m.float64)
    N = gpu.para_gpu(N, m.float64)
    D = gpu.para_gpu(_grade(F, raio).mean_shift(S, gpu.para_cpu(N), W, VF, sigma_t, sigma_h, max_passo))
    return S + D[:, None] * N, D


def por_voxel(X, V, sel, w, voxel, minimo):
    """Centroide ponderado, visada media (unitaria), soma dos pesos e contagem dos voxels com >= minimo observacoes."""
    m = gpu.xp()
    Xs, Vs, ws = X[sel], V[sel], w[sel]
    inv, nn = grupos(Xs, voxel)
    soma = gpu.somar_por_grupo(ws, inv, len(nn))
    C = gpu.somar_por_grupo(Xs * ws[:, None], inv, len(nn)) / soma[:, None]
    Vm = gpu.somar_por_grupo(Vs, inv, len(nn))
    k = nn >= minimo
    return C[k], Vm[k] / m.maximum(m.linalg.norm(Vm[k], axis=1, keepdims=True), 1e-12), soma[k], nn[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--voxel", type=float, default=0.1, help="amostras do mean-shift")
    ap.add_argument("--voxel-saida", type=float, default=0.07, help="espacamento final depois do colapso")
    ap.add_argument(
        "--esp-max", type=float, default=0.07, help="mm; RMS local (r=0,3) acima disso e nuvem, nao superficie"
    )
    ap.add_argument("--esp-max-fraco", type=float, default=0.05, help="idem na camada de apoio fraco (r=0,6)")
    ap.add_argument("--cos-min", type=float, default=0.3)
    ap.add_argument("--dist-max", type=float, default=0.25)
    ap.add_argument("--min-obs", type=int, default=6)
    ap.add_argument("--viz-min", type=int, default=12)
    ap.add_argument("--sigma-t", type=float, default=0.12)
    ap.add_argument("--voxel-fraco", type=float, default=0.15)
    ap.add_argument("--min-obs-fraco", type=int, default=5)
    ap.add_argument("--raio-fraco", type=float, default=0.3)
    ap.add_argument("--scans-fraco", type=int, default=2)
    ap.add_argument("--comp-min", type=float, default=0.002)
    ap.add_argument("--sem-cache", action="store_true", help="nao usa o cache de calibracao e pontos (le tudo de novo)")
    ap.add_argument("--cache-raiz", default=None, help="pasta do cache (padrao: pipeline/output/cache_obs)")
    ap.add_argument(
        "--motor", choices=["cuda"], default="cuda", help=argparse.SUPPRESS
    )  # compatibilidade: so existe GPU
    a = ap.parse_args()
    baixa_prioridade()
    m = gpu.xp()  # ErroGPU se nao houver CUDA
    t0 = time.time()
    tempos = {}
    marca = [time.time()]

    def etapa(nome):
        if gpu.em_gpu():
            m.cuda.Stream.null.synchronize()
        agora = time.time()
        tempos[nome] = round(agora - marca[0], 2)
        marca[0] = agora
        print(f"[{nome}: {tempos[nome]:.1f} s]", flush=True)

    if os.path.exists(a.saida) and os.listdir(a.saida):
        sys.exit(f"destino ja existe: {a.saida}")
    os.makedirs(a.saida, exist_ok=True)
    poses0 = {g: np.array(T) for g, T in json.load(open(a.poses)).items()}
    rel = dict(parametros=vars(a))
    usar_cache = not a.sem_cache
    cache_info = dict(ativo=usar_cache, calibracao_reutilizada=0, pontos_reutilizados=0)

    with tempfile.TemporaryDirectory() as tmp:
        pasta = {g: os.path.join(a.projeto, "data", g) for g in poses0}
        abertos = {}

        def scan(g):  # RevoScan so quando o cache nao basta (copia o frames.dataset)
            if g not in abertos:
                abertos[g] = RevoScan(pasta[g], tmp)
            return abertos[g]

        f = RevoScan.foco_gravado(types.SimpleNamespace(pasta=pasta[next(iter(poses0))]), 0)
        centros = []
        for g in poses0:
            if usar_cache:
                Kg, _, reuso = calibracao_scan(pasta[g], lambda g=g: scan(g).calibrar(0), 0, a.cache_raiz)
                cache_info["calibracao_reutilizada"] += int(reuso)
            else:
                Kg, _ = scan(g).calibrar(0)
            centros.append(Kg[2:])
        cx, cy = np.median(centros, 0)
        K = (f, f, float(cx), float(cy))
        rel["K"] = K
        print("K", np.round(K, 3), flush=True)
        dados = {}
        for g in poses0:
            P, cam, cont, info = carregar(pasta[g], lambda g=g: scan(g), K, a.cache_raiz, usar_cache)
            dados[g] = (P, cam, cont)
            cache_info["pontos_reutilizados"] += int(info["reutilizado"])
        for sc in abertos.values():
            sc.c.close()
    rel["cache"] = cache_info
    rel["quadros"] = {g: len(d[2]) for g, d in dados.items()}
    rel["pontos_entrada"] = {g: len(d[0]) for g, d in dados.items()}
    etapa("leitura_e_calibracao")

    nuv = {g: centroides(d[0], 0.2, min_obs=3)[0] for g, d in dados.items()}
    T = icp_multivista(nuv, {g: x.copy() for g, x in poses0.items()})
    del nuv
    etapa("icp_multivista")
    json.dump({g: x.tolist() for g, x in T.items()}, open(os.path.join(a.saida, "poses_finais.json"), "w"), indent=1)
    for g, x in T.items():
        D = np.linalg.inv(poses0[g]) @ x
        print(
            g,
            "ICP: %.3f graus %.3f mm"
            % (np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))), np.linalg.norm(D[:3, 3])),
            flush=True,
        )

    Xl, Vl, rl = [], [], []
    for i, (g, (P, cam, cont)) in enumerate(dados.items()):
        Rt = gpu.para_gpu(T[g][:3, :3].T.copy(), m.float64)
        t = gpu.para_gpu(T[g][:3, 3].copy(), m.float64)
        Xg = P @ Rt + t
        Vg = m.repeat(gpu.para_gpu(cam, m.float64) @ Rt + t, gpu.para_gpu(cont), axis=0) - Xg
        Vg /= m.linalg.norm(Vg, axis=1, keepdims=True)
        Xl.append(Xg)
        Vl.append(Vg)
        rl.append(m.full(len(Xg), i, m.int8))
        dados[g] = None
    del dados, P
    X, V, rot = m.concatenate(Xl), m.concatenate(Vl), m.concatenate(rl)
    del Xl, Vl, rl
    print("observacoes", len(X), flush=True)

    # superficie grossa e normais orientadas
    inv_t, n_t = grupos(X, 0.2)
    grosso = n_t >= 3
    C0 = (gpu.somar_por_grupo(X, inv_t, len(n_t)) / n_t[:, None])[grosso]
    v_med = gpu.somar_por_grupo(V, inv_t, len(n_t))[grosso]
    del inv_t, n_t
    N0 = normais_locais(C0, v_med)
    dist, idx = mais_proximo(C0, X, 0.4)
    sem = idx < 0  # sem centroide a < 0,4 mm
    idx = m.maximum(idx, 0)
    cosv = m.empty(len(X), m.float64)
    h = m.empty(len(X), m.float64)
    for b0, b1 in gpu.blocos(len(X), bytes_por_item=400):
        N_obs = N0[idx[b0:b1]]
        cosv[b0:b1] = (N_obs * V[b0:b1]).sum(1)
        h[b0:b1] = ((X[b0:b1] - C0[idx[b0:b1]]) * N_obs).sum(1)
    del idx, N_obs
    if bool(sem.any()):  # contam como fora do plano (nunca entram em ok)
        cosv[sem] = 1.0
        h[sem] = m.inf
        rel["obs_sem_vizinho_04mm"] = int(sem.sum())
    del sem
    plano_ok = (dist < 0.4) & (m.abs(h) <= a.dist_max)
    ok = plano_ok & (cosv >= a.cos_min)
    del dist, plano_ok
    rel["obs_boas"] = float(ok.mean())
    rel["obs_rasantes"] = float((cosv < a.cos_min).mean())
    rel["obs_fora_plano"] = float((m.abs(h) > a.dist_max).mean())
    print(
        "obs boas %.1f%% (rasantes %.1f%%, fora do plano %.1f%%)"
        % (100 * rel["obs_boas"], 100 * rel["obs_rasantes"], 100 * rel["obs_fora_plano"]),
        flush=True,
    )

    pesos = m.maximum(cosv, 0.05) ** 2
    del cosv
    F, VF, W, _ = por_voxel(X, V, ok, pesos, 0.05, 1)
    S, ref, _, n = por_voxel(X, V, ok, pesos, a.voxel_saida, a.min_obs)
    del ok, pesos
    rel["grade_fina"] = len(F)
    rel["amostras_iniciais"] = len(S)
    rel["obs_por_voxel_mediana"] = float(m.median(n))
    print("grade fina", len(F), "| amostras", len(S), "| obs/voxel", rel["obs_por_voxel_mediana"], flush=True)
    etapa("superficie_grossa_e_grades")
    S, ref, soltos = sem_soltos(S, ref, a.viz_min)  # grade mais densa: so tira o que esta realmente solto
    rel["soltos_removidos"] = soltos
    print("soltos removidos", soltos, flush=True)

    for sig_h in (0.12, 0.10, 0.08, 0.06, 0.05):
        N = normais_locais(S, ref, raio=0.4, sigma=0.18)
        S, D = mean_shift(S, N, F, W, VF, a.sigma_t, sig_h)
        rel[f"desloc_h{sig_h}_um"] = dict(
            mediana=float(m.median(m.abs(D)) * 1000), p99=float(m.percentile(m.abs(D), 99) * 1000)
        )
        print("mean-shift", sig_h, rel[f"desloc_h{sig_h}_um"], flush=True)
    del F, VF, W
    _GRADES.clear()
    gpu.liberar_memoria()
    etapa("mean_shift_5_passadas")

    # uma amostra por voxel: as camadas colapsadas caem no mesmo voxel
    inv_, nn = grupos(S, a.voxel_saida)
    C = gpu.somar_por_grupo(S, inv_, len(nn)) / nn[:, None]
    ref = gpu.somar_por_grupo(ref, inv_, len(nn))
    rel["amostras_apos_colapso"] = len(C)
    print("apos colapso", len(S), "->", len(C), flush=True)
    C, ref, soltos = sem_soltos(C, ref, 12)
    rel["soltos_removidos_fim"] = soltos
    e, _ = espessura_local(C, 0.3)
    fina = e <= a.esp_max
    rel["nuvem_removida_principal"] = int((~fina).sum())
    print(
        "espessura local: mediana %.1f um p99 %.1f um | removidos > %.0f um: %d"
        % (
            float(m.median(e)) * 1000,
            float(m.percentile(e, 99)) * 1000,
            a.esp_max * 1000,
            rel["nuvem_removida_principal"],
        ),
        flush=True,
    )
    C, ref = C[fina], ref[fina]
    C, ref, soltos = sem_soltos(C, ref, 12)
    rel["soltos_removidos_fim2"] = soltos
    etapa("colapso_espessura_soltos")

    # apoio fraco: confirmado por >= scans-fraco scans e longe da superficie principal
    d_princ = mais_proximo(C, X, a.raio_fraco)[0]  # so importa d > raio_fraco
    fracos = m.flatnonzero((d_princ > a.raio_fraco) & (m.abs(h) <= 1.0))
    del d_princ, h
    if len(fracos):
        Xf, Vxf = X[fracos], V[fracos]
        inv_, nn = grupos(Xf, a.voxel_fraco)
        cont = m.zeros((len(nn), int(m.unique(rot).size)), bool)
        cont[inv_, rot[fracos]] = True
        sel = (nn >= a.min_obs_fraco) & (cont.sum(1) >= a.scans_fraco)
        Cf = (gpu.somar_por_grupo(Xf, inv_, len(nn)) / nn[:, None])[sel]
        Vf = gpu.somar_por_grupo(Vxf, inv_, len(nn))[sel]
        n_antes = 0
        if len(Cf):
            camada = o3d.geometry.PointCloud(
                o3d.utility.Vector3dVector(gpu.para_cpu(Cf))
            )  # todas as confirmadas, em arquivo separado
            camada.normals = o3d.utility.Vector3dVector(gpu.para_cpu(normais_locais(Cf, Vf, raio=0.6, sigma=0.3)))
            o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_apoio_fraco.ply"), camada, write_ascii=False)
            ef, nv = espessura_local(Cf, 0.6)
            sup = (ef <= a.esp_max_fraco) & (nv >= 20)  # lamina, nao volume: multipercurso e descartado
            n_antes = len(Cf)
            Cf, Vf = Cf[sup], Vf[sup]
            if len(Cf):
                Nf = normais_locais(Cf, Vf, raio=0.6, sigma=0.3)
                for sig_h in (0.1, 0.07, 0.05):  # mesmo colapso, sobre as observacoes da propria camada
                    Cf, _ = mean_shift(Cf, Nf, Xf, m.ones(len(Xf)), Vxf, 0.2, sig_h, raio=0.5)
                    Nf = normais_locais(Cf, Vf, raio=0.6, sigma=0.3)
        rel["apoio_fraco"] = dict(
            obs=len(fracos),
            voxels_candidatos=len(nn),
            voxels_um_scan=int(((nn >= a.min_obs_fraco) & (cont.sum(1) < a.scans_fraco)).sum()),
            confirmados=n_antes if len(Cf) else 0,
            pontos=len(Cf),
        )
        print("apoio fraco", rel["apoio_fraco"], flush=True)
        C, ref = m.concatenate([C, Cf]), m.concatenate([ref, Vf])
        del Xf, Vxf
    del X, V, rot
    _GRADES.clear()
    gpu.liberar_memoria()
    etapa("apoio_fraco")
    N = normais_locais(C, ref, raio=0.5, sigma=0.22)

    # componentes soltos (grade de 0,5 mm)
    lp, tam = gpu.componentes_conexos_voxel(C, 0.5, 26)
    manter = tam[lp] >= a.comp_min * len(C)
    rel["componentes"] = dict(total=len(tam), pontos_removidos=int((~manter).sum()), maiores=tam[:8].tolist())
    print("componentes", rel["componentes"], flush=True)
    C, N = gpu.para_cpu(C[manter]), gpu.para_cpu(N[manter])

    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(C))
    p.normals = o3d.utility.Vector3dVector(N)
    o3d.io.write_point_cloud(os.path.join(a.saida, "nuvem_nitida.ply"), p, write_ascii=False)
    rel["pontos_saida"] = len(C)
    etapa("normais_componentes_gravacao")
    rel["motor"] = "cuda" if gpu.em_gpu() else "cpu (REVO_PERMITIR_CPU)"
    rel["tempos_s"] = tempos
    rel["segundos"] = round(time.time() - t0)
    json.dump(rel, open(os.path.join(a.saida, "relatorio.json"), "w"), indent=1, default=float)
    print("pontos saida", len(C), "| extensao", np.round(np.ptp(C, axis=0), 2), "|", rel["segundos"], "s", flush=True)


if __name__ == "__main__":
    main()
