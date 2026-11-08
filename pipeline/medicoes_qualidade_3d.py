"""Prepara medições 3D por scan sem perder a origem de quadro/pixel.

    resultado = preparar_medicoes(projeto, poses, cache=None, voxel=0.2, raio=0.4)

Usa o cache de observações da fusão e os representantes orientados por visada. Coordenadas e métricas numéricas
ficam na GPU; SQLite, NPY e proveniência ficam na CPU. `rastreio_ruim` é somente um sinal geométrico de
inconsistência temporal, nunca prova isolada de falha de rastreamento.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import gpu
import numpy as np
from cache_observacoes import obter
from comum import sha256
from fusao_superficies import reduzir_orientado

VERSAO = 1


def _hash_objeto(obj) -> str:
    bruto = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=float).encode()
    return hashlib.sha256(bruto).hexdigest()


def _transformar(x, T):
    xp = gpu.xp()
    T = gpu.para_gpu(T, xp.float64)
    return x @ T[:3, :3].T + T[:3, 3]


def _medir_normais_ruido(p, v, bruto_p, bruto_v, bruto_frame, raio):
    """PCA intraquadro, ruído entre quadros e RMS espacial sem misturar camadas de jitter.

    Cada representante consulta a nuvem RAW de cada quadro em sua própria grade. Assim deslocamentos normais que
    atravessam fronteiras voxel não separam a mesma região em grupos temporais distintos.
    """
    xp = gpu.xp()
    n = len(p)
    normals = xp.full((n, 3), xp.nan, dtype=xp.float64)
    rms_espacial = xp.full(n, xp.inf, dtype=xp.float64)
    confiavel = xp.zeros(n, dtype=xp.bool_)
    suspeito = xp.zeros(n, dtype=xp.bool_)
    if not n or not len(bruto_p):
        return (
            normals,
            rms_espacial,
            rms_espacial.copy(),
            confiavel,
            confiavel.copy(),
            confiavel.copy(),
            confiavel.copy(),
            xp.zeros(n, dtype=xp.int32),
            suspeito,
        )
    frames_cpu = gpu.para_cpu(bruto_frame)
    quadros_ids = np.unique(frames_cpu)
    quadros_cpu = [np.flatnonzero(frames_cpu == frame) for frame in quadros_ids]
    bloco = 20_000
    cov_total = xp.zeros((n, 3, 3), dtype=xp.float64)
    suporte_normais = xp.zeros(n, dtype=xp.int32)

    def visitar_quadro(ids_cpu, callback):
        if not len(ids_cpu):
            return
        f = bruto_p[ids_cpu]
        fv = bruto_v[ids_cpu]
        grade = gpu.Grade(f, raio)
        for a in range(0, n, bloco):
            b = min(a + bloco, n)
            q = p[a:b]
            _, j = grade.knn(q, 32, gpu=True)
            valido = j >= 0
            jj = xp.maximum(j, 0)
            valido &= xp.einsum("nkj,nj->nk", fv[jj], v[a:b]) > 0.35
            y = f[jj] - q[:, None, :]
            cont = valido.sum(1)
            callback(a, b, y, valido, cont)

    # A covariance junta apenas a forma intraquadro; não inclui a distância entre poses temporais.
    for ids_cpu in quadros_cpu:

        def acumular_cov(a, b, y, valido, cont):
            mean = xp.where(valido[:, :, None], y, 0).sum(1) / xp.maximum(cont, 1)[:, None]
            z = xp.where(valido[:, :, None], y - mean[:, None, :], 0)
            cov = xp.einsum("nki,nkj->nij", z, z) / xp.maximum(cont, 1)[:, None, None]
            ok = cont >= 6
            cov_total[a:b] += cov * ok[:, None, None]
            suporte_normais[a:b] += ok.astype(xp.int32)

        visitar_quadro(ids_cpu, acumular_cov)

    cov_media = cov_total / xp.maximum(suporte_normais, 1)[:, None, None]
    eig, vec = xp.linalg.eigh(cov_media)
    nn = vec[:, :, 0]
    nn *= xp.where((nn * v).sum(1) < 0, -1.0, 1.0)[:, None]
    normals[:] = nn
    gap = (eig[:, 1] - eig[:, 0]) / xp.maximum(eig[:, 2], 1e-12)
    curv = eig[:, 0] / xp.maximum(eig.sum(1), 1e-12)
    confiavel[:] = (suporte_normais >= 1) & (gap > 0.12) & (curv < 0.08) & xp.isfinite(nn).all(1)

    soma_temporal = xp.zeros(n, dtype=xp.float64)
    soma_temporal2 = xp.zeros(n, dtype=xp.float64)
    soma_spatial = xp.zeros(n, dtype=xp.float64)
    detalhe_quadros = xp.zeros(n, dtype=xp.int32)
    suporte = xp.zeros(n, dtype=xp.int32)
    for ids_cpu in quadros_cpu:

        def acumular_residuos(a, b, y, valido, cont):
            signed = xp.einsum("nki,ni->nk", y, normals[a:b])
            # Pareia quadros pela menor distância tangencial; a média local pode aplainar um pico fino.
            tangent2 = xp.maximum((y * y).sum(2) - signed * signed, 0)
            tangent2 = xp.where(valido, tangent2, xp.inf)
            escolhido = xp.argmin(tangent2, axis=1)
            linhas = xp.arange(b - a)
            tem_vizinho = cont > 0
            mean = signed[linhas, escolhido]
            second = xp.where(valido, signed * signed, 0).sum(1) / xp.maximum(cont, 1)
            mean_espacial = xp.where(valido, signed, 0).sum(1) / xp.maximum(cont, 1)
            var_esp = xp.maximum(second - mean_espacial * mean_espacial, 0)
            ok = (cont >= 3) & confiavel[a:b]
            pico_espacial = xp.where(valido, xp.abs(signed - mean_espacial[:, None]), 0).max(1)
            detalhe_quadros[a:b] += ((cont >= 6) & (pico_espacial > 0.035)).astype(xp.int32)
            soma_temporal[a:b] += mean * ok * tem_vizinho
            soma_temporal2[a:b] += mean * mean * ok * tem_vizinho
            soma_spatial[a:b] += var_esp * ok
            suporte[a:b] += ok.astype(xp.int32)

        visitar_quadro(ids_cpu, acumular_residuos)
    media_temporal = soma_temporal / xp.maximum(suporte, 1)
    var_temporal = soma_temporal2 / xp.maximum(suporte, 1) - media_temporal * media_temporal
    temporal = xp.where(suporte >= 3, xp.sqrt(xp.maximum(var_temporal, 0)), xp.inf)
    rms_espacial = xp.where(suporte > 0, xp.sqrt(soma_spatial / xp.maximum(suporte, 1)), xp.inf)
    confiavel &= suporte >= 3
    # Corrige apenas o viés normal entre quadros e só com suporte; nunca faz média tangencial.
    centralizar = (suporte >= 3) & confiavel & xp.isfinite(temporal)
    p += normals * (media_temporal * centralizar)[:, None]
    consistencia = (suporte >= 3) & (temporal <= 0.05) & confiavel
    evidencia = (suporte >= 3) & confiavel
    # Dispersão espacial alta ou PCA ambígua é detalhe/borda incerta, nunca motivo suficiente para apagar.
    detalhe = (~confiavel) | (rms_espacial > 0.05) | (detalhe_quadros >= 3)
    suspeito = evidencia & (temporal > 0.05) & ~detalhe
    return normals, temporal, rms_espacial, confiavel, consistencia, evidencia, detalhe, suporte, suspeito


def preparar_medicoes(projeto, poses, saida=None, cache=None, voxel=0.2, raio=0.4, calibracoes_fixas=None):
    """Prepara dados de qualidade para GUIDs em `poses`.

    `poses` mapeia GUID para matriz 4x4 scan→referencial comum. Retorna `medicoes` (arrays CuPy), `origem`
    (índice global, quadro, pixel de cada representante), `observacoes` (mmap estruturado do cache) e `hashes`
    para vincular qualquer máscara à mesma versão de dataset, calibração e poses.
    """
    if not np.isfinite(voxel) or voxel <= 0 or not np.isfinite(raio) or raio <= 0:
        raise ValueError("voxel e raio devem ser finitos e positivos")
    xp = gpu.xp()
    projeto = Path(projeto)
    data = projeto / "data" if (projeto / "data").is_dir() else projeto
    calibracoes = {}
    med = {}
    origem = {}
    observacoes = {}
    hashes = {}
    poses_comuns = {}
    calibracoes_resultado = {}
    calibracoes_fixas = calibracoes_fixas or {}
    for guid, matriz in poses.items():
        T = np.asarray(matriz, dtype=np.float64)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"pose inválida para {guid}")
        if not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=2e-3):
            raise ValueError(f"rotação da pose não rígida para {guid}")
        scan = data / str(guid)
        calib_fixa = calibracoes_fixas.get(guid)
        cache_dir, meta, _ = obter(scan, cache, calib_fixa)
        obs = np.load(cache_dir / "observacoes.npy", mmap_mode="r", allow_pickle=False)
        if not len(obs):
            raise ValueError(f"scan sem observações: {guid}")
        observacoes_sha = meta.get("arquivo_observacoes_sha256") or sha256(cache_dir / "observacoes.npy")
        if sha256(cache_dir / "observacoes.npy") != observacoes_sha:
            raise ValueError(f"cache de observações alterado durante a preparação: {guid}")
        observacoes[guid] = obs
        calibracoes[guid] = meta.get("calibracao", {})
        calibracoes_resultado[guid] = calibracoes[guid]
        hashes[guid] = {
            "dataset_sha256": sha256(scan / "cache" / "frames.dataset"),
            "preview_sha256": sha256(scan / "raw_preview.ply"),
            "calibracao_sha256": _hash_objeto(calibracoes[guid]),
            "poses_sha256": _hash_objeto(T.tolist()),
            "cache_sha256": sha256(cache_dir / "observacoes.json"),
            "arquivo_observacoes_sha256": observacoes_sha,
            "cache_pasta": str(cache_dir.resolve()),
        }
        poses_comuns[guid] = T.copy()
        # A redução orientada é feita sobre observações reais cacheadas; seus índices continuam globais.
        p0, v0, _, _, rep, _ = reduzir_orientado(obs, meta["quadros"], voxel)
        p = _transformar(gpu.para_gpu(p0, xp.float64), T)
        v = gpu.para_gpu(v0, xp.float64) @ gpu.para_gpu(T[:3, :3].T, xp.float64)
        v /= xp.maximum(xp.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        raw_p = gpu.para_gpu(np.asarray(obs["p"]), xp.float64)
        raw_v = gpu.para_gpu(np.asarray(obs["v"]), xp.float64)
        raw_p = _transformar(raw_p, T)
        raw_v = raw_v @ gpu.para_gpu(T[:3, :3].T, xp.float64)
        raw_v /= xp.maximum(xp.linalg.norm(raw_v, axis=1, keepdims=True), 1e-12)
        raw_frame = gpu.para_gpu(np.asarray(obs["quadro"], dtype=np.int32), xp.int32)
        normals, noise, spatial_rms, reliable, temporal_ok, temporal_evidence, detail, quadros_n, bad_track = (
            _medir_normais_ruido(p, v, raw_p, raw_v, raw_frame, raio)
        )
        incidence = xp.einsum("ij,ij->i", normals, v)
        med[guid] = {
            "pontos": p,
            "normais": normals,
            "ruido_mm": noise,
            "rms_espacial_mm": spatial_rms,
            "quadros": quadros_n,
            "incidencia": incidence,
            "normal_confiavel": reliable,
            "consistencia_temporal": temporal_ok,
            "evidencia_temporal": temporal_evidence,
            "detalhe_protegido": detail,
            "rastreio_ruim": bad_track,
        }
        ids = np.asarray(rep, dtype=np.int64)
        origem[guid] = {
            "observacao_global": ids,
            "quadro": np.asarray(obs["quadro"][ids], dtype=np.int32),
            "pixel": np.asarray(obs["pixel"][ids], dtype=np.int32),
        }
    return {
        "medicoes": med,
        "origem": origem,
        "observacoes": observacoes,
        "hashes": hashes,
        "poses": poses_comuns,
        "calibracoes": calibracoes_resultado,
        "_meta": {"versao": VERSAO, "voxel_mm": float(voxel), "raio_mm": float(raio)},
    }
