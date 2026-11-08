"""Selecao conservadora de observacoes redundantes por qualidade geometrica 3D.

    resultado = selecionar_regioes(medicoes)  # CUDA obrigatoria; uma medicao por GUID

Entradas ja alinhadas em mm. Nao le/escreve projetos nem corrige poses; apenas devolve
mascaras e referencias auditaveis. As observacoes sem substituto seguro permanecem.
"""

from __future__ import annotations

from typing import Any

import gpu
import numpy as np

_CAMPOS = (
    "pontos",
    "normais",
    "ruido_mm",
    "quadros",
    "incidencia",
    "normal_confiavel",
    "rastreio_ruim",
)
_CAMPOS_OPCIONAIS = {
    "evidencia_temporal": None,
    "consistencia_temporal": None,
    "detalhe_protegido": None,
    "rms_espacial_mm": "float64",
}


def _validar(medicoes: dict) -> dict:
    if not isinstance(medicoes, dict):
        raise TypeError("medicoes deve ser dict[guid, dict]")
    m = gpu.xp()  # sem fallback implicito: so usa CPU com REVO_PERMITIR_CPU=1 explicito
    out = {}
    for guid, dado in medicoes.items():
        faltam = set(_CAMPOS) - set(dado)
        if faltam:
            raise ValueError(f"{guid}: campos ausentes: {sorted(faltam)}")
        p = gpu.para_gpu(dado["pontos"], m.float64)
        n = gpu.para_gpu(dado["normais"], m.float64)
        if p.ndim != 2 or p.shape[1] != 3 or n.shape != p.shape:
            raise ValueError(f"{guid}: pontos e normais devem ter forma (N,3)")
        d = {"pontos": p, "normais": n}
        for campo, dtype in (
            ("ruido_mm", m.float64),
            ("quadros", m.float64),
            ("incidencia", m.float64),
            ("normal_confiavel", m.bool_),
            ("rastreio_ruim", m.bool_),
        ):
            a = gpu.para_gpu(dado[campo], dtype)
            if a.shape != (len(p),):
                raise ValueError(f"{guid}: {campo} deve ter forma (N,)")
            d[campo] = a
        for campo, dtype_nome in _CAMPOS_OPCIONAIS.items():
            if campo in dado:
                dtype = getattr(m, dtype_nome) if dtype_nome else m.bool_
                a = gpu.para_gpu(dado[campo], dtype)
                if a.shape != (len(p),):
                    raise ValueError(f"{guid}: {campo} deve ter forma (N,)")
                d[campo] = a
        if not bool(m.isfinite(p).all()):
            raise ValueError(f"{guid}: pontos devem ser finitos")
        # Normais/incidencia nao finitas sao observacoes sem prova geometrica;
        # ficam inelegiveis e nunca validam cobertura.
        out[guid] = d
    return out


def selecionar_regioes(
    medicoes: dict,
    dist_max: float = 0.3,
    ruido_max: float = 0.05,
    razao_melhoria: float = 1.5,
    margem_melhoria: float = 0.015,
    min_quadros: int = 3,
    cos_min: float = 0.35,
    cos_normais: float = 0.95,
) -> dict[str, Any]:
    """Marca somente observacoes ruins cobertas por evidencias boas de outro scan.

    ``remover[guid]`` e mascara bool (True = remover). ``referencia_scan`` contem
    listas de GUID/None e ``referencia_indice`` vetores GPU int64 (-1 sem fonte).
    ``relatorio`` e serializavel e resume cobertura, motivos e pendencias. Array
    de string e metadados permanecem na CPU; toda geometria e decisao vetorial usa
    GPU. A fonte escolhida nunca e removida por outra decisao simultanea.
    """
    limites = (dist_max, ruido_max, razao_melhoria, margem_melhoria, cos_min, cos_normais)
    if not all(np.isfinite(x) for x in limites):
        raise ValueError("limites devem ser finitos")
    if dist_max <= 0 or ruido_max < 0 or razao_melhoria <= 1 or margem_melhoria < 0:
        raise ValueError("limites de distancia/ruido/melhoria invalidos")
    if min_quadros < 1 or not -1 <= cos_min <= 1 or not -1 <= cos_normais <= 1:
        raise ValueError("limites de quadros/cosseno invalidos")
    med = _validar(medicoes)
    m = gpu.xp()
    guids = list(med)
    remover = {g: m.zeros(len(med[g]["pontos"]), dtype=m.bool_) for g in guids}
    refs_idx = {g: m.full(len(med[g]["pontos"]), -1, dtype=m.int64) for g in guids}
    refs_scan = {g: np.full(len(med[g]["pontos"]), None, dtype=object) for g in guids}
    cobertura = {g: m.zeros(len(med[g]["pontos"]), dtype=m.bool_) for g in guids}
    motivos = {g: {"ruido_significativamente_maior": 0, "rastreio_ruim_coberto": 0} for g in guids}
    pendencias = {g: {"sem_referencia_segura": 0, "suspeito_sem_cobertura": 0, "conflito_geometrico": 0} for g in guids}
    fontes_escolhidas: dict[str, Any] = {g: m.zeros(len(med[g]["pontos"]), dtype=m.bool_) for g in guids}

    # Refs elegiveis: score combina ruido, evidencia temporal e incidencia; densidade
    # de pontos nunca entra. A triagem geom. posterior exige pareamento + normal assinada.
    elegivel = {}
    qualidade = {}
    for g, d in med.items():
        norma = m.linalg.norm(d["normais"], axis=1)
        n_ok = d["normal_confiavel"] & m.isfinite(norma) & (norma > 1e-8) & m.isfinite(d["incidencia"])
        detalhe_conhecido = "detalhe_protegido" in d
        detalhe_geom = d.get("detalhe_protegido", m.ones(len(d["pontos"]), dtype=m.bool_)).copy()
        if "rms_espacial_mm" in d:
            detalhe_geom |= ~m.isfinite(d["rms_espacial_mm"]) | (d["rms_espacial_mm"] > 0.05)
        temporal_ok = d.get("consistencia_temporal", d["ruido_mm"] <= ruido_max)
        elegivel[g] = (
            (d["quadros"] >= min_quadros)
            & (d["incidencia"] >= cos_min)
            & n_ok
            & (d["ruido_mm"] <= ruido_max)
            & temporal_ok
            & ~d["rastreio_ruim"]
            & (m.zeros(len(d["pontos"]), dtype=m.bool_) if not detalhe_conhecido else ~detalhe_geom)
            & m.isfinite(d["ruido_mm"])
        )
        # Qualidade usa apenas evidencias medidas; quantidade de pontos nunca entra.
        qualidade[g] = d["ruido_mm"] + 0.01 / m.maximum(d["quadros"], 1) + 0.01 * (1.0 - m.clip(d["incidencia"], -1, 1))

    propostas = {g: m.zeros(len(med[g]["pontos"]), dtype=m.bool_) for g in guids}
    tolerancia_cobertura = min(dist_max, max(0.05, 2.0 * ruido_max))
    for gc in guids:
        c = med[gc]
        n = len(c["pontos"])
        if n == 0:
            continue
        cn = c["normais"] / m.maximum(m.linalg.norm(c["normais"], axis=1), 1e-30)[:, None]
        norma_c = m.linalg.norm(c["normais"], axis=1)
        candidato_seguro = (
            c["normal_confiavel"]
            & m.isfinite(norma_c)
            & (norma_c > 1e-8)
            & m.isfinite(c["incidencia"])
            & (c["incidencia"] >= cos_min)
            & (c["quadros"] >= min_quadros)
        )
        best_quality = m.full(n, m.inf, dtype=m.float64)
        best_guid = np.full(n, None, dtype=object)
        best_index = m.full(n, -1, dtype=m.int64)
        best_noise = m.full(n, m.inf, dtype=m.float64)
        best_residual = m.full(n, m.inf, dtype=m.float64)
        best_tangential = m.full(n, m.inf, dtype=m.float64)
        # Consultas por blocos para limitar memoria de distancias e manter apenas
        # um indice de referencia residente por vez.
        bloco = max(1, min(n, gpu.tamanho_bloco(160, 0.2)))
        for gr in guids:
            if gr == gc or len(med[gr]["pontos"]) == 0:
                continue
            r = med[gr]
            # Indexar apenas fontes elegiveis evita que um ponto invalido mais
            # proximo esconda uma referencia boa ligeiramente mais distante.
            ids_ref = m.flatnonzero(elegivel[gr])
            if not len(ids_ref):
                continue
            rp = r["pontos"][ids_ref]
            rn = r["normais"][ids_ref]
            rn /= m.maximum(m.linalg.norm(rn, axis=1), 1e-30)[:, None]
            arv = gpu.Indice(rp, dist_max)
            for a in range(0, n, bloco):
                b = min(a + bloco, n)
                dist, j = arv.mais_proximo(c["pontos"][a:b])
                idx_local = m.maximum(j, 0)
                idx = ids_ref[idx_local]
                dot = (cn[a:b] * rn[idx_local]).sum(axis=1)
                residual = m.abs(((c["pontos"][a:b] - rp[idx_local]) * rn[idx_local]).sum(axis=1))
                tangential = m.sqrt(m.maximum(dist * dist - residual * residual, 0.0))
                # dist_max so limita a busca; cobertura de substituicao e bem mais
                # estrita para nao trocar ressaltos/arestas por uma superficie vizinha.
                cob = (j >= 0) & candidato_seguro[a:b] & (tangential <= tolerancia_cobertura) & (dot >= cos_normais)
                cobertura[gc][a:b] |= cob
                # Fonte de maior qualidade observavel por ponto; estavel em empate
                # (ordem GUID de entrada), sem favorecer a nuvem mais densa.
                q = qualidade[gr][idx]
                escolhe = cob & (q < best_quality[a:b])
                pos = m.flatnonzero(escolhe) + a
                if len(pos):
                    loc = pos - a
                    best_quality[pos] = q[loc]
                    best_index[pos] = idx[loc]
                    best_noise[pos] = r["ruido_mm"][idx[loc]]
                    best_residual[pos] = residual[loc]
                    best_tangential[pos] = tangential[loc]
                    # Apenas identificadores/GUID saem para a CPU; geometria permanece na GPU.
                    best_guid[gpu.para_cpu(pos)] = gr

        tem_ref = best_index >= 0
        refboa = tem_ref  # ja passou quadros, incidencia, normal, ruido e rastreio
        incerteza_ref = m.maximum(0.02, 3.0 * best_noise)
        sem_conflito = (best_residual <= incerteza_ref) & (best_tangential <= tolerancia_cobertura)
        diferenca = (c["ruido_mm"] >= best_noise * razao_melhoria) & ((c["ruido_mm"] - best_noise) >= margem_melhoria)
        evidencia_temporal = c.get("evidencia_temporal")
        detalhe = c.get("detalhe_protegido", m.ones(n, dtype=m.bool_)).copy()
        detalhe_conhecido = "detalhe_protegido" in c
        if "rms_espacial_mm" in c:
            detalhe = detalhe | ~m.isfinite(c["rms_espacial_mm"]) | (c["rms_espacial_mm"] > 0.05)
        # Consistencia_temporal significa sigma baixo, nao prova de jitter. Exigir
        # suporte comparavel em >=3 quadros (evidencia_temporal), sigma acima do
        # limite e vantagem da ref. Detalhe/incerteza geometricos sao um gate separado.
        ruido_ruim = (
            diferenca
            & (c["ruido_mm"] > ruido_max)
            & refboa
            & sem_conflito
            & (evidencia_temporal if evidencia_temporal is not None else False)
            & candidato_seguro
            & detalhe_conhecido
            & ~detalhe
        )
        rastreio_coberto = c["rastreio_ruim"] & refboa & sem_conflito & candidato_seguro & detalhe_conhecido & ~detalhe
        remover[gc] = ruido_ruim | rastreio_coberto
        propostas[gc] = remover[gc].copy()
        refs_idx[gc] = best_index
        refs_scan[gc] = best_guid
        motivos[gc]["ruido_significativamente_maior"] = int(ruido_ruim.sum())
        motivos[gc]["rastreio_ruim_coberto"] = int(rastreio_coberto.sum())
        pendencias[gc]["sem_referencia_segura"] = int((~tem_ref).sum())
        suspeito = c["rastreio_ruim"] | (c["ruido_mm"] > ruido_max) | (diferenca & (evidencia_temporal is None))
        pendencias[gc]["suspeito_sem_cobertura"] = int((suspeito & ~tem_ref).sum())
        pendencias[gc]["suspeito_sem_prova_temporal"] = int(
            (
                (diferenca & refboa & (evidencia_temporal is None))
                | (diferenca & refboa & ~evidencia_temporal if evidencia_temporal is not None else False)
                | (diferenca & refboa & detalhe)
            ).sum()
        )
        pendencias[gc]["conflito_geometrico"] = int((refboa & ~sem_conflito & (diferenca | c["rastreio_ruim"])).sum())

    # Proteger cada observacao efetivamente escolhida como fonte; evita ciclos A->B->A.
    for gc in guids:
        idx = refs_idx[gc]
        for gr in set(refs_scan[gc][refs_scan[gc] != None].tolist()):  # noqa: E711
            escolhidos = m.asarray(refs_scan[gc] == gr) & propostas[gc]
            fontes_escolhidas[gr][idx[escolhidos]] = True
    removidos_finais = {}
    relatorio = {"revisao_pendente": True, "scans": {}, "pendencias": []}
    for g in guids:
        removidos = propostas[g] & ~fontes_escolhidas[g]
        bloqueados = propostas[g] & fontes_escolhidas[g]
        remover[g] = removidos
        removidos_finais[g] = removidos
        total = len(med[g]["pontos"])
        nref = int(cobertura[g].sum())
        nr = int(removidos.sum())
        relatorio["scans"][g] = {
            "observacoes": total,
            "cobertas_por_ref_qualificada": nref,
            "remover": nr,
            "preservadas": total - nr,
            "propostas_bloqueadas_como_fonte": int(bloqueados.sum()),
            "motivos": motivos[g],
            "pendencias": pendencias[g],
        }
        if pendencias[g]["suspeito_sem_cobertura"] or pendencias[g]["suspeito_sem_prova_temporal"]:
            relatorio["pendencias"].append({"guid": str(g), **pendencias[g]})
        if pendencias[g]["conflito_geometrico"]:
            relatorio["pendencias"].append(
                {
                    "guid": str(g),
                    "motivo": "conflito_geometrico_preservado",
                    "quantidade": pendencias[g]["conflito_geometrico"],
                }
            )
        if int(bloqueados.sum()):
            relatorio["pendencias"].append(
                {
                    "guid": str(g),
                    "motivo": "proposta_bloqueada_para_preservar_fonte",
                    "quantidade": int(bloqueados.sum()),
                }
            )
    return {
        "remover": removidos_finais,
        "mascara": removidos_finais,
        "referencia_scan": refs_scan,
        "referencia_indice": refs_idx,
        "relatorio": relatorio,
    }
