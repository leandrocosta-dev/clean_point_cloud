"""Avalia corte de apoio por scan e recuperacao por outras vistas, sem editar frames.

    python pipeline/avaliar_corte_apoio.py ORIGINAL LIMPO --poses alinhamento.json --saida NOVA

Plano estimado no preview ORIGINAL (a limpeza pode ter apagado a mesa).
Cobertura medida contra amostras dos outros scans apos o mesmo corte.
Nao classifica automaticamente os pontos retirados como mesa ou peca.
Todo o calculo (RANSAC, centroides, distancias) roda na GPU.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

from recursos import baixa_prioridade

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "REVO_THREADS"):
    os.environ[_k] = "12"
import gpu
import numpy as np
from fusao_superficies import salvar_json
from limpar_por_placa import _area_plano, amostra_mundo
from revo_projeto import RevoScan, ler_ply, plano_ransac


def plano_de_apoio(original, peca):
    """Maior area dentre quatro planos; rejeita corte atravessando a selecao. GPU (RANSAC, residuos, area)."""
    m = gpu.xp()
    original = gpu.para_gpu(original, m.float64)
    peca = gpu.para_gpu(peca, m.float64)
    resto = original.copy()
    candidatos = []
    for i in range(4):
        if len(resto) < 100:
            break
        n, d, _ = plano_ransac(resto, thr=0.4, it=1500, seed=i)
        n = n / np.linalg.norm(n)
        ng = gpu.para_gpu(n)
        if float(m.median(peca @ ng + d)) < 0:
            n, d = -n, -d
            ng = gpu.para_gpu(n)
        res = original @ ng + d
        inliers = m.abs(res) < 0.4
        ri = res[inliers]
        mad = float(1.4826 * m.median(m.abs(ri - m.median(ri))))
        candidatos.append(
            dict(
                n=n.tolist(),
                d=float(d),
                area_mm2=_area_plano(original, n, d, thr=0.4),
                inliers=int(inliers.sum()),
                dispersao_mad_mm=mad,
                selecao_abaixo_menos_05_pct=float(100 * (peca @ ng + d < -0.5).mean()),
            )
        )
        resto = resto[m.abs(resto @ ng + d) >= 0.4]
    if not candidatos:
        raise ValueError("nenhum plano de apoio encontrado")
    best = max(candidatos, key=lambda c: c["area_mm2"])
    if best["selecao_abaixo_menos_05_pct"] > 5 or best["dispersao_mad_mm"] > 0.25:
        raise ValueError("plano incerto ou atravessando a selecao; nao autorizar corte")
    return best, candidatos


def cobertura_removidos(p, mascara, outros, tolerancias):
    """Cobertura dos pontos removidos por `outros` a cada tolerancia (GPU: vizinho mais proximo exato)."""
    m = gpu.xp()
    p = gpu.para_gpu(p, m.float64)
    mascara = gpu.para_gpu(mascara, bool)
    rem = p[mascara]
    if not len(rem):
        return dict(removidos=0, cobertura={})
    if not len(outros):
        return dict(removidos=len(rem), cobertura={str(t): 0 for t in tolerancias})
    dist = gpu.vizinho_mais_proximo(gpu.para_gpu(outros, m.float64), rem)[0]
    return dict(
        removidos=len(rem),
        distancia_outros_mediana_mm=float(m.median(dist)),
        cobertura={str(t): float(100 * (dist <= t).mean()) for t in tolerancias},
        sem_apoio_na_maior_tolerancia=int((dist > max(tolerancias)).sum()),
    )


def executar(original, limpo, poses, saida, alturas=(0.5, 1.0), tolerancias=(0.3, 0.5)):
    if any(not np.isfinite(x) or x <= 0 for x in (*alturas, *tolerancias)):
        raise ValueError("alturas e tolerancias devem ser positivas")
    baixa_prioridade()
    original, limpo, saida = map(Path, (original, limpo, saida))
    saida.mkdir(parents=True, exist_ok=False)
    doc = json.loads(Path(poses).read_text())
    transforms = doc.get("poses_usadas", doc)
    mapa = json.loads((limpo / "identificadores.json").read_text())
    scans = [g for g in transforms if g in mapa and (original / "data" / g / "raw_preview.ply").exists()]
    if len(scans) < 2:
        raise ValueError("necessarios dois scans com poses e IDs correspondentes")
    pontos = {}
    mundos = {}
    heights = {}
    rel = {}
    for g in scans:
        with tempfile.TemporaryDirectory() as tmp:
            scan = RevoScan(str(limpo / "data" / mapa[g]), tmp)
            try:
                calib = {}
                for m in sorted(set(scan.modo[scan.tem_prof].tolist())):
                    k, e = scan.calibrar(m)
                    calib[str(m)] = dict(K=k, erro_preview_mm=e)
                p = amostra_mundo(scan, 4, 4).astype(gpu.xp().float64)  # amostra ja nasce na GPU
            finally:
                scan.c.close()
        # Centroides reduzem ruido e redundancia; nao sao amostra uniforme de area.
        p, _ = gpu.reduzir_voxel(p, 0.15, origem=p.min(0) - 0.075)  # centroides por voxel (grade como a do Open3D)
        _, raw = ler_ply(original / "data" / g / "raw_preview.ply")
        plano, candidatos = plano_de_apoio(raw[:, :3], p)
        h = p @ gpu.para_gpu(np.array(plano["n"])) + plano["d"]
        T = np.asarray(transforms[g], float)
        if T.shape != (4, 4) or not np.isfinite(T).all() or not np.allclose(T[3], [0, 0, 0, 1]):
            raise ValueError("pose invalida")
        pontos[g] = p
        heights[g] = h
        mundos[g] = p @ gpu.para_gpu(T[:3, :3].T) + gpu.para_gpu(T[:3, 3])
        rel[g] = dict(plano=plano, candidatos=candidatos, calibracao=calib, centroides=len(p), cortes={})
        np.savez(
            saida / f"{g}.npz",
            pontos_scan=gpu.para_cpu(p),
            pontos_comuns=gpu.para_cpu(mundos[g]),
            altura_mm=gpu.para_cpu(h),
        )
        print(g, "plano", plano, "centroides", len(p), flush=True)
    for altura in alturas:
        for g in scans:
            outros = gpu.xp().vstack([mundos[j][heights[j] > altura] for j in scans if j != g])
            rem = heights[g] <= altura
            m = cobertura_removidos(mundos[g], rem, outros, tolerancias)
            m.update(mantidos=int((~rem).sum()), remocao_pct=float(100 * rem.mean()))
            # Faixa adicional para comparar 1 mm contra 0,5 mm na mesma referencia.
            if altura > min(alturas):
                extra = (heights[g] > min(alturas)) & (heights[g] <= altura)
                m["faixa_adicional"] = cobertura_removidos(mundos[g], extra, outros, tolerancias)
            rel[g]["cortes"][str(altura)] = m
    resultado = dict(
        scans=rel,
        alturas_mm=list(alturas),
        tolerancias_mm=list(tolerancias),
        origem=str(original.resolve()),
        base_limpa=str(limpo.resolve()),
        aviso="Medida amostrada de proximidade entre scans, nao prova de pertencimento a peca. "
        "Regioes removidas incluem mesa residual e superficie real. Falta de cobertura impede presumir recuperacao. "
        "Planos requerem conferencia geometrica antes de aplicar. Nenhum frame foi alterado.",
    )
    salvar_json(saida / "avaliacao.json", resultado)
    return resultado


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("original")
    ap.add_argument("limpo")
    ap.add_argument("--poses", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--alturas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--tolerancias", nargs="+", type=float, default=[0.3, 0.5])
    a = ap.parse_args()
    print(json.dumps(executar(a.original, a.limpo, a.poses, a.saida, a.alturas, a.tolerancias), indent=2))


if __name__ == "__main__":
    main()
