"""Refina as poses de varios scans ja alinhados usando so superficie de MESMA face (normais orientadas pela camera).

    python pipeline/refinar_poses.py <projeto_limpo> --poses poses.json --saida poses_refinadas.json [--escala 3.9]

Para que serve: alinhadores por ponto mais proximo (FPFH/ICP) puxam a face da FRENTE de um scan contra a face de TRAS
do outro em parede fina, e o resultado fica deslocado ao longo da normal (Project20261012_105241, moldura de painel:
todas as celulas de 30 mm com o mesmo sinal, ate 1,24 mm). Aqui o residuo e a altura da superficie dos outros scans
sobre a do proprio scan, medida ao longo da normal e so entre vizinhos de mesma face (a mesma medida do
`ciclo_uniao`), e cada scan e movido como corpo rigido (Gauss-Newton ponto-plano, pesos de Tukey) ate o vies sumir.
O scan de referencia (pose mais perto da identidade) fica parado. A pose so e trocada se o desvio mediano do scan
cair; o relatorio (`<saida>.relatorio.json`) traz desvio e vies por celula antes e depois. Nao escreve no projeto.
Medidas em mm multiplicadas por --escala (1 = peca de ~45 mm; `fluxo_uniao --refinar-poses` passa a do atlas).
"""

import argparse
import json
import os
import sys
import time

import ciclo_uniao as C
import gpu
import numpy as np
from comum import configurar_log, salvar_json_atomico

_LOG = configurar_log(__name__)


def rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def passo_rigido(P, N, d, sel):
    """Movimento rigido pequeno (R, v, centro) que leva a superficie P (normais N) a subir `d` ao longo de N nos pontos
    `sel`: minimos quadrados ponto-plano com pesos de Tukey. Devolve None se houver poucos pontos."""
    m = gpu.xp()
    p, n, r = P[sel], N[sel], d[sel]
    if len(p) < 200:
        return None
    cen = p.mean(0)
    q = p - cen
    A = m.concatenate([m.cross(q, n), n], axis=1)
    mad = float(m.median(m.abs(r - m.median(r)))) * 1.4826
    c = 4.685 * max(mad, 1e-6)
    u = r / c
    w = m.where(m.abs(u) < 1, (1 - u * u) ** 2, 0.0)
    H = gpu.para_cpu((A * w[:, None]).T @ A)
    g = gpu.para_cpu((A * w[:, None]).T @ r)
    x = np.linalg.solve(H + 1e-9 * np.trace(H) * np.eye(6), g)
    return rodrigues(x[:3]), x[3:], gpu.para_cpu(cen)


def medir(con, par, cel):
    an = con.analisar()
    out = {}
    for s, x in an.items():
        sel = (x["n_cobrem"] > 0) if par else x["consenso"]
        ad = gpu.xp().abs(x["desvio"][sel])
        v = C.vies_por_celula(con.sup[s]["P"], x["desvio"], sel, cel)
        out[s] = dict(
            pontos_medidos=int(sel.sum()),
            desvio_p50_mm=round(float(gpu.xp().percentile(ad, 50)), 4) if len(ad) else None,
            desvio_p90_mm=round(float(gpu.xp().percentile(ad, 90)), 4) if len(ad) else None,
            vies_abs_max_mm=v.get("vies_abs_max_mm"),
            vies_global_mm=v.get("vies_global_mm"),
        )
    return an, out


def refinar(sup, poses, a, rodadas=12, cel=30.0):
    """Move os scans (menos a referencia) ate o desvio ao longo da normal parar de cair. Devolve (poses novas,
    relatorio). `sup` e alterado no lugar."""
    m = gpu.xp()
    par = len(sup) == 2
    ref = min(sup, key=lambda s: float(np.abs(poses[s] - np.eye(4)).sum()))
    novas = {s: np.array(T, float) for s, T in poses.items()}
    con = C.Consenso(sup, a)
    an, antes = medir(con, par, cel)
    hist = []
    for rodada in range(1, rodadas + 1):
        maior = 0.0
        for s in sup:
            if s == ref:
                continue
            x = con.analisar([s])[s]
            sel = (x["n_cobrem"] > 0) if par else x["consenso"]
            passo = passo_rigido(sup[s]["P"], sup[s]["N"], x["desvio"], sel)
            if passo is None:
                continue
            R, v, cen = passo
            D = np.eye(4)
            D[:3, :3], D[:3, 3] = R, cen - R @ cen + v
            Rg, tg = gpu.para_gpu(R.T, m.float64), gpu.para_gpu(D[:3, 3], m.float64)
            sup[s]["P"] = sup[s]["P"] @ Rg + tg
            sup[s]["N"] = sup[s]["N"] @ Rg
            novas[s] = D @ novas[s]
            con.mover(s)
            ext = float(gpu.para_cpu(m.abs(sup[s]["P"] - gpu.para_gpu(cen)).max()))
            maior = max(
                maior,
                float(np.linalg.norm(v))
                + float(np.linalg.norm(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))) * ext,
            )
        hist.append(round(maior, 4))
        _LOG.info("rodada %d: maior movimento %.4f mm", rodada, maior)
        if maior < 0.02 * a.tol:
            break
    _, depois = medir(con, par, cel)
    aceito = {}
    for s in sup:
        p0, p1 = antes[s]["desvio_p50_mm"], depois[s]["desvio_p50_mm"]
        aceito[s] = s == ref or (p0 is not None and p1 is not None and p1 < p0)
    if not all(aceito.values()):  # sem melhora: ficam as poses de entrada
        novas = {s: np.array(T, float) for s, T in poses.items()}
    rel = dict(referencia=ref, movimento_por_rodada_mm=hist, antes=antes, depois=depois, aceito=all(aceito.values()))
    return novas, rel


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto", help="projeto ja limpo (so a peca)")
    ap.add_argument("--poses", required=True, help="JSON {guid: 4x4 scan->referencia}")
    ap.add_argument("--saida", required=True, help="JSON novo com as poses refinadas")
    ap.add_argument("--scans", nargs="+", help="padrao: todos os que tem pose")
    ap.add_argument("--escala", type=float, default=1.0, help="multiplica as medidas em mm (1 = peca de ~45 mm)")
    ap.add_argument("--rodadas", type=int, default=12)
    a = ap.parse_args(argv)
    saida = os.path.abspath(a.saida)
    if os.path.exists(saida):
        sys.exit(f"destino ja existe: {saida}")
    k = a.escala
    par = argparse.Namespace(
        voxel=0.15 * k, min_obs=4, raio=0.4 * k, alcance_normal=1.2 * k, viz_min=6, cos_min=0.6, tol=0.15 * k,
        acordo=0.10 * k, min_scans=2, area_min=0.5 * k * k, orientados=set(),
    )  # fmt: skip
    with open(a.poses, encoding="utf-8") as fh:
        poses = {s: np.asarray(v, float) for s, v in json.load(fh).items()}
    _, _, todos = C.AP.scans_do_projeto(os.path.abspath(a.projeto))
    scans = [s for s in todos if s in poses and (not a.scans or s in a.scans)]
    if len(scans) < 2:
        sys.exit("preciso de pelo menos 2 scans com pose")
    t0 = time.time()
    sup = C.fundir_todos(os.path.abspath(a.projeto), scans, poses, par)
    novas, rel = refinar(sup, {s: poses[s] for s in sup}, par, a.rodadas)
    for s in sup:
        _LOG.info(
            "%s: desvio p50 %s -> %s mm | vies maximo por celula %s -> %s mm",
            s,
            rel["antes"][s]["desvio_p50_mm"],
            rel["depois"][s]["desvio_p50_mm"],
            rel["antes"][s]["vies_abs_max_mm"],
            rel["depois"][s]["vies_abs_max_mm"],
        )
    final = {s: (novas[s] if s in novas else poses[s]).tolist() for s in poses}
    salvar_json_atomico(saida, final)
    rel.update(
        origem=os.path.abspath(a.poses),
        projeto=os.path.abspath(a.projeto),
        escala=k,
        tempo_s=round(time.time() - t0, 1),
    )
    salvar_json_atomico(saida + ".relatorio.json", rel)
    _LOG.info(
        "gravado: %s%s (%.0f s)",
        saida,
        "" if rel["aceito"] else " | SEM MELHORA: poses de entrada mantidas",
        time.time() - t0,
    )
    return 0 if rel["aceito"] else 2


if __name__ == "__main__":
    sys.exit(main())
