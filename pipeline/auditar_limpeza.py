"""Audita uma limpeza de raw: compara, quadro a quadro, o scan original com o limpo.

Uso:
    python pipeline/auditar_limpeza.py <projeto_original> <projeto_limpo> <scan_id> [--passo 6] [--altura 15]

Mede o que a limpeza REMOVEU, por altura acima do plano da mesa (RANSAC no original):
  - removidos acima de `altura` mm  -> corpo da peca: o esperado e ZERO (e a garantia de nao cortar);
  - removidos entre 4 e `altura` mm -> base/pes da peca ou relevo da mesa: conferir;
  - removidos ate 4 mm             -> mesa.
Tambem mostra a extensao do que ficou e o maior pedaco removido acima de 4 mm (para ver se foi
uma peca de apoio inteira ou uma fatia da peca).
Compara so pixels: nao depende do fuse.ply do Revo.
Todas as medidas (RANSAC, alturas, projecao, distancias) rodam na GPU; os pixels sao comparados por indice (np.isin
vira pertinencia por mascara na GPU).
"""

import argparse
import os
import tempfile

import gpu
import numpy as np
from revo_projeto import RevoScan, _morf, plano_ransac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original")
    ap.add_argument("limpo")
    ap.add_argument("scan")
    ap.add_argument(
        "--passo",
        type=int,
        default=8,
        help="usa 1 a cada N quadros (os pixels sao todos comparados: passo de pixel desloca a amostra apos a limpeza)",
    )
    ap.add_argument("--altura", type=float, default=15.0)
    ap.add_argument(
        "--referencia",
        help="projeto limpo de referencia (ex.: limpeza padrao sem remover mesa): conta o que o novo tirou a mais",
    )
    a = ap.parse_args()
    with (
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t1,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t2,
        tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t3,
    ):
        so = RevoScan(os.path.join(a.original, "data", a.scan), t1)
        sl = RevoScan(os.path.join(a.limpo, "data", a.scan), t2)
        assert list(so.ids) == list(sl.ids), "os dois scans tem quadros diferentes"
        for m in sorted(set(int(m) for m, p in zip(so.modo, so.tem_prof) if p)):
            so.calibrar(m)
        sl.K = so.K  # mesma calibracao nos dois lados: a comparacao e pixel a pixel
        ks = [k for k in range(len(so.ids)) if so.tem_prof[k]][:: a.passo]
        sr = None
        if a.referencia:
            sr = RevoScan(os.path.join(a.referencia, "data", a.scan), t3)
            sr.K = so.K
        m = gpu.xp()
        orig, kept, kref = [], [], []

        def pertence(pix, pix_sub, n):
            """True para os pixels de `pix` que tambem estao em `pix_sub` (mascara na GPU em vez de np.isin)."""
            tab = m.zeros(int(max(pix.max(initial=0), pix_sub.max(initial=0))) + 1 if n else 1, bool)
            tab[gpu.para_gpu(pix_sub)] = True
            return tab[gpu.para_gpu(pix)]

        for k in ks:
            P, pix = so.pontos(k, na_gpu=True)
            _, pixl = sl.pontos(k)
            if P is None:
                continue
            orig.append(P)
            if sr is not None:
                _, pr = sr.pontos(k)
                kref.append(pertence(pix, pr, len(pix)) if pr is not None else m.zeros(len(P), bool))
            kept.append(pertence(pix, pixl, len(pix)) if pixl is not None else m.zeros(len(P), bool))
        P = m.vstack(orig)
        kept = m.concatenate(kept)
        n, d, frac = plano_ransac(P[:: max(1, len(P) // 60000)], thr=1.0, it=800)
        ng = gpu.para_gpu(n)
        h = P @ ng + d
        rem = ~kept
        print(
            f"quadros {len(ks)} | pontos {len(P):,} | mantidos {float(kept.mean()) * 100:.1f}% | plano da mesa: {frac * 100:.0f}% dos pontos"
        )
        for nome, mk in (
            (f"acima de {a.altura:g} mm (corpo)", h > a.altura),
            (f"4 a {a.altura:g} mm", (h > 4) & (h <= a.altura)),
            ("ate 4 mm (mesa)", h <= 4),
        ):
            print(
                f"  {nome:28s} total {int(mk.sum()):>10,} | removidos {int((mk & rem).sum()):>10,} ({int((mk & rem).sum()) / max(1, int(mk.sum())) * 100:5.1f}%)"
            )
        k_ = P[kept]
        print(
            "extensao do que ficou (mm):",
            np.round(gpu.para_cpu(k_.max(0) - k_.min(0)), 0),
            "| altura max acima do plano:",
            round(float((k_ @ ng + d).max()), 1),
        )
        # Garantia de nao cortar: pontos removidos que caem DENTRO da projecao vertical do corpo mantido
        u = np.cross(n, [1.0, 0, 0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        uv = m.stack([P @ gpu.para_gpu(u), P @ gpu.para_gpu(v)], axis=1)
        lo = uv.min(0) - 5
        ij = m.floor((uv - lo) / 2.0).astype(m.int64)
        fp = m.zeros(tuple(int(x) for x in gpu.para_cpu(ij.max(0)) + 3), bool)
        ik = ij[kept & (h > a.altura)]
        fp[ik[:, 0], ik[:, 1]] = True
        fp = _morf("binary_closing", fp, iterations=3)
        dentro = fp[ij[:, 0], ij[:, 1]]
        corte = rem & (h > a.altura) & dentro
        print(
            f"removidos acima de {a.altura:g} mm DENTRO da projecao do corpo mantido: {int(corte.sum()):,} "
            f"({int(corte.sum()) / max(1, int((kept & (h > a.altura)).sum())) * 100:.3f}% do corpo mantido)"
        )
        if sr is not None:
            kr = m.concatenate(kref)
            mais = kr & ~kept
            print(
                f"vs referencia: tirou a mais {int(mais.sum()):,} pts ({int(mais.sum()) / max(1, int(kr.sum())) * 100:.1f}% do que a referencia mantinha)"
            )
            for nome, mk in (
                (f"acima de {a.altura:g} mm", h > a.altura),
                (f"4 a {a.altura:g} mm", (h > 4) & (h <= a.altura)),
                ("ate 4 mm", h <= 4),
            ):
                print(
                    f"   a mais {nome:20s}: {int((mais & mk).sum()):>10,} | dentro da projecao do corpo: {int((mais & mk & dentro).sum()):>9,}"
                )
        # Os removidos dentro da coluna do corpo sao pixels da propria peca ou de outro objeto? Distancia 3D
        # ate a superficie mantida: <= 1 mm = pixel da peca apagado (corte); longe = objeto separado.
        Pk = P[kept]
        arv = gpu.para_gpu(Pk[:: max(1, int(kept.sum()) // 3_000_000)])

        def hist(mk, rotulo):
            n_ = int(mk.sum())
            if n_ == 0:
                return
            Q = P[mk][:: max(1, n_ // 400_000)]
            dd = gpu.vizinho_mais_proximo(arv, Q)[0]  # distancia exata ao mantido, na GPU
            print(
                f"   {rotulo}: n={n_:,} | dist. ate o mantido: <=1 mm {float((dd <= 1).mean()) * 100:.1f}% | 1-3 mm {float(((dd > 1) & (dd <= 3)).mean()) * 100:.1f}% | 3-10 mm {float(((dd > 3) & (dd <= 10)).mean()) * 100:.1f}% | >10 mm {float((dd > 10).mean()) * 100:.1f}%"
            )

        hist(corte, "removidos na coluna do corpo")
        if sr is not None:
            hist(mais & (h > a.altura), "a mais que a referencia (acima de 15 mm)")
            hist(mais & (h > 4) & (h <= a.altura), "a mais que a referencia (4 a 15 mm)")
        alto = rem & (h > 4)
        if bool(alto.any()):
            ha = h[alto]
            print(
                f"removidos acima de 4 mm: {int(alto.sum()):,} pts | altura media {float(ha.mean()):.1f} mm | p99 {float(m.percentile(ha, 99)):.1f} mm | max {float(ha.max()):.1f} mm"
            )
        so.c.close()
        sl.c.close()


if __name__ == "__main__":
    main()
