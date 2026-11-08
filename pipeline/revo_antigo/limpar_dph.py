"""Limpa os quadros .dph (formato antigo do Revo) de um projeto: fica so a peca.

Uso: python pipeline/revo_antigo/limpar_dph.py <projeto> --saida <pasta nova> --scans id1 id2 [--jobs 3]

Por quadro: separa a imagem de profundidade por saltos (> --salto mm entre vizinhos), junta de novo
com dilatacao e fica com o MAIOR componente (a peca); marcadores, bolhas da mesa e pixels voadores
somem. Poses e o resto do projeto nao mudam. O original nao e tocado; o destino nao pode existir.
Validado no scan 172242: 97,4% dos pixels ficam e 99,8% deles estao a < 1 mm do fuse.ply do Revo.
"""

import argparse
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import geom_gpu as gg
import gpu

from pipeline.revo_antigo.dph import intrinsecos, listar, profundidade


def mascara_peca(d, salto=2.0, dilat=4, escala=0.1, saida_gpu=False):
    """Mascara bool da peca (maior componente apos separar por salto de profundidade). Toda a imagem na GPU
    (cupyx.scipy.ndimage); `d` numpy ou cupy; devolve numpy (cupy se saida_gpu)."""
    xp = gpu.xp()
    ndi = gg.ndi()
    d = gpu.para_gpu(d)
    m = d > 0
    if not bool(m.any()):
        return m if saida_gpu else gpu.para_cpu(m)
    z = d.astype(xp.float32) * xp.float32(escala)
    g = xp.zeros_like(m)
    for dy, dx in ((0, 1), (1, 0)):
        a = xp.abs(z - xp.roll(z, (dy, dx), (0, 1))) > salto
        g |= a | xp.roll(a, (-dy, -dx), (0, 1))
    mm = m & ~g
    lab, n = ndi.label(gg.dilatar(mm, dilat) & m)
    if int(n) == 0:
        r = m & False
    else:
        sz = xp.bincount(lab[m & (lab > 0)], minlength=int(n) + 1)[1:]  # pixels da imagem por componente
        r = lab == 1 + int(sz.argmax())
    return r if saida_gpu else gpu.para_cpu(r)


def limpar_scan(args):
    pasta, salto, dilat = args
    tot = mant = 0
    esc = intrinsecos(pasta)[4]
    for f in listar(pasta):
        d = profundidade(f)
        dg = gpu.para_gpu(d)
        s = mascara_peca(dg, salto, dilat, esc, saida_gpu=True)
        tot += int((dg > 0).sum())
        mant += int(s.sum())
        gpu.para_cpu(gpu.xp().where(s, dg, 0).astype(gpu.xp().uint16)).tofile(f)
    return os.path.basename(pasta), tot, mant


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--scans", nargs="+", required=True)
    ap.add_argument("--salto", type=float, default=2.0)
    ap.add_argument("--dilat", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    if os.path.exists(a.saida):
        sys.exit("destino ja existe: " + a.saida)
    shutil.copytree(a.projeto, a.saida)
    tarefas = [(os.path.join(a.saida, "data", s), a.salto, a.dilat) for s in a.scans]
    with ThreadPoolExecutor(a.jobs) as ex:  # a GPU faz a conta; as threads so sobrepoem leitura e escrita dos .dph
        for s, tot, mant in ex.map(limpar_scan, tarefas):
            print(f"{s}: {mant}/{tot} pixels mantidos ({100 * mant / max(tot, 1):.1f}%)")


if __name__ == "__main__":
    main()
