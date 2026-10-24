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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dph import intrinsecos, listar, profundidade


def mascara_peca(d, salto=2.0, dilat=4, escala=0.1):
    m = d > 0
    if not m.any():
        return m
    z = d.astype(np.float32) * escala
    g = np.zeros_like(m)
    for dy, dx in ((0, 1), (1, 0)):
        a = np.abs(z - np.roll(z, (dy, dx), (0, 1))) > salto
        g |= a | np.roll(a, (-dy, -dx), (0, 1))
    mm = m & ~g
    lab, n = ndi.label(ndi.binary_dilation(mm, iterations=dilat) & m)
    if n == 0:
        return m & False
    sz = ndi.sum(m, lab, range(1, n + 1))
    return lab == 1 + int(np.argmax(sz))


def limpar_scan(args):
    pasta, salto, dilat = args
    tot = mant = 0
    esc = intrinsecos(pasta)[4]
    for f in listar(pasta):
        d = profundidade(f)
        s = mascara_peca(d, salto, dilat, esc)
        tot += int((d > 0).sum())
        mant += int(s.sum())
        np.where(s, d, 0).astype(np.uint16).tofile(f)
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
    with ProcessPoolExecutor(a.jobs) as ex:
        for s, tot, mant in ex.map(limpar_scan, tarefas):
            print(f"{s}: {mant}/{tot} pixels mantidos ({100 * mant / max(tot, 1):.1f}%)")


if __name__ == "__main__":
    main()
