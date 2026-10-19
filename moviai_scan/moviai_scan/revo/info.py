"""Resumo de um scan: quadros, modos, profundidade e maiores saltos de pose entre quadros seguidos.

Os saltos apontam onde o rastreamento pode ter se perdido. Conferir no Revo Scan
(numeracao 'Revo') antes de excluir ou corrigir qualquer quadro.
"""
from collections import Counter

import numpy as np

MODO_SCAN = {0: "geometria", 1: "marcadores"}


def passos(T):
    """Giro (graus) e deslocamento da camera (mm) entre cada quadro e o seguinte."""
    R = np.einsum("kji,kjl->kil", T[:-1, :3, :3], T[1:, :3, :3])   # A^T B
    ang = np.degrees(np.arccos(np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    desl = np.linalg.norm(T[1:, :3, 3] - T[:-1, :3, 3], axis=1)
    return ang, desl


def resumo(scan, n_saltos=8, log=print):
    try:
        modo_scan = scan.propriedades()["scan_param"]["scan_mode"]
    except (OSError, KeyError, ValueError):
        modo_scan = None
    log(f"== {scan.nome} | rastreamento por {MODO_SCAN.get(modo_scan, '?')} | "
        f"{len(scan.ids)} quadros, {int(scan.tem_prof.sum())} com profundidade")
    for m, cnt in sorted(Counter(scan.modo.tolist()).items()):
        com = int(scan.tem_prof[scan.modo == m].sum())
        log(f"   modo {m}: {cnt} quadros ({com} com profundidade)")
    if len(scan.ids) < 2:
        return
    ang, desl = passos(scan.T)
    log(f"   passo entre quadros: giro mediana {np.median(ang):.2f} gr, max {ang.max():.2f} gr | "
        f"deslocamento mediana {np.median(desl):.1f} mm, max {desl.max():.1f} mm")
    rv = scan.numero_revo()
    log("   maiores saltos (seq = ordem de captura; Revo = numero mostrado no Revo Scan; - = sem profundidade):")
    for j in sorted(np.argsort(-ang)[:n_saltos]):
        k = j + 1
        revo = str(rv[k]) if rv[k] >= 0 else "-"
        log(f"     seq {j}->{k} (Revo {revo}): {ang[j]:6.2f} gr, {desl[j]:6.1f} mm | "
            f"modo {scan.modo[j]}->{scan.modo[k]} | hdr0 {scan.hdr[j, 0]}->{scan.hdr[k, 0]}")
