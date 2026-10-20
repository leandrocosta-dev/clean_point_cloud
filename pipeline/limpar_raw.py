"""Limpa o raw (quadros brutos) de um projeto .revo ANTES da fusao no Revo Scan.

Uso:
    python pipeline/limpar_raw.py <pasta_do_projeto> [--scans id1 id2 ...] [--saida <pasta>]

Sem --scans processa todos os scans do .revo. Em cada quadro do frames.dataset apaga
os pixels de profundidade que nao sao a peca (mesa, marcadores, apoios, mao). As poses
nao mudam. Grava uma copia do projeto; o original nao e tocado e o destino nao pode existir.
Scans com calibracao ruim sao copiados sem alteracao. Depois abra o .revo limpo no
Revo Scan e refaca a fusao.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import (RevoScan, segmentar_peca, adicionar_referencia_fuse, limpar_scan,
                          gravar_revo, render_comparacao, ler_ply)

ERRO_MAX_CALIB = 0.6   # mm; acima disso (e acima de ERRO_REL_MAX x espacamento do preview) o scan nao e limpo
ERRO_REL_MAX = 1.25    # erro/espacamento do preview; ajuste perfeito ja da ~0,55, e preview esparso sobe o erro absoluto
MIN_MANTIDO = 0.30     # se a limpeza tirar mais de 70% dos pontos, a escolha da peca e suspeita: scan fica intacto


def copiar_intacto(proj, saida, s):
    shutil.copytree(os.path.join(proj, "data", s), os.path.join(saida, "data", s))


def processar(proj, s, saida, pixels=True, ilhas=True, periferia_mm=0.0, per_banda=2.5, per_alargue=20.0):
    """Limpa um scan (roda num processo proprio). Devolve (id, relatorio, log).

    Qualquer scan que nao possa ser limpo com seguranca (sem raw_preview, calibracao ruim, limpeza
    que tira pontos demais, erro) e copiado sem alteracao: um scan problematico nao derruba o lote.
    """
    t0 = time.time()
    log = [f"== {s}"]
    pasta = os.path.join(proj, "data", s)
    if not (os.path.isfile(os.path.join(pasta, "raw_preview.ply")) and os.path.isfile(os.path.join(pasta, "cache", "frames.dataset"))):
        log.append("   sem raw_preview.ply/frames.dataset (nao e scan de quadros): copiado sem alteracao")
        copiar_intacto(proj, saida, s)
        return s, dict(status="intacto", motivo="sem quadros"), log
    scan = None
    try:
        with tempfile.TemporaryDirectory() as tmp:
            try:
                scan = RevoScan(pasta, tmp)
                modos = sorted(set(int(m) for m, p in zip(scan.modo, scan.tem_prof) if p))
                erros = {}
                for m in modos:
                    K, e = scan.calibrar(m)
                    erros[m] = e
                    log.append(f"   calibracao modo {m}: foco {K[0]:.0f}/{K[1]:.0f} | erro {e:.3f} mm")
                # erro = distancia ao vizinho mais proximo no raw_preview; preview esparso sobe o erro
                # sem a calibracao estar ruim, por isso vale o erro relativo ao espacamento
                _, prev = ler_ply(os.path.join(scan.pasta, "raw_preview.ply"))
                X = prev[:, :3].astype(float)
                esp = float(np.median(cKDTree(X).query(X, k=2, workers=-1)[0][:, 1]))
                if max(erros.values()) > ERRO_MAX_CALIB and max(erros.values()) / esp > ERRO_REL_MAX:
                    log.append(f"   PULADO: erro {max(erros.values()):.2f} mm = {max(erros.values())/esp:.2f} x espacamento do preview; copiado sem alteracao")
                    return s, dict(status="pulado", erros=erros), _fechar(scan, proj, saida, s, log)
                seg = segmentar_peca(scan, periferia_mm=periferia_mm, per_banda=per_banda, per_alargue=per_alargue, log=lambda *a: log.append(" ".join(map(str, a))))
                seg = adicionar_referencia_fuse(scan, seg, log=lambda *a: log.append(" ".join(map(str, a))))
                r = limpar_scan(scan, seg, os.path.join(saida, "data", s), log=lambda *a: log.append(" ".join(map(str, a))),
                                pixels=pixels, ilhas=ilhas, min_mantido=MIN_MANTIDO)
                if r.get("revisar"):
                    return s, dict(status="revisar", erros=erros, **r), _fechar(scan, proj, saida, s, log, copiar=False)
                render_comparacao(scan, seg, os.path.join(saida, f"{s}_antes_depois.png"))
                return s, dict(status="limpo", erros=erros, peca=seg["peca"], **r, tempo_s=round(time.time() - t0)), _fechar(scan, proj, saida, s, log, copiar=False)
            except Exception as e:
                log.append(f"   ERRO: {type(e).__name__}: {e}; scan copiado sem alteracao")
                shutil.rmtree(os.path.join(saida, "data", s), ignore_errors=True)
                return s, dict(status="erro", erro=str(e)), _fechar(scan, proj, saida, s, log)
    finally:
        if scan is not None:
            scan.c.close()


def _fechar(scan, proj, saida, s, log, copiar=True):
    """Fecha o banco do scan e, se pedido, copia o scan original intacto para a saida."""
    scan.c.close()
    if copiar and not os.path.exists(os.path.join(saida, "data", s)):
        copiar_intacto(proj, saida, s)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projeto")
    ap.add_argument("--scans", nargs="+")
    ap.add_argument("--saida")
    ap.add_argument("--sem-pixels", action="store_true", help="nao filtra pixels voadores nem manchas 2D")
    ap.add_argument("--sem-ilhas", action="store_true", help="nao remove ilhas 3D soltas")
    ap.add_argument("--periferia-mm", type=float, default=0.0, help="remove a mesa (faixa do plano) a mais de X mm da pegada da peca; 0 = desligado")
    ap.add_argument("--periferia-banda", type=float, default=2.5, help="espessura (mm) da faixa do plano removida")
    ap.add_argument("--periferia-alargue", type=float, default=20.0, help="mm do corpo ate onde o relevo baixo ainda conta como peca (bases alargadas)")
    ap.add_argument("--jobs", type=int, default=0, help="scans em paralelo (0 = automatico)")
    a = ap.parse_args()
    proj = os.path.abspath(a.projeto)
    revo = [f for f in os.listdir(proj) if f.endswith(".revo")][0]
    nome = os.path.splitext(revo)[0] + "_limpo"
    saida = os.path.abspath(a.saida or os.path.join(os.path.dirname(proj), nome))
    if os.path.exists(saida):
        sys.exit(f"destino ja existe: {saida}")
    todos = [n["guid"] for n in json.load(open(os.path.join(proj, revo), encoding="utf-8"))["nodes"]]
    scans = a.scans or todos
    os.makedirs(os.path.join(saida, "data"))
    # cada scan usa ~2 GB de RAM por 10 M de pontos; 6 em paralelo cabem folgados em 64 GB
    jobs = a.jobs or min(len(scans), 6, os.cpu_count() or 1)
    os.environ["REVO_THREADS"] = str(max(1, (os.cpu_count() or 1) // jobs))
    print(f"{len(scans)} scans, {jobs} em paralelo, {os.environ['REVO_THREADS']} threads cada", flush=True)
    rel = {}
    with ProcessPoolExecutor(jobs) as ex:
        for s, r, log in ex.map(processar, [proj] * len(scans), scans, [saida] * len(scans),
                                    [not a.sem_pixels] * len(scans), [not a.sem_ilhas] * len(scans),
                                    [a.periferia_mm] * len(scans), [a.periferia_banda] * len(scans), [a.periferia_alargue] * len(scans)):
            print("\n".join(log), flush=True)
            rel[s] = r
    for s in todos:   # scans fora de --scans vao sem alteracao
        if s not in scans:
            shutil.copytree(os.path.join(proj, "data", s), os.path.join(saida, "data", s))
    gravar_revo(os.path.join(proj, revo), saida, nome, todos)
    json.dump(rel, open(os.path.join(saida, "relatorio_limpeza.json"), "w", encoding="utf-8"), indent=2)
    print("saida:", saida)


if __name__ == "__main__":
    main()
