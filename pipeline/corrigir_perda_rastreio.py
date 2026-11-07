"""Corrige um scan que perdeu o rastreamento no meio da captura.

Uso:
    python pipeline/corrigir_perda_rastreio.py <projeto> <scan> --ultimo-bom 292 --primeiro-bom 297 --saida <projeto_novo>

Depois da perda, os quadros seguintes ficam num referencial deslocado em relacao aos anteriores (rigido: a
fusao mostra a peca duplicada). O script:
  1. estima por ICP ponto-a-plano a transformacao que leva os quadros >= primeiro-bom para o referencial dos
     quadros <= ultimo-bom (janelas de --janela quadros de cada lado);
  2. regrava a pose (FrameInfo.bin, 128 bytes a partir do byte 16) dos quadros >= primeiro-bom;
  3. remove a profundidade dos quadros entre os dois (os de pose perdida), como o Revo faz com quadros sem profundidade;
  4. confere: os quadros corrigidos tem que encaixar nos anteriores.
Copia o projeto inteiro para --saida (recusa destino existente); so o frames.dataset do scan muda.
"""
import argparse
import os
import shutil
import sqlite3
import sys
import tempfile

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan  # noqa: E402


def nuvem(scan, K, ks, passo=8):
    X = np.vstack([scan.pontos(k, K, passo=passo)[0] for k in ks if scan.tem_prof[k]])
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(X)
    p = p.voxel_down_sample(1.0)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4, max_nn=30))
    return p


def estimar(scan, K, a, b, janela):
    """Transformacao que leva a nuvem dos quadros b..b+janela para o referencial da nuvem a-janela..a."""
    A = nuvem(scan, K, range(max(0, a - janela), a + 1))
    D = nuvem(scan, K, range(b, min(len(scan.ids), b + janela + 1)))
    T = np.eye(4)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    for thr in (10, 6, 4, 2):
        r = o3d.pipelines.registration.registration_icp(
            D, A, thr, T, est, o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T = np.array(r.transformation)
    return T, r


def encaixe(scan, K, ks, ref):
    arv = cKDTree(np.vstack([scan.pontos(k, K, passo=20)[0] for k in ref]))
    return {k: float(np.mean(arv.query(scan.pontos(k, K, passo=20)[0])[0] < 3.0)) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("projeto")
    ap.add_argument("scan")
    ap.add_argument("--ultimo-bom", type=int, required=True, help="indice do ultimo quadro antes da perda")
    ap.add_argument("--primeiro-bom", type=int, required=True, help="indice do primeiro quadro depois da perda")
    ap.add_argument("--janela", type=int, default=50)
    ap.add_argument("--saida", required=True)
    a = ap.parse_args()
    if os.path.exists(a.saida):
        sys.exit("destino ja existe: " + a.saida)
    a_, b_ = a.ultimo_bom, a.primeiro_bom

    work = tempfile.mkdtemp(prefix="perda_rastreio_")
    scan = RevoScan(os.path.join(a.projeto, "data", a.scan), work)
    K, err = scan.calibrar(0)
    print(f"K {np.round(K, 1)} (erro vs preview {err:.2f} mm)")
    T, r = estimar(scan, K, a_, b_, a.janela)
    ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
    print(f"ICP: fitness {r.fitness:.2f}, rmse {r.inlier_rmse:.2f} mm | translacao {np.linalg.norm(T[:3, 3]):.1f} mm "
          f"{np.round(T[:3, 3], 1)} | giro {ang:.2f} graus")
    if r.fitness < 0.6:
        sys.exit("fitness baixo: o ICP nao convergiu para um encaixe confiavel; nada gravado")

    ant = encaixe(scan, K, range(b_, b_ + 10), range(a_ - 22, a_ + 1))
    print("antes: encaixe <3mm dos quadros %d-%d com os %d-%d: %.0f%%" %
          (b_, b_ + 9, a_ - 22, a_, 100 * np.mean(list(ant.values()))))

    shutil.copytree(a.projeto, a.saida)
    db = scan.db + ".novo"
    scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
    shutil.copy(scan.db, db)
    cn = sqlite3.connect(db)
    for k, i in enumerate(scan.ids):
        if k >= b_:
            nova = T @ scan.T[k]
            bin_ = cn.execute("select bin from FrameInfo where id=?", (i,)).fetchone()[0]
            cn.execute("update FrameInfo set bin=? where id=?", (bin_[:16] + nova.astype("<f8").tobytes() + bin_[144:], i))
        if a_ < k < b_:
            cn.execute("delete from FrameRGBD where id=?", (i,))
    cn.commit()
    integ = cn.execute("pragma integrity_check").fetchone()[0]
    cn.execute("pragma wal_checkpoint(TRUNCATE)")
    cn.close()
    for ext in ("-wal", "-shm", "-journal"):
        if os.path.exists(db + ext):
            raise RuntimeError("sobrou arquivo de journal " + ext)
    with open(db, "r+b") as f:
        f.write(scan.sig)
    shutil.move(db, os.path.join(a.saida, "data", a.scan, "cache", "frames.dataset"))
    print(f"gravado (integridade {integ}); quadros {a_ + 1}-{b_ - 1} sem profundidade, quadros >= {b_} com pose corrigida")

    # conferencia lendo o arquivo gravado
    os.makedirs(os.path.join(work, "conf"), exist_ok=True)
    novo = RevoScan(os.path.join(a.saida, "data", a.scan), os.path.join(work, "conf"))
    dep = encaixe(novo, K, range(b_, b_ + 10), range(a_ - 22, a_ + 1))
    print("depois: encaixe <3mm dos quadros %d-%d com os %d-%d: %.0f%%" %
          (b_, b_ + 9, a_ - 22, a_, 100 * np.mean(list(dep.values()))))
    passo = np.linalg.norm(novo.T[b_, :3, 3] - novo.T[a_, :3, 3])
    print(f"distancia entre as poses {a_} e {b_}: {np.linalg.norm(scan.T[b_, :3, 3] - scan.T[a_, :3, 3]):.1f} mm -> {passo:.1f} mm "
          f"(passo normal ~10,7 mm por quadro, {b_ - a_} passos)")


if __name__ == "__main__":
    main()
