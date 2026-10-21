"""Une dois ou mais scans de uma mesma peca (escaneada em partes) num unico scan, no nivel dos frames.

    python pipeline/uniao_scans.py <pasta_do_projeto> --scans A B --saida <pasta_nova> [--nome <guid>]

1. Nuvem de cada scan a partir dos proprios quadros (referencial do mundo dos quadros, so a peca: use o
   projeto ja limpo por limpar_raw).
2. Alinhamento B -> A: FPFH + RANSAC (varias sementes) e ICP ponto-plano. O resultado so vale se as
   sementes concordam entre si e a sobreposicao medida nos dois sentidos for boa; senao aborta.
3. Grava um projeto .revo com UM scan cujo frames.dataset tem os quadros de A e os de B com a pose
   multiplicada pela transformacao (pose = camera->mundo, 4x4 float64 no `FrameInfo.bin`). O que mais o
   `bin` guarda (marcadores) esta em coordenadas da camera e nao muda. Depois e so fundir no Revo.
O projeto original nunca e tocado; a saida nao pode existir.
"""
import argparse
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile

import numpy as np
import open3d as o3d

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "revo_frames"))
from revo_projeto import RevoScan, SIG_SQL, SIG_REVO


def nuvem_dos_frames(scan, alvo_pontos=1_500_000):
    """Pontos no mundo dos quadros, amostrados uniformemente (todos os quadros com profundidade)."""
    for m in sorted(set(int(x) for x, p in zip(scan.modo, scan.tem_prof) if p)):
        scan.K[m], _ = scan.calibrar(m)
    n_total = sum(int(scan.profundidade(scan.ids[k])[3].size) for k in range(len(scan.ids)) if scan.tem_prof[k])
    passo = max(1, n_total // alvo_pontos)
    P = [scan.pontos(k, passo=passo)[0] for k in range(len(scan.ids)) if scan.tem_prof[k]]
    return np.vstack(P)


def _pcd(P, voxel):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    p = p.voxel_down_sample(voxel)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 3, max_nn=50))
    return p


def _pca(P):
    c = P.mean(0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    V = V[:, ::-1]
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return c, V


def _plano_dominante(P, voxel=0.25):
    """(n, d) do maior plano da nuvem (n.x + d = 0): numa peca em disco, e a face de extremidade."""
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(voxel)
    m, _ = p.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=3000)
    n = np.array(m[:3])
    return n / np.linalg.norm(n), float(m[3]) / np.linalg.norm(n)


def alinhar(PA, PB, voxel=1.0, sementes=5, lados="opostos"):
    """Candidatos de alinhamento B -> A, cada um refinado, e o escolhido segundo `lados`.

    Uma peca com simetria (disco, cilindro) tem solucoes quase igualmente boas: cada scan so ve a PROPRIA
    face de extremidade, entao a face de um nao pesa contra a do outro e o ICP nao distingue "faces no
    mesmo lado" de "faces em lados opostos" (peca virada). Por isso se gera tambem o giro de 180 graus em
    torno de cada eixo principal e se mede a separacao entre o maior plano de A e o de B alinhada:
      'opostos'  planos paralelos e separados por > 40% da extensao da uniao (peca virada: topo + base)
      'mesmo'    planos paralelos e a < 1 mm (mesma face para o mesmo lado)
      'auto'     o de maior encaixe, sem exigir nada.
    Aborta se nenhum candidato com bom encaixe tem a relacao pedida."""
    reg = o3d.pipelines.registration
    a, b = _pcd(PA, voxel), _pcd(PB, voxel)
    feat = lambda p: reg.compute_fpfh_feature(p, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 8, max_nn=200))
    fa, fb = feat(a), feat(b)
    inicios = []
    for sd in range(sementes):
        o3d.utility.random.seed(sd)
        r = reg.registration_ransac_based_on_feature_matching(
            b, a, fb, fa, True, voxel * 3, reg.TransformationEstimationPointToPoint(False), 3,
            [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 3)],
            reg.RANSACConvergenceCriteria(4_000_000, 0.9999))
        inicios.append(("ransac", r.transformation))
    cA, VA = _pca(PA)
    cB, VB = _pca(PB)
    for nome, F in (("pca+", np.diag([1, 1, 1])), ("pca-x", np.diag([1, -1, -1])),
                    ("pca-y", np.diag([-1, 1, -1])), ("pca-z", np.diag([-1, -1, 1]))):
        T = np.eye(4)
        T[:3, :3] = VA @ F @ VB.T
        T[:3, 3] = cA - T[:3, :3] @ cB
        inicios.append((nome, T))
    escalas = [(1.0, 2.0), (0.5, 1.0), (0.25, 0.5), (0.15, 0.3)]
    nuvens = {v: (_pcd(PA, v), _pcd(PB, v)) for v, _ in escalas}
    nA, dA = _plano_dominante(PA)
    nB, dB = _plano_dominante(PB)
    cands = []
    for origem, T in inicios:
        for v, lim in escalas:
            pa, pb = nuvens[v]
            ic = reg.registration_icp(pb, pa, lim, T, reg.TransformationEstimationPointToPlane(),
                                      reg.ICPConvergenceCriteria(max_iteration=100))
            T = ic.transformation
        if any(np.degrees(np.arccos(np.clip((np.trace((np.linalg.inv(T) @ c["T"])[:3, :3]) - 1) / 2, -1, 1))) < 1.0
               and np.linalg.norm((np.linalg.inv(T) @ c["T"])[:3, 3]) < 1.0 for c in cands):
            continue                                            # mesma solucao ja vista
        n2 = T[:3, :3] @ nB
        d2 = dB - n2 @ T[:3, 3]
        if n2 @ nA < 0:
            n2, d2 = -n2, -d2
        par = abs(float(n2 @ nA))
        sep = abs(dA - d2) if par > 0.9 else None
        Pb = PB @ T[:3, :3].T + T[:3, 3]
        z = np.r_[PA @ nA, Pb @ nA]
        ext = float(np.ptp(z))
        cands.append(dict(origem=origem, T=T, fitness=float(ic.fitness), rmse=float(ic.inlier_rmse),
                          separacao_planos_mm=None if sep is None else float(sep), extensao_uniao_mm=ext,
                          relacao=("indef" if sep is None else "mesmo" if sep < 1.0 else "opostos" if sep > 0.4 * ext else "intermediaria")))
    melhor_fit = max(c["fitness"] for c in cands)
    bons = [c for c in cands if c["fitness"] >= 0.5 * melhor_fit and c["rmse"] < 0.25]
    if lados != "auto":
        bons = [c for c in bons if c["relacao"] == lados]
    if not bons:
        raise SystemExit(f"nenhum candidato com bom encaixe tem faces '{lados}': " + json.dumps(
            [{k: v for k, v in c.items() if k != "T"} for c in cands], indent=1))
    esc = max(bons, key=lambda c: c["fitness"] if lados == "auto" else c["fitness"] - 3 * c["rmse"])
    T = esc["T"]
    # cobertura nos dois sentidos, com B ja alinhada (a 2 mm, em voxel de 1 mm)
    b2 = o3d.geometry.PointCloud(b)
    b2.transform(T)
    dab = np.asarray(a.compute_point_cloud_distance(b2))
    dba = np.asarray(b2.compute_point_cloud_distance(a))
    return dict(T=T, escolhido={k: v for k, v in esc.items() if k != "T"}, lados=lados,
                candidatos=[{k: v for k, v in c.items() if k != "T"} for c in cands],
                cobertura_A_por_B=float((dab < voxel * 2).mean()), cobertura_B_por_A=float((dba < voxel * 2).mean()))


def _bin_com_pose(b, T):
    """`FrameInfo.bin` com a pose (camera->mundo, 4x4 float64 nos bytes 16..144) levada por T."""
    pose = np.frombuffer(b[16:144], "<f8").reshape(4, 4)
    nova = np.ascontiguousarray(T @ pose, dtype="<f8")
    return b[:16] + nova.tobytes() + b[144:]


def unir(proj, gA, gB, T, saida, nome_scan):
    """Grava `saida` (projeto .revo com um scan `nome_scan`) com os quadros de A e os de B levados por T."""
    from revo_projeto import ler_ply, gravar_ply
    saida = os.path.abspath(saida)
    if os.path.exists(saida):
        sys.exit(f"destino ja existe: {saida}")
    pasta = os.path.join(saida, "data", nome_scan)
    shutil.copytree(os.path.join(proj, "data", gA), pasta)
    ds = os.path.join(pasta, "cache", "frames.dataset")
    with tempfile.TemporaryDirectory() as tmp:
        dsB = os.path.join(tmp, "B.sqlite")
        shutil.copy(os.path.join(proj, "data", gB, "cache", "frames.dataset"), dsB)
        for f in (ds, dsB):
            with open(f, "r+b") as h:
                assert h.read(16) == SIG_REVO
                h.seek(0)
                h.write(SIG_SQL)
        cn = sqlite3.connect(ds)
        cn.execute("pragma journal_mode=DELETE")
        cn.execute("attach ? as b", (dsB,))
        maxA = cn.execute("select max(id) from FrameInfo").fetchone()[0]
        off = maxA + 2 - cn.execute("select min(id) from b.FrameInfo").fetchone()[0]
        nA = cn.execute("select count(*) from FrameInfo").fetchone()[0]
        cn.execute("insert into FrameRGBD select id+?, version, type, width, height, data, weights from b.FrameRGBD", (off,))
        for (id_, ver, seq, drop, gap, pose, opt, mk, mkd, b) in cn.execute("select * from b.FrameInfo order by id").fetchall():
            cn.execute("insert into FrameInfo values (?,?,?,?,?,?,?,?,?,?)",
                       (id_ + off, ver, seq, drop, gap, pose, opt, mk, mkd, _bin_com_pose(b, T)))
        cn.commit()
        n = dict(A=nA, B=cn.execute("select count(*) from b.FrameInfo").fetchone()[0],
                 total=cn.execute("select count(*) from FrameInfo").fetchone()[0],
                 rgbd=cn.execute("select count(*) from FrameRGBD").fetchone()[0])
        cn.execute("detach b")
        cn.close()
    for ext in ("-wal", "-shm", "-journal"):
        if os.path.exists(ds + ext):
            os.remove(ds + ext)
    with open(ds, "r+b") as h:
        h.write(SIG_REVO)
    # preview 3D: A + B transformada (posicao e normal)
    hA, pA = ler_ply(os.path.join(pasta, "raw_preview.ply"))
    _, pB = ler_ply(os.path.join(proj, "data", gB, "raw_preview.ply"))
    pB = pB.copy()
    pB[:, :3] = pB[:, :3] @ T[:3, :3].T + T[:3, 3]
    if pB.shape[1] >= 6:
        pB[:, 3:6] = pB[:, 3:6] @ T[:3, :3].T
    gravar_ply(os.path.join(pasta, "raw_preview.ply"), hA, np.vstack([pA, pB]))
    # metadados do scan
    rv = os.path.join(pasta, "property.rvproj")
    meta = json.load(open(rv, encoding="utf-8"))
    meta["guid"] = nome_scan
    if "vf_count" in meta:
        meta["vf_count"] = n["rgbd"]
    json.dump(meta, open(rv, "w", encoding="utf-8"), indent=4)
    # projeto .revo com um scan so
    origem = [f for f in os.listdir(proj) if f.endswith(".revo")][0]
    r = json.load(open(os.path.join(proj, origem), encoding="utf-8"))
    no = next(x for x in r["nodes"] if x["guid"] == gA)
    r.update(guid=str(__import__("uuid").uuid4()), name=nome_scan, edit_time=str(int(__import__("time").time())),
             nodes=[dict(no, guid=nome_scan, name=nome_scan)])
    r["size"] = str(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(os.path.join(saida, "data")) for f in fs))
    json.dump(r, open(os.path.join(saida, nome_scan + ".revo"), "w", encoding="utf-8"), indent=4)
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--scans", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--saida")
    ap.add_argument("--nome", help="nome/guid do scan unido (padrao: <A>_uniao)")
    ap.add_argument("--lados", choices=("opostos", "mesmo", "auto"), default="opostos",
                    help="relacao entre as faces de A e B: opostos = peca virada (topo + base); mesmo = mesma face")
    ap.add_argument("--so-alinhar", action="store_true")
    a = ap.parse_args()
    proj = os.path.abspath(a.projeto)
    with tempfile.TemporaryDirectory() as tmp:
        sA = RevoScan(os.path.join(proj, "data", a.scans[0]), tmp)
        sB = RevoScan(os.path.join(proj, "data", a.scans[1]), tmp)
        PA, PB = nuvem_dos_frames(sA), nuvem_dos_frames(sB)
        print(f"pontos A {len(PA):,} B {len(PB):,}", flush=True)
        r = alinhar(PA, PB, lados=a.lados)
        T = r.pop("T")
        print(json.dumps(r, indent=1))
        print(np.array2string(T, precision=4, suppress_small=True))
        np.save(os.path.join(tempfile.gettempdir(), "uniao_T.npy"), T)
        sA.c.close()
        sB.c.close()
    if a.so_alinhar:
        return
    if not a.saida:
        sys.exit("falta --saida")
    if r["cobertura_A_por_B"] < 0.3 and r["cobertura_B_por_A"] < 0.3:
        sys.exit("pouca sobreposicao entre as partes: alinhamento nao confiavel, nada foi gravado")
    nome = a.nome or a.scans[0] + "_uniao"
    n = unir(proj, a.scans[0], a.scans[1], T, a.saida, nome)
    json.dump(dict(r, T=T.tolist(), quadros=n), open(os.path.join(os.path.abspath(a.saida), "uniao.json"), "w"), indent=1)
    print("quadros:", n, "| saida:", os.path.abspath(a.saida))


if __name__ == "__main__":
    main()
