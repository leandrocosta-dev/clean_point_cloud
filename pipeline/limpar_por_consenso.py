"""Limpeza dos quadros por CONSENSO entre scans alinhados (antes da fusao).

    python pipeline/limpar_por_consenso.py <projeto_limpo> <pasta_segmentos> <segmentos.json> --saida <projeto_novo>

Com todos os scans no mesmo referencial (poses por segmento de `alinhar_direto.py`), cada pixel de cada quadro
vira um ponto 3D comum e so fica se:
  (a) pelo menos `--apoio-outros` OUTROS scans tem um ponto a ate `--tol` mm (distancia euclidiana real, metodo
      padrao): fantasmas, reflexos e cenario que so um ou dois scans veem saem; e
  (b) esta a ate `--raio` mm (euclidiano) do NUCLEO da peca (centroides vistos por >= `--nucleo` scans a `--tol`):
      chapas, suportes e hastes fixos em relacao a peca (passam em (a)) mas fora do casco saem.
Quadros que ficam com menos de `--min-pontos` pontos sao apagados. Poses, calibracao e raw_preview nao mudam.

Portoes (o script PARA antes de gravar a entrega, com diagnostico): `--nucleo` maior que o numero de scans,
`--apoio-outros` >= numero de scans, nucleo vazio. Scans que perderiam mais que `--perda-max` dos pontos sao
gravados mas marcados REVISAR e o codigo de saida e 2 (resultado parcial, nao aprovar sem olhar).
`--metodo voxel` reproduz a regra antiga (duas grades de 2 mm; casa ate 3,46 mm na diagonal; nucleo 26-conexo),
agora com contagem de bits completa (ate 63 scans) e sem dilatacao quando --raio 0.
O projeto de origem nunca e alterado; o destino nao pode existir.
Todo o calculo (indices de vizinhos, contagem de apoio, grades, mascaras de bits) roda na GPU; ficam na CPU so o
SQLite, o empacotamento dos bits dos quadros e os arquivos .npz/.json.
"""

import argparse
import json
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
import gpu  # noqa: E402
from recursos import baixa_prioridade, orcamento_threads  # noqa: E402
from revo_projeto import ArvoreGPU, RevoScan, _morf  # noqa: E402

VOX = 2.0
OFF = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # duas grades deslocadas de 1 mm (metodo voxel)
POP = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)


class ConsensoInviavel(ValueError):
    """Parametros ou dados que nao permitem consenso: nada deve ser gravado."""


def _eh_cupy(a):
    return type(a).__module__.split(".")[0] == "cupy"


def chaves(P, o, mn):
    """Chave int64 do voxel de 2 mm de cada ponto (numpy ou cupy, na mesma grade deslocada por `o`)."""
    m = gpu.xp() if _eh_cupy(P) else np
    if m is not np:
        mn, o = gpu.para_gpu(mn), gpu.para_gpu(o)
    k = m.floor((P - mn + o) / VOX).astype(m.int64)
    return (k[:, 0] * 400003 + k[:, 1]) * 400031 + k[:, 2]


_POP_G = {}


def popcount(m):
    """Bits 1 em inteiros de 64 bits (antes contava so 48: scans 49+ sumiam da contagem). numpy ou cupy."""
    xm = gpu.xp() if _eh_cupy(m) else np
    m = xm.asarray(m).astype(xm.uint64)
    if hasattr(xm, "bitwise_count"):
        return xm.bitwise_count(m).astype(xm.int64)
    tab = POP if xm is np else _POP_G.setdefault("t", gpu.para_gpu(POP))
    tot = xm.zeros(m.shape, xm.int64)
    for s in (0, 16, 32, 48):
        tot += tab[((m >> xm.uint64(s)) & xm.uint64(0xFFFF)).astype(xm.int64)]
    return tot


def validar(n_scans, nucleo, apoio_outros, metodo):
    if n_scans < 2:
        raise ConsensoInviavel(f"consenso exige >= 2 scans alinhados; ha {n_scans}")
    if nucleo > n_scans:
        raise ConsensoInviavel(
            f"--nucleo {nucleo} > {n_scans} scans: nenhum ponto pode formar o nucleo e TODA a peca "
            f"seria apagada. Use --nucleo <= {n_scans} (decisao de confianca: nao foi reduzido sozinho)"
        )
    if apoio_outros >= n_scans:
        raise ConsensoInviavel(
            f"--apoio-outros {apoio_outros} exige mais scans do que os {n_scans - 1} outros disponiveis"
        )
    if metodo == "voxel" and n_scans > 63:
        raise ConsensoInviavel("metodo voxel usa mascara de 64 bits: no maximo 63 scans (use --metodo euclidiano)")


def nuvens_alinhadas(pasta, segjson, scans):
    ind = json.load(open(os.path.join(pasta, "segmentos_indice.json")))
    pj = json.load(open(segjson))["segmentos"]
    todos = []
    for g in scans:
        X = []
        for s in ind[g]:
            T = np.array(next(x for x in pj[g] if x["seg"] == s["seg"])["T"])
            C = gpu.para_gpu(np.load(s["arquivo"]), np.float64)
            X.append(C @ gpu.para_gpu(T[:3, :3].T) + gpu.para_gpu(T[:3, 3]))  # transformacao na GPU
        todos.append(gpu.para_cpu(gpu.xp().vstack(X)) if X else np.empty((0, 3)))
    return todos


def contar_apoio(P, arvores, tol, ignorar=None, alvo=None, workers=None):
    """Numero de scans (arvores `ArvoreGPU`) com ponto a <= tol de cada ponto (GPU; P numpy ou cupy, saida do mesmo
    tipo). Com `alvo`, para de consultar os pontos que ja atingiram o alvo (resultado identico para o teste
    cont >= alvo, bem mais rapido). `workers` so existe por compatibilidade (a GPU nao usa threads de CPU)."""
    m = gpu.xp()
    na_gpu = _eh_cupy(P)
    P = gpu.para_gpu(P, m.float64)
    cont = m.zeros(len(P), m.int32)
    ativos = m.arange(len(P))
    for b, arv in enumerate(arvores):
        if b == ignorar or arv is None or not len(ativos):
            continue
        d, _ = arv.query(P[ativos], distance_upper_bound=tol)
        cont[ativos[m.isfinite(d)]] += 1
        if alvo is not None:
            ativos = ativos[cont[ativos] < alvo]
    return cont if na_gpu else gpu.para_cpu(cont)


def construir_euclidiano(todos, nucleo, tol, saida_npz, workers=None):
    m = gpu.xp()
    arv = [ArvoreGPU(X) if len(X) else None for X in todos]
    U = gpu.para_gpu(np.vstack([X for X in todos if len(X)]), m.float64)
    cont = contar_apoio(U, arv, tol, alvo=nucleo)
    N = gpu.para_cpu(U[cont >= nucleo])
    np.savez(saida_npz, metodo="euclidiano", nucleo_pts=N, **{f"scan{b}": X for b, X in enumerate(todos)})
    return dict(centroides=int(len(U)), nucleo=int(len(N)), fracao_nucleo=float(len(N) / max(1, len(U))))


def construir_voxel(todos, nucleo, raio, saida_npz, vox_nucleo=3.0):
    m = gpu.xp()
    tg = [gpu.para_gpu(X, m.float64) for X in todos]
    mn = gpu.para_cpu(m.vstack([x.min(0, keepdims=True) for x in tg if len(x)]).min(0)) - 40
    out = dict(mn=mn)
    for gi, o in enumerate(OFF):
        # chaves unicas de cada scan; o OR dos bits por voxel vira posicao em `u` + OR do bit do scan (cada scan so
        # aparece uma vez por voxel, entao a ordem nao importa)
        ks = [m.unique(chaves(X, o, gpu.para_gpu(mn))) for X in tg]
        u = m.unique(m.concatenate(ks))
        mk = m.zeros(len(u), m.uint64)
        for bit, k in enumerate(ks):
            pos = m.searchsorted(u, k)
            mk[pos] |= m.uint64(1 << bit)
        out[f"u{gi}"] = gpu.para_cpu(u)
        out[f"m{gi}"] = gpu.para_cpu(mk)
        if gi == 0:
            u0, m0 = u, mk
    forte = u0[popcount(m0) >= nucleo]
    kz = forte % 400031
    t = forte // 400031
    ky = t % 400003
    kx = t // 400003
    cen = (m.stack([kx, ky, kz], 1) + 0.5) * VOX + gpu.para_gpu(mn)
    lo = mn
    dims = np.ceil((gpu.para_cpu(m.vstack(tg).max(0)) + 40 - lo) / vox_nucleo).astype(int) + 2
    G = m.zeros(tuple(int(x) for x in dims), bool)
    idx = m.floor((cen - gpu.para_gpu(lo)) / vox_nucleo).astype(m.int64)
    if len(idx):
        G[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    it = int(round(raio / vox_nucleo))
    if it > 0:  # iterations=0 no scipy repete ate encher o volume
        G = _morf("binary_dilation", G, structure=m.ones((3, 3, 3), bool), iterations=it)
    G = gpu.para_cpu(G)
    np.savez(
        saida_npz,
        metodo="voxel",
        mn=mn,
        u0=out["u0"],
        m0=out["m0"],
        u1=out["u1"],
        m1=out["m1"],
        lo=lo,
        vox_nucleo=vox_nucleo,
        nucleo_shape=np.array(G.shape),
    )
    np.save(saida_npz + ".nucleo.npy", np.packbits(G.ravel()))
    return dict(nucleo=int(len(forte)), voxels_nucleo_dilatado=int(G.sum()))


class Decisor:
    """Decide ok/nao para pontos ja no referencial comum (carregado uma vez por processo; indices e grades na GPU)."""

    def __init__(self, npz, bit, apoio_outros, tol, raio):
        z = np.load(npz)
        self.metodo = str(z["metodo"])
        self.bit, self.apoio, self.tol, self.raio = bit, apoio_outros, tol, raio
        if self.metodo == "euclidiano":
            n = len([k for k in z.files if k.startswith("scan")])
            self.arv = [ArvoreGPU(z[f"scan{b}"]) if len(z[f"scan{b}"]) else None for b in range(n)]
            N = z["nucleo_pts"]
            self.arv_nucleo = ArvoreGPU(N) if len(N) else None
        else:
            shp = tuple(z["nucleo_shape"])
            G = np.unpackbits(np.load(npz + ".nucleo.npy"))[: int(np.prod(shp))].astype(bool).reshape(shp)
            self.G = gpu.para_gpu(G)
            self.mn, self.lo, self.vn = gpu.para_gpu(z["mn"]), gpu.para_gpu(z["lo"]), float(z["vox_nucleo"])
            self.masc = [(gpu.para_gpu(z["u0"]), gpu.para_gpu(z["m0"])), (gpu.para_gpu(z["u1"]), gpu.para_gpu(z["m1"]))]

    def __call__(self, Pg):
        """Pg numpy ou cupy (n, 3); devolve bool do mesmo tipo."""
        m = gpu.xp()
        na_gpu = _eh_cupy(Pg)
        Pg = gpu.para_gpu(Pg, m.float64)
        if self.metodo == "euclidiano":
            if self.arv_nucleo is None:
                ok = m.zeros(len(Pg), bool)
            else:
                d, _ = self.arv_nucleo.query(Pg, distance_upper_bound=self.raio)
                ok = m.isfinite(d)
                idx = m.flatnonzero(ok)
                if len(idx):
                    c = contar_apoio(Pg[idx], self.arv, self.tol, ignorar=self.bit, alvo=self.apoio)
                    ok[idx] = c >= self.apoio
            return ok if na_gpu else gpu.para_cpu(ok)
        outros = ~(m.uint64(1) << m.uint64(self.bit))
        sup = m.zeros(len(Pg), dtype=m.uint64)
        for o, (u, mk) in zip(OFF, self.masc):
            kk = chaves(Pg, o, self.mn)
            j = m.searchsorted(u, kk)
            j[j >= len(u)] = len(u) - 1
            sup |= m.where(u[j] == kk, mk[j], m.uint64(0))
        ok = popcount(sup & outros) >= self.apoio
        v = m.floor((Pg - self.lo) / self.vn).astype(m.int64)
        forma = gpu.para_gpu(self.G.shape)
        dentro = ((v >= 0) & (v < forma)).all(axis=1)
        vc = m.clip(v, 0, forma - 1)
        ok = ok & dentro & self.G[vc[:, 0], vc[:, 1], vc[:, 2]]
        return ok if na_gpu else gpu.para_cpu(ok)


def _limpar_scan(args):
    proj, destino, g, bit, segs, npz, apoio_outros, min_pontos, tol, raio = args
    baixa_prioridade()
    decide = Decisor(npz, bit, apoio_outros, tol, raio)
    stats = dict(scan=g, antes=0, depois=0, quadros=0, apagados=0)
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(os.path.join(proj, "data", g), tmp)
        try:
            for m in sorted(set(int(x) for x, p in zip(scan.modo, scan.tem_prof) if p)):
                scan.K[m], _ = scan.calibrar(m)
            db = os.path.join(tmp, g + ".limpo")
            scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
            shutil.copy(scan.db, db)
            cn = sqlite3.connect(db)
            for k, i in enumerate(scan.ids):
                if not scan.tem_prof[k]:
                    continue
                seg = next((s for s in segs if s["k0"] <= k <= s["k1"]), None)
                if seg is None:
                    continue
                T = np.array(seg["T"])
                w, h, mask, zz, quatro = scan.profundidade(i)
                P, pix = scan.pontos(k, na_gpu=True)
                Pg = P @ gpu.para_gpu(T[:3, :3].T) + gpu.para_gpu(T[:3, 3])
                ok = gpu.para_cpu(decide(Pg))
                stats["quadros"] += 1
                stats["antes"] += len(pix)
                if ok.sum() < min_pontos:
                    cn.execute("delete from FrameInfo where id=?", (i,))
                    cn.execute("delete from FrameRGBD where id=?", (i,))
                    stats["apagados"] += 1
                    continue
                nova = np.zeros(w * h, bool)
                nova[pix[ok]] = True
                dados = (
                    struct.pack("<III", w, h, int(ok.sum()))
                    + np.packbits(nova, bitorder="little").tobytes()
                    + quatro
                    + zz[ok].astype("<f4").tobytes()
                )
                cn.execute("update FrameRGBD set data=? where id=?", (dados, i))
                stats["depois"] += int(ok.sum())
            cn.commit()
            cn.execute("pragma wal_checkpoint(TRUNCATE)")
            integ = cn.execute("pragma integrity_check").fetchone()[0]
            cn.close()
            for ext in ("-wal", "-shm", "-journal"):
                if os.path.exists(db + ext):
                    os.remove(db + ext)
            with open(db, "r+b") as f:
                f.write(scan.sig)
            os.makedirs(os.path.join(destino, "data", g, "cache"), exist_ok=True)
            shutil.copyfile(db, os.path.join(destino, "data", g, "cache", "frames.dataset"))
            stats["integridade"] = integ
        finally:
            scan.c.close()
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("pasta_segmentos")
    ap.add_argument("segmentos")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--apoio-outros", type=int, default=2)
    ap.add_argument("--nucleo", type=int, default=6)
    ap.add_argument("--raio", type=float, default=15.0, help="distancia maxima ao nucleo (mm, euclidiana)")
    ap.add_argument("--tol", type=float, default=1.0, help="distancia de apoio entre scans (mm, euclidiana)")
    ap.add_argument("--metodo", choices=["euclidiano", "voxel"], default="euclidiano")
    ap.add_argument("--min-pontos", type=int, default=1500)
    ap.add_argument(
        "--perda-max", type=float, default=0.6, help="fracao maxima removida por scan antes de marcar REVISAR"
    )
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.tol <= 0 or a.raio < 0 or a.apoio_outros < 1 or a.nucleo < 1 or not 0 < a.perda_max <= 1:
        ap.error("tol > 0, raio >= 0, apoio-outros >= 1, nucleo >= 1 e 0 < perda-max <= 1")
    baixa_prioridade()
    proj, saida = os.path.abspath(a.projeto), os.path.abspath(a.saida)
    if os.path.exists(saida) and (not os.path.isdir(saida) or os.listdir(saida)):  # pasta vazia pode ser reaproveitada
        sys.exit(f"destino ja existe: {saida}")
    pj = json.load(open(a.segmentos))
    scans = sorted(
        g for g in json.load(open(os.path.join(a.pasta_segmentos, "segmentos_indice.json"))) if g in pj["segmentos"]
    )
    try:
        validar(len(scans), a.nucleo, a.apoio_outros, a.metodo)
    except ConsensoInviavel as e:
        sys.exit(f"CONSENSO INVIAVEL: {e}")
    jobs, por_job = orcamento_threads(min(a.jobs, len(scans)))
    t0 = time.time()
    tmp_npz = tempfile.mkdtemp(prefix="consenso_")
    npz = os.path.join(tmp_npz, "consenso.npz")
    todos = nuvens_alinhadas(a.pasta_segmentos, a.segmentos, scans)
    if a.metodo == "euclidiano":
        info = construir_euclidiano(todos, a.nucleo, a.tol, npz, workers=jobs * por_job)
        vazio = info["nucleo"] == 0
    else:
        info = construir_voxel(todos, a.nucleo, a.raio, npz)
        vazio = info["nucleo"] == 0
    print(f"consenso de {len(scans)} scans ({a.metodo}); {info} ({time.time() - t0:.0f}s)", flush=True)
    if vazio:
        shutil.rmtree(tmp_npz, ignore_errors=True)
        sys.exit(
            f"CONSENSO INVIAVEL: nucleo vazio (nenhum ponto visto por >= {a.nucleo} scans a {a.tol:g} mm). "
            "Sem sobreposicao suficiente ou poses erradas: conferir alinhamento antes de limpar. Nada foi gravado."
        )
    os.makedirs(saida, exist_ok=True)
    for g in os.listdir(os.path.join(proj, "data")):
        shutil.copytree(
            os.path.join(proj, "data", g),
            os.path.join(saida, "data", g),
            ignore=shutil.ignore_patterns("frames.dataset"),
        )
    for f in os.listdir(proj):
        if os.path.isfile(os.path.join(proj, f)):
            shutil.copy2(os.path.join(proj, f), os.path.join(saida, f))
    shutil.copy2(npz, os.path.join(saida, "consenso.npz"))
    if os.path.exists(npz + ".nucleo.npy"):
        shutil.copy2(npz + ".nucleo.npy", os.path.join(saida, "consenso.npz.nucleo.npy"))
    npz = os.path.join(saida, "consenso.npz")
    shutil.rmtree(tmp_npz, ignore_errors=True)
    tarefas = [
        (proj, saida, g, scans.index(g), pj["segmentos"][g], npz, a.apoio_outros, a.min_pontos, a.tol, a.raio)
        for g in scans
    ]
    rel, revisar = [], []
    with ProcessPoolExecutor(jobs) as ex:
        for st in ex.map(_limpar_scan, tarefas):
            frac = st["depois"] / max(1, st["antes"])
            st["status"] = "limpo" if frac >= 1 - a.perda_max and st.get("integridade") == "ok" else "revisar"
            if st["status"] == "revisar":
                revisar.append(st["scan"])
            rel.append(st)
            print(
                f"  {st['scan']}: {st['antes']:,} -> {st['depois']:,} pontos ({frac * 100:.0f}% mantidos), "
                f"{st['apagados']} de {st['quadros']} quadros apagados, integridade {st.get('integridade')}"
                + ("  << REVISAR" if st["status"] == "revisar" else ""),
                flush=True,
            )
    for g in os.listdir(os.path.join(proj, "data")):  # scans fora do consenso (ex.: outra calibracao) vao como estavam
        if g not in scans and not os.path.exists(os.path.join(saida, "data", g, "cache", "frames.dataset")):
            os.makedirs(os.path.join(saida, "data", g, "cache"), exist_ok=True)
            shutil.copy2(
                os.path.join(proj, "data", g, "cache", "frames.dataset"),
                os.path.join(saida, "data", g, "cache", "frames.dataset"),
            )
    json.dump(
        dict(parametros=vars(a), nucleo=info, scans=rel, revisar=revisar, estado="parcial" if revisar else "completo"),
        open(os.path.join(saida, "limpeza_consenso.json"), "w"),
        indent=1,
    )
    tot_a, tot_d = sum(s["antes"] for s in rel), sum(s["depois"] for s in rel)
    print(
        f"total: {tot_a:,} -> {tot_d:,} pontos ({tot_d / max(1, tot_a) * 100:.0f}% mantidos) em {time.time() - t0:.0f}s"
    )
    if revisar:
        print(
            f"RESULTADO PARCIAL: {len(revisar)} scan(s) perderam mais de {a.perda_max * 100:.0f}% ou falharam na "
            f"integridade: {revisar}. Nao aprovar sem inspecao.",
            flush=True,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
