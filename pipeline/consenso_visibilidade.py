"""Limpeza dos quadros por VISIBILIDADE entre scans alinhados: so sai o que outra camera viu ATRAVES.

    python pipeline/consenso_visibilidade.py <projeto_limpo> --poses poses.json --saida <projeto_novo>
        [--voxel 0.5] [--tol 0.5] [--passo-quadros 6] [--min-contra N] [--min-livre-quadros 2]
        [--calibracoes calibracoes_fixas.json]

poses.json: {guid: 4x4} leva cada scan (referencial dos seus quadros) ao referencial comum (ex.: Merge Alignment).
1. Candidatos: centroides de --voxel mm de TODOS os pixels de cada scan, no referencial comum.
2. Cada candidato do scan s e reprojetado nos quadros de evidencia (1 a cada --passo-quadros) de cada OUTRO scan o:
   - confirmado: a profundidade medida no pixel bate com a do candidato (|D - z| <= --tol);
   - espaco livre: a janela 3x3 inteira e valida e a MENOR profundidade dela esta atras do candidato (> z + tol):
     o raio da camera de o passou pelo candidato e achou superficie mais longe;
   - inconclusivo: pixel fora da imagem/sem dado (sem cobertura) ou superficie na frente (oclusao).
   O scan o CONTRADIZ o candidato se >= --min-livre-quadros quadros dele viram espaco livre ali.
3. Remove so candidato contradito por >= --min-contra scans (padrao min(2, n-1)) e confirmado por NENHUM.
   Falta de observacao nunca remove (nervura vista por uma posicao so fica, como inconclusiva).
4. Grava COPIA nova do projeto: pixels dos candidatos removidos apagados de cada quadro (poses intactas);
   quadro que fica com < --min-pix-quadro pixels perde a profundidade (FrameRGBD), sem mexer em FrameInfo
   (indices de quadro nao mudam). raw_preview filtrado pela mesma regra. Relatorio JSON e PLYs
   (removidos / inconclusivos / confirmados) por scan, no referencial comum, para inspecao 3D.
Roda na GPU (CuPy, `gpu.xp()`): votacao, voxels e remocao dos pixels. Sem CUDA aborta (ErroGPU); so com REVO_PERMITIR_CPU=1
usa numpy/scipy (depuracao).
Limites: scans repetidos da MESMA pose de apoio contam como scans distintos (podem confirmar a mesa entre si);
pose errada acima de --tol vira contradicao falsa; calibracao por modo vem do ajuste ao raw_preview.
O projeto de origem nunca e alterado; o destino nao pode existir.
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

import numpy as np
from comum import configurar_log

AQUI = os.path.dirname(os.path.abspath(__file__))
import gpu  # noqa: E402
from recursos import baixa_prioridade, orcamento_threads  # noqa: E402
from registro_gpu import transformar  # noqa: E402

OFF = 1 << 20  # deslocamento dos indices de voxel (chave int64: 21 bits por eixo, +-2^20 voxels)
NB = 21
LOTE_PONTOS = 16_000_000  # pontos acumulados na GPU antes de reduzir a voxels (~0,5 GB)


# ---------------------------------------------------------------- nucleo numerico (testavel sem SQLite)
_LOGGER = configurar_log(__name__)


def _ndimage():
    """scipy.ndimage (CPU, so com REVO_PERMITIR_CPU) ou cupyx.scipy.ndimage (GPU)."""
    if gpu.em_gpu():
        from cupyx.scipy import ndimage
    else:
        from scipy import ndimage
    return ndimage


def chaves_voxel(P, voxel):
    """Chave int64 do voxel de cada ponto (21 bits por eixo, deslocada por OFF); sai no mesmo lado da entrada (numpy/cupy)."""
    xp = gpu.xp()
    saida_gpu = gpu._eh_gpu(P)
    k = xp.floor(gpu.para_gpu(P, xp.float64) / voxel).astype(xp.int64) + OFF
    if k.size and (int(k.min()) < 0 or int(k.max()) >= (1 << NB)):
        raise ValueError("coordenadas fora da faixa da chave de voxel")
    chave = (k[:, 0] << (2 * NB)) | (k[:, 1] << NB) | k[:, 2]
    return chave if saida_gpu else gpu.para_cpu(chave)


class Vista:
    """Um quadro de evidencia: profundidade D (h, w; NaN = sem dado), K = (fx, fy, cx, cy) e T camera->ref.
    Convencao da camera igual a RevoScan.pontos: P_cam = ((u-cx)z/fx, -(v-cy)z/fy, -z)."""

    def __init__(self, D, K, T, scan):
        xp = gpu.xp()
        nd = _ndimage()
        D = gpu.para_gpu(D, xp.float32)
        val = xp.isfinite(D) & (D > 0)
        self.h, self.w = D.shape
        self.D = xp.where(val, D, xp.nan).astype(xp.float32)
        self.Dmin = nd.minimum_filter(xp.where(val, D, xp.inf).astype(xp.float32), size=3, mode="constant", cval=np.inf)
        self.val9 = nd.minimum_filter(val.astype(xp.uint8), size=3, mode="constant", cval=0).astype(bool)
        self.K = tuple(float(x) for x in K)
        self.T = np.asarray(T, float)
        self.scan = scan


def _para_xp(vistas, xp=None):
    """Compatibilidade: as vistas ja nascem no dispositivo (GPU). Devolve a mesma lista."""
    return vistas


def votar_vista(C, v, tol, xp=None):
    """Para candidatos C (N x 3, no referencial comum): (confirmado, livre, coberto) booleanos."""
    xp = xp or gpu.xp()
    R, t = v.T[:3, :3], v.T[:3, 3]
    Xc = (C - xp.asarray(t)) @ xp.asarray(R)  # R^T (C - t) por linha
    z = -Xc[:, 2]
    fx, fy, cx, cy = v.K
    zs = xp.where(z > 1e-6, z, 1.0)
    u = xp.rint(fx * Xc[:, 0] / zs + cx).astype(xp.int64)
    vv = xp.rint(cy - fy * Xc[:, 1] / zs).astype(xp.int64)
    dentro = (z > 1e-6) & (u >= 0) & (u < v.w) & (vv >= 0) & (vv < v.h)
    ui = xp.where(dentro, u, 0)
    vi = xp.where(dentro, vv, 0)
    D, Dmin, val9 = v.D, v.Dmin, v.val9
    d = D[vi, ui]
    dv = dentro & xp.isfinite(d)
    conf = dv & (xp.abs(d - z) <= tol)
    livre = dentro & val9[vi, ui] & (Dmin[vi, ui] > z + tol)
    return conf, livre, dv


def avaliar_candidatos(C, scan_c, vistas, tol=0.5, min_livre_quadros=2, xp=None):
    """Conta, por candidato, quantos OUTROS scans confirmam, contradizem (>= min_livre_quadros quadros com
    espaco livre) e cobrem (algum pixel valido). Devolve dict de arrays numpy."""
    xp = xp or gpu.xp()
    Cx = xp.asarray(gpu.para_cpu(C), dtype=xp.float64)
    n = len(C)
    por_scan = {}
    for v in vistas:
        if v.scan == scan_c:
            continue
        por_scan.setdefault(v.scan, []).append(v)
    conf_s = xp.zeros(n, xp.int32)
    contra_s = xp.zeros(n, xp.int32)
    cob_s = xp.zeros(n, xp.int32)
    for o, lista in por_scan.items():
        conf = xp.zeros(n, bool)
        cob = xp.zeros(n, bool)
        nlivre = xp.zeros(n, xp.int32)
        for v in lista:
            c, lv, dv = votar_vista(Cx, v, tol, xp)
            conf |= c
            cob |= dv
            nlivre += lv.astype(xp.int32)
        conf_s += conf.astype(xp.int32)
        contra_s += (nlivre >= min_livre_quadros).astype(xp.int32)
        cob_s += cob.astype(xp.int32)
    return dict(
        confirmam=gpu.para_cpu(conf_s),
        contradizem=gpu.para_cpu(contra_s),
        cobrem=gpu.para_cpu(cob_s),
        outros_scans=len(por_scan),
    )


def decidir(votos, min_contra):
    """Estados: 2 = confirmado, 1 = contradito (remover), 0 = inconclusivo."""
    rem = (votos["contradizem"] >= min_contra) & (votos["confirmam"] == 0)
    est = np.zeros(len(rem), np.int8)
    est[votos["confirmam"] > 0] = 2
    est[rem] = 1
    return est


def gravar_ply_pontos(caminho, P):
    P = np.asarray(P, "<f4").reshape(-1, 3)
    with open(caminho, "wb") as f:
        f.write(
            (
                f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
                "property float x\nproperty float y\nproperty float z\nend_header\n"
            ).encode("ascii")
        )
        f.write(P.tobytes())


# ---------------------------------------------------------------- leitura dos scans
def validar_calibracao_fixa(calibracao, modos):
    """Valida K por modo sem inicializar CUDA; todos os modos usados precisam estar cobertos."""
    if not isinstance(calibracao, dict):
        raise ValueError("calibracao fixa deve mapear modo -> {K, erro_preview_mm}")
    normalizada = {}
    for chave, dados in calibracao.items():
        try:
            modo = int(chave)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("modo de calibracao deve ser inteiro nao negativo") from exc
        if str(modo) != str(chave) or modo < 0:
            raise ValueError("modo de calibracao deve ser inteiro nao negativo")
        if modo in normalizada or not isinstance(dados, dict):
            raise ValueError("modo de calibracao repetido ou invalido")
        try:
            K = np.asarray(dados.get("K"), float)
            erro = float(dados["erro_preview_mm"]) if dados.get("erro_preview_mm") is not None else None
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"calibracao do modo {modo}: valores numericos invalidos") from exc
        if K.shape != (4,) or not np.isfinite(K).all() or not (K[:2] > 0).all():
            raise ValueError(f"calibracao do modo {modo}: K deve ter fx/fy positivos e quatro valores finitos")
        if erro is not None and (not np.isfinite(erro) or erro < 0):
            raise ValueError(f"calibracao do modo {modo}: erro invalido")
        normalizada[modo] = dict(K=K.tolist(), erro_preview_mm=erro)
    faltam = set(modos) - set(normalizada)
    if faltam:
        raise ValueError(f"calibracao fixa nao cobre os modos dos quadros: {sorted(faltam)}")
    return normalizada


def carregar_scan(proj, g, T_ref, voxel, passo, tmp, calibracao_fixa=None):
    """Candidatos (chaves e centroides no ref), vistas de evidencia, K por modo e contagem de pixels."""
    from revo_projeto import RevoScan

    sc = RevoScan(os.path.join(proj, "data", g), tmp)
    try:
        modos = sorted(set(int(m) for m, p in zip(sc.modo, sc.tem_prof) if p))
        if calibracao_fixa is None:
            calib = {}
            for m in modos:
                K, e = sc.calibrar(m)
                calib[m] = dict(K=list(K), erro_preview_mm=e)
        else:
            calib = validar_calibracao_fixa(calibracao_fixa, modos)
            for m in modos:
                sc.K[m] = tuple(calib[m]["K"])
        xp = gpu.xp()
        ks_all, soma_all, n_all = [], [], []
        buffer = []  # pontos no referencial comum; reduzidos a voxels em lotes (menos lancamentos de kernel)

        def esvaziar():
            if not buffer:
                return
            Pr = xp.concatenate(buffer)
            buffer.clear()
            u, inv, cnt_k = gpu.agrupar(chaves_voxel(Pr, voxel))
            ks_all.append(u)
            soma_all.append(gpu.somar_por_grupo(Pr, inv, len(u)))
            n_all.append(cnt_k)

        vistas = []
        n_pix = 0
        com = [k for k in range(len(sc.ids)) if sc.tem_prof[k]]
        evid = set(com[:: max(1, passo)])
        for k in com:
            P, pix = sc.pontos(k)
            if P is None or not len(P):
                continue
            n_pix += len(P)
            buffer.append(transformar(gpu.para_gpu(P, xp.float64), T_ref))
            if sum(len(x) for x in buffer) >= LOTE_PONTOS:
                esvaziar()
            if k in evid:
                w, h, mask, z, _ = sc.profundidade(sc.ids[k])
                D = np.full(w * h, np.nan, np.float32)
                D[mask] = z
                vistas.append(Vista(D.reshape(h, w), sc.K[int(sc.modo[k])], T_ref @ sc.T[k], g))
        esvaziar()
        if not ks_all:
            return dict(
                chaves=np.empty(0, np.int64),
                C=np.empty((0, 3)),
                n=np.empty(0, np.int64),
                vistas=[],
                calib=calib,
                pixels=0,
                quadros=len(com),
            )
        u, inv, _ = gpu.agrupar(xp.concatenate(ks_all))
        soma = gpu.somar_por_grupo(xp.concatenate(soma_all), inv, len(u))
        cnt = gpu.somar_por_grupo(xp.concatenate(n_all), inv, len(u))
        return dict(
            chaves=gpu.para_cpu(u),
            C=gpu.para_cpu(soma / cnt[:, None]),
            n=gpu.para_cpu(cnt).astype(np.int64),
            vistas=vistas,
            calib=calib,
            pixels=n_pix,
            quadros=len(com),
        )
    finally:
        sc.c.close()


def gravar_scan(proj, destino, g, T_ref, voxel, chaves_rem, calib, min_pix, tmp):
    """Reescreve o frames.dataset do scan sem os pixels dos voxels removidos. Devolve estatisticas."""
    from revo_projeto import RevoScan, gravar_ply, ler_ply

    sc = RevoScan(os.path.join(proj, "data", g), tmp)
    st = dict(pixels_antes=0, pixels_depois=0, quadros=0, quadros_zerados=0)
    try:
        for m, c in calib.items():
            sc.K[int(m)] = tuple(c["K"])
        xp = gpu.xp()
        rem = xp.sort(gpu.para_gpu(np.asarray(chaves_rem, np.int64)))

        def removido(P):
            """Mascara numpy dos pontos cujo voxel (no referencial comum) esta na lista de removidos (busca na GPU)."""
            if not len(rem):
                return np.zeros(len(P), bool)
            kk = chaves_voxel(transformar(gpu.para_gpu(P, xp.float64), T_ref), voxel)
            j = xp.clip(xp.searchsorted(rem, kk), 0, len(rem) - 1)
            return gpu.para_cpu(rem[j] == kk)

        db = os.path.join(tmp, g + ".vis")
        sc.c.execute("pragma wal_checkpoint(TRUNCATE)")
        shutil.copy(sc.db, db)
        cn = sqlite3.connect(db)
        for k, i in enumerate(sc.ids):
            if not sc.tem_prof[k]:
                continue
            w, h, mask, zz, quatro = sc.profundidade(i)
            P, pix = sc.pontos(k)
            ok = ~removido(P)
            st["quadros"] += 1
            st["pixels_antes"] += len(pix)
            if ok.all():
                st["pixels_depois"] += len(pix)
                continue
            if ok.sum() < min_pix:
                cn.execute("delete from FrameRGBD where id=?", (i,))
                st["quadros_zerados"] += 1
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
            st["pixels_depois"] += int(ok.sum())
        cn.commit()
        cn.execute("pragma wal_checkpoint(TRUNCATE)")
        integ = cn.execute("pragma integrity_check").fetchone()[0]
        cn.close()
        if integ != "ok":
            raise RuntimeError(f"{g}: integridade SQLite {integ}")
        for ext in ("-wal", "-shm", "-journal"):
            if os.path.exists(db + ext):
                raise RuntimeError(f"{g}: sobrou arquivo de journal {ext}")
        with open(db, "r+b") as f:
            f.write(sc.sig)
        os.makedirs(os.path.join(destino, "data", g, "cache"), exist_ok=True)
        # copyfile cria o arquivo no destino: herda a ACL da entrega (mover do temporario deixava o Revo sem acesso)
        shutil.copyfile(db, os.path.join(destino, "data", g, "cache", "frames.dataset"))
        st["integridade"] = integ
        prev = os.path.join(proj, "data", g, "raw_preview.ply")
        if os.path.isfile(prev):
            hdr, arr = ler_ply(prev)
            okp = ~removido(arr[:, :3].astype(float))
            gravar_ply(os.path.join(destino, "data", g, "raw_preview.ply"), hdr, arr[okp])
            st["preview_antes"], st["preview_depois"] = int(len(arr)), int(okp.sum())
    finally:
        sc.c.close()
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("projeto")
    ap.add_argument("--poses", required=True, help="JSON {guid: 4x4 scan->referencial comum}")
    ap.add_argument("--calibracoes", help="JSON opcional {guid: {modo: {K, erro_preview_mm}}}; preserva K conhecido")
    ap.add_argument("--saida", required=True)
    ap.add_argument("--scans", nargs="+", help="padrao: todos os guids de poses.json presentes no projeto")
    ap.add_argument("--voxel", type=float, default=0.5, help="mm dos candidatos")
    ap.add_argument("--tol", type=float, default=0.5, help="mm de tolerancia de profundidade")
    ap.add_argument("--passo-quadros", type=int, default=6, help="1 a cada N quadros de cada scan vira evidencia")
    ap.add_argument("--min-contra", type=int, help="scans que precisam ver espaco livre (padrao min(2, n-1))")
    ap.add_argument(
        "--min-livre-quadros", type=int, default=2, help="quadros de um scan com espaco livre para ele contradizer"
    )
    ap.add_argument("--min-pix-quadro", type=int, default=200, help="quadro com menos pixels perde a profundidade")
    a = ap.parse_args()
    if a.voxel <= 0 or a.tol <= 0 or a.passo_quadros < 1 or a.min_livre_quadros < 1:
        ap.error("voxel e tol > 0; passo-quadros e min-livre-quadros >= 1")
    baixa_prioridade()
    orcamento_threads(1, teto=4)
    proj, saida = os.path.abspath(a.projeto), os.path.abspath(a.saida)
    if os.path.exists(saida) and (not os.path.isdir(saida) or os.listdir(saida)):
        sys.exit(f"destino ja existe: {saida}")
    try:
        dentro = os.path.commonpath([proj, saida]) == proj
    except ValueError:  # drives diferentes
        dentro = False
    if dentro:
        sys.exit("a saida deve ficar fora do projeto de origem")
    poses = {g: np.array(T, float) for g, T in json.load(open(a.poses, encoding="utf-8")).items()}
    scans = [
        g
        for g in (a.scans or sorted(poses))
        if os.path.isfile(os.path.join(proj, "data", g, "cache", "frames.dataset"))
    ]
    faltam = [g for g in (a.scans or []) if g not in poses]
    if faltam:
        sys.exit(f"scans sem pose em {a.poses}: {faltam}")
    if len(scans) < 2:
        sys.exit("consenso por visibilidade exige >= 2 scans com pose e quadros")
    calibracoes = {}
    if a.calibracoes:
        with open(a.calibracoes, encoding="utf-8") as f:
            calibracoes = json.load(f)
        if not isinstance(calibracoes, dict) or set(calibracoes) - set(scans):
            ap.error("calibracoes devem conter somente os IDs dos scans selecionados")
        calibracoes = {g: validar_calibracao_fixa(c, ()) for g, c in calibracoes.items()}
    min_contra = a.min_contra if a.min_contra is not None else min(2, len(scans) - 1)
    if not 1 <= min_contra <= len(scans) - 1:
        sys.exit(f"--min-contra deve estar entre 1 e {len(scans) - 1}")
    try:
        gpu.xp()  # GPU obrigatoria (ErroGPU sem CUDA, salvo REVO_PERMITIR_CPU=1)
    except gpu.ErroGPU as e:
        sys.exit(str(e))
    _LOGGER.info("%s %s", "motor:", "cuda" if gpu.em_gpu() else "cpu (REVO_PERMITIR_CPU=1)")
    t0 = time.time()
    dados = {}
    with tempfile.TemporaryDirectory() as tmp:
        for g in scans:
            t1 = time.time()
            dados[g] = carregar_scan(proj, g, poses[g], a.voxel, a.passo_quadros, tmp, calibracoes.get(g))
            _LOGGER.info(
                "%s",
                f"  {g}: {dados[g]['quadros']} quadros, {dados[g]['pixels']:,} pixels, {len(dados[g]['C']):,} candidatos de {a.voxel:g} mm, {len(dados[g]['vistas'])} vistas de evidencia ({time.time() - t1:.0f}s)",
            )
        vistas = [v for g in scans for v in dados[g]["vistas"]]
        rel = dict(parametros=dict(vars(a), min_contra=min_contra), scans={})
        os.makedirs(saida, exist_ok=True)
        insp = os.path.join(saida, "_inspecao_visibilidade")
        os.makedirs(insp, exist_ok=True)
        rem_por_scan = {}
        for g in scans:
            t1 = time.time()
            d = dados[g]
            votos = avaliar_candidatos(d["C"], g, vistas, a.tol, a.min_livre_quadros)
            est = decidir(votos, min_contra)
            rem_por_scan[g] = d["chaves"][est == 1]
            n_obs = d["n"]
            sem_cob = votos["cobrem"] == 0
            r = dict(
                candidatos=int(len(est)),
                confirmados=int((est == 2).sum()),
                removidos=int((est == 1).sum()),
                inconclusivos=int((est == 0).sum()),
                sem_cobertura_de_outros=int(sem_cob.sum()),
                pixels_em_removidos_pct=float(100 * n_obs[est == 1].sum() / max(1, n_obs.sum())),
                pixels_em_inconclusivos_pct=float(100 * n_obs[est == 0].sum() / max(1, n_obs.sum())),
                calibracao=d["calib"],
                segundos_votacao=round(time.time() - t1, 1),
            )
            rel["scans"][g] = r
            for nome, m in (("removidos", est == 1), ("inconclusivos", est == 0), ("confirmados", est == 2)):
                gravar_ply_pontos(os.path.join(insp, f"{g[:8]}_{nome}.ply"), d["C"][m])
            _LOGGER.info(
                "%s",
                f"  {g}: confirmados {r['confirmados']:,} | REMOVER {r['removidos']:,} ({r['pixels_em_removidos_pct']:.2f}% dos pixels) | inconclusivos {r['inconclusivos']:,} (sem cobertura {r['sem_cobertura_de_outros']:,}) ({r['segundos_votacao']}s)",
            )
        del vistas
        for g in dados:
            dados[g]["vistas"] = []
        # copia do projeto (menos os datasets reescritos)
        for g in os.listdir(os.path.join(proj, "data")):
            ign = shutil.ignore_patterns("frames.dataset", "raw_preview.ply") if g in scans else None
            shutil.copytree(os.path.join(proj, "data", g), os.path.join(saida, "data", g), ignore=ign)
        for f in os.listdir(proj):
            if os.path.isfile(os.path.join(proj, f)):
                shutil.copy2(os.path.join(proj, f), os.path.join(saida, f))
        for g in scans:
            st = gravar_scan(
                proj, saida, g, poses[g], a.voxel, rem_por_scan[g], dados[g]["calib"], a.min_pix_quadro, tmp
            )
            rel["scans"][g].update(st)
            _LOGGER.info(
                "%s",
                f"  {g}: pixels {st['pixels_antes']:,} -> {st['pixels_depois']:,} ({100 * (1 - st['pixels_depois'] / max(1, st['pixels_antes'])):.2f}% removidos), {st['quadros_zerados']} quadros sem profundidade, integridade {st['integridade']}",
            )
    rel["segundos"] = round(time.time() - t0, 1)
    rel["estado"] = "revisao_pendente"
    rel["limites"] = (
        "Remove so o contradito por espaco livre e confirmado por nenhum outro scan. Inconclusivo fica. "
        "Scans da mesma pose de apoio contam como distintos; pose errada > tol gera contradicao falsa. "
        "Fusoes/malhas antigas copiadas (se houver) ficam desatualizadas: refundir."
    )
    with open(os.path.join(saida, "consenso_visibilidade.json"), "w", encoding="utf-8") as f:
        json.dump(rel, f, indent=1, default=float)
    _LOGGER.info(
        "%s", f"concluido em {rel['segundos']:.0f}s; relatorio {os.path.join(saida, 'consenso_visibilidade.json')}"
    )


if __name__ == "__main__":
    main()
