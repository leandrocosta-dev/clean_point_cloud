"""Prato (base) da cena e amostra dos quadros no referencial do mundo. Sem Open3D."""
import numpy as np


def plano_ransac(P, thr=1.0, it=3000, seed=0):
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(it):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ a
        k = int((np.abs(P @ n + d) < thr).sum())
        if k > best[0]:
            best = (k, (n, d))
    n, d = best[1]
    m = np.abs(P @ n + d) < thr
    c = P[m].mean(0)
    _, _, V = np.linalg.svd(P[m] - c, full_matrices=False)
    n = V[2]
    d = -n @ c
    # orienta a normal para o lado onde estao os objetos
    if np.mean(P @ n + d > 3 * thr) < np.mean(P @ n + d < -3 * thr):
        n, d = -n, -d
    return n, d, float(m.mean())


def achar_base(A, cams, candidatos=4, min_frac=0.03, min_cam=0.8, max_abaixo=0.05, log=print):
    """Procura o plano do prato entre os maiores planos da cena.

    Um plano so e aceito como base se:
      - as cameras estao do lado de cima dele (>= 80%): o scanner sempre olha o
        prato de cima. Isso rejeita faces planas da propria peca, que sao vistas
        do lado de fora (teste real: face de cima do difusor tinha 0% de cameras
        do lado da peca);
      - quase nenhum ponto fica abaixo dele (<= 5%);
      - tem pelo menos 3% dos pontos da amostra.
    Retorna (n, d) com a normal apontando para as cameras, ou None se nao achar.
    """
    resto = A
    for i in range(candidatos):
        if len(resto) < 1000:
            break
        n, d, _ = plano_ransac(resto[:: max(1, len(resto) // 40000)], thr=1.0, it=800)
        if np.mean(cams @ n + d > 0) < 0.5:          # normal para o lado das cameras
            n, d = -n, -d
        hA = A @ n + d
        frac = float(np.mean(np.abs(hA) < 1.0))
        cam_ok = float(np.mean(cams @ n + d > 0))
        abaixo = float(np.mean(hA < -2.5))
        aceito = frac >= min_frac and cam_ok >= min_cam and abaixo <= max_abaixo
        log(f"   plano {i}: {frac*100:.1f}% dos pontos | cameras acima {cam_ok*100:.0f}% | "
            f"pontos abaixo {abaixo*100:.1f}% -> {'BASE' if aceito else 'rejeitado'}")
        if aceito:
            return n, d
        resto = resto[np.abs(resto @ n + d) >= 1.5]
    log("   nenhum plano passou como base: nada sera removido por altura, so objetos soltos")
    return None


def amostrar(scan, passo=6, nquadros=150, raio_centro=0.15):
    """Funde uma amostra dos quadros no referencial do mundo.

    Retorna (pontos, posicoes das cameras, pontos do centro da imagem de cada quadro).
    """
    ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
    amostra, centrais, cams = [], [], []
    for k in ks[:: max(1, len(ks) // nquadros)]:
        r = scan.profundidade(scan.ids[k])
        if r is None:
            continue
        w = r[0]
        P, pix = scan.pontos(k, passo=passo)
        if P is None or len(P) < 100:
            continue
        amostra.append(P)
        cams.append(scan.T[k][:3, 3])
        fx, fy, cx, cy = scan.K[int(scan.modo[k])]
        v, u = np.divmod(pix, w)
        centro = (u - cx) ** 2 + (v - cy) ** 2 < (raio_centro * w) ** 2
        centrais.append(P[centro])
    return np.vstack(amostra), np.array(cams), centrais
