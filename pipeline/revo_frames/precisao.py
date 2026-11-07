"""Evidencia entre quadros por reprojecao, sem mover pontos ou poses.

Um ponto so e contradito quando outra camera observa espaco livre na sua
posicao. Oclusao, falta de cobertura e pares com vies alto sao inconclusivos.
"""
from collections import OrderedDict

import numpy as np


def reprojetar(P, T, K, D, tol):
    """Retorna apoio, contradicao e residuo central (medido - previsto), em mm."""
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tolerancia deve ser positiva")
    camera = (P - T[:3, 3]) @ T[:3, :3]
    z = -camera[:, 2]
    h, w = D.shape
    fx, fy, cx, cy = K
    frente = np.isfinite(camera).all(axis=1) & (z > 0)
    u = np.full(len(P), -10, dtype=np.int64)
    v = u.copy()
    u[frente] = np.rint(camera[frente, 0] * fx / z[frente] + cx).astype(np.int64)
    v[frente] = np.rint(-camera[frente, 1] * fy / z[frente] + cy).astype(np.int64)
    dentro = frente & (u >= 1) & (v >= 1) & (u < w-1) & (v < h-1)
    ids = np.flatnonzero(dentro)
    apoio, livre = np.zeros(len(P), bool), np.zeros(len(P), bool)
    residuo = np.full(len(P), np.nan)
    if len(ids):
        janela = np.stack([D[v[ids]+dy, u[ids]+dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=1)
        valido = np.isfinite(janela) & (janela > 0)
        delta = janela - z[ids, None]
        apoio[ids] = np.any(valido & (np.abs(delta) <= tol), axis=1)
        # Exige janela completa: um fundo visto ao lado de uma borda nao prova
        # que a borda nao existe. Superficie mais proxima e oclusao, nao ruido.
        livre[ids] = np.all(valido & (delta > tol), axis=1)
        central = valido[:, 4]
        residuo[ids[central]] = delta[central, 4]
    return apoio, livre, residuo


def decidir(candidato, apoios, contradicoes, minimo=2):
    """Apoio prevalece. Sem duas contradicoes de quadros distintos, conservar."""
    return candidato & (apoios == 0) & (contradicoes >= minimo)


def render_cortes(base, local, final, caminho):
    """Tres cortes de 1 mm na mesma amostra; nao interpola superficie."""
    if len(base) < 3:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    centro = np.median(base, axis=0)
    _, _, eixos = np.linalg.svd(base-centro, full_matrices=False)
    b, l, f = [(p-centro) @ eixos.T for p in (base, local, final)]
    fig, axs = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    for linha, pos in enumerate(np.quantile(b[:, 0], [0.25, 0.5, 0.75])):
        for col, (p, nome) in enumerate(((l, 'Filtro local'), (f, 'Validacao temporal'))):
            ax = axs[linha, col]
            bg = b[np.abs(b[:, 0]-pos) <= 0.5]
            q = p[np.abs(p[:, 0]-pos) <= 0.5]
            ax.scatter(bg[:, 1], bg[:, 2], s=3, c='#c8c8c8', label='peca selecionada')
            ax.scatter(q[:, 1], q[:, 2], s=2, c='#126f98', label='mantidos')
            ax.set_title(f'{nome}: corte em {pos:.1f} mm')
            ax.set_xlabel('mm'); ax.set_ylabel('mm'); ax.set_aspect('equal'); ax.grid(alpha=0.2)
    axs[0, 0].legend(fontsize=8)
    fig.suptitle('Mesma amostra (1/8 dos pixels); cortes com espessura de 1 mm')
    fig.tight_layout()
    fig.savefig(caminho, dpi=140)
    plt.close(fig)


class ValidadorTemporal:
    def __init__(self, scan, tol=1.0, passo=8, minimo=2):
        if tol <= 0 or not np.isfinite(tol) or passo < 1 or minimo not in (2, 3, 4):
            raise ValueError("tol > 0, passo >= 1 e minimo entre 2 e 4")
        self.scan, self.tol, self.passo, self.minimo = scan, tol, passo, minimo
        self.ks = [k for k in range(len(scan.ids)) if scan.tem_prof[k]]
        self.pos = {k: j for j, k in enumerate(self.ks)}
        self.cache = OrderedDict()
        self.diagnosticos = []

    def mapa(self, k):
        if k not in self.cache:
            w, h, mask, z, _ = self.scan.profundidade(self.scan.ids[k])
            d = np.full(w*h, np.nan, np.float32)
            d[mask] = z
            self.cache[k] = d.reshape(h, w)
            if len(self.cache) > 12:
                self.cache.popitem(last=False)
        self.cache.move_to_end(k)
        return self.cache[k]

    def validar(self, k, P, candidato, referencia=None):
        suporte = np.zeros(len(P), np.uint8)
        livre = suporte.copy()
        j = self.pos[k]
        diagnostico = []
        for deslocamento in (-2*self.passo, -self.passo, self.passo, 2*self.passo):
            jj = j + deslocamento
            if not 0 <= jj < len(self.ks):
                continue
            viz = self.ks[jj]
            d = self.mapa(viz)
            K = self.scan.K[int(self.scan.modo[viz])]
            # Diagnostico limitado a 1500 pontos, sem ICP e sem alterar poses.
            ref = P if referencia is None else P[referencia]
            amostra = ref[::max(1, len(ref)//1500)]
            _, _, r = reprojetar(amostra, self.scan.T[viz], K, d, self.tol)
            # Nao truncar residuos grandes: isso esconderia justamente deriva
            # de rastreio. Oclusoes podem gerar alertas conservadores de revisao.
            r = r[np.isfinite(r)]
            med = float(np.median(r)) if len(r) >= 100 else None
            mad = float(1.4826*np.median(np.abs(r-med))) if med is not None else None
            suspeito = med is not None and abs(med) > 1.5*self.tol
            # Dispersao entre quadros so pode elevar a tolerancia ate 2x.
            tol = max(self.tol, min(2*self.tol, 2.5*mad)) if mad is not None else self.tol
            diagnostico.append(dict(vizinho=int(self.scan.ids[viz]), amostras=len(r),
                                    vies_profundidade_mm=med, dispersao_mm=mad,
                                    suspeita_desalinhamento=suspeito, tolerancia_mm=tol))
            if suspeito or not candidato.any():
                continue
            ids = np.flatnonzero(candidato)
            a, b, _ = reprojetar(P[ids], self.scan.T[viz], K, d, tol)
            suporte[ids] += a
            # Pouca sobreposicao nao permite diagnosticar o par com seguranca.
            if med is not None:
                livre[ids] += b
        self.diagnosticos.append(dict(quadro=int(self.scan.ids[k]), pares=diagnostico))
        rejeitar = decidir(candidato, suporte, livre, self.minimo)
        self.ultima_evidencia = dict(apoio=candidato & (suporte > 0),
            incertos=candidato & (suporte == 0) & ~rejeitar, removidos=rejeitar)
        return rejeitar, dict(candidatos=int(candidato.sum()), confirmados=int(rejeitar.sum()),
                             com_apoio=int((candidato & (suporte > 0)).sum()),
                             inconclusivos=int((candidato & (suporte == 0) & ~rejeitar).sum()))
