"""Merge Alignment (Feature) automatico no Revo Scan 6: marca a fusao mais recente de cada scan do grupo e gera o modelo.

    python pipeline/revo_merge.py --grupo Projeto9_v3            # [mouse] Merge Alignment > Feature > Preview > Generate Model
    python pipeline/revo_merge.py --detectar captura.png        # so testa a deteccao das caixas numa captura (sem mouse)

Tela de Merge (Revo 6.4.2): lista a esquerda com cada scan (icone de arquivo) e, embaixo, suas fusoes com caixa de
marcar; botoes Manual align / Feature / Generate Model no topo e "Preview the Alignment" embaixo a direita. As caixas
NAO aparecem na arvore UIA: sao achadas na imagem (quadrado de borda clara a ~x=48..64 px da janela). Para cada scan
marca a ULTIMA caixa do bloco (fusao mais nova). O manual: Feature aceita ate 9 modelos, sobreposicao >= 10%.
Resultado: nova pasta na biblioteca com Merge_XX.revox (ver `revo_cli.py biblioteca-scans --nome Merge`), que
`avaliar_merge_revo.py` mede.
"""

import argparse
import os
import sys
import time

import numpy as np
import revo_ui as R


def detectar(img):
    """img: array RGB da janela. Devolve blocos (um por scan, na ordem da lista), cada um = [(x, y) do centro de cada caixa].
    Caixa (Revo 6.4.2): quadrado de 17 px com borda branca de 1 px em x=48..64 da janela, interior escuro.
    Agrupamento: caixas a <= 40 px = mesmo scan; o titulo do scan abre um intervalo de 64 px."""
    g = img.astype(int).sum(axis=2) // 3
    h = g.shape[0]
    caixas = []
    y = 150
    while y < h - 20:
        if (
            (g[y, 48:65] > 200).all()
            and (g[y + 16, 48:65] > 200).all()
            and g[y + 8, 48] > 200
            and g[y + 8, 64] > 200
            and g[y + 8, 56] < 120
        ):
            caixas.append(y + 8)
            y += 17
        else:
            y += 1
    # caixas do mesmo scan ficam a 1 linha (32 px) uma da outra; entre scans ha a linha do titulo (>= 2 linhas)
    blocos = []
    for c in caixas:
        if blocos and c - blocos[-1][-1][1] <= 40:
            blocos[-1].append((56, c))
        else:
            blocos.append([(56, c)])
    return blocos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grupo")
    ap.add_argument("--detectar", help="captura PNG da tela de merge: so imprime as caixas achadas")
    ap.add_argument("--espera-preview", type=float, default=600)
    a = ap.parse_args()
    from PIL import Image, ImageGrab

    if a.detectar:
        b = detectar(np.asarray(Image.open(a.detectar).convert("RGB")))
        print(len(b), "scans;", [len(x) for x in b], "caixas; marcaria:", [x[-1] for x in b])
        return
    if not a.grupo:
        sys.exit("--grupo e obrigatorio")
    from pywinauto import mouse

    ui = R.RevoUI.abrir()
    try:  # se ficou na tela de merge de uma rodada anterior, sai dela
        ui._botao("Exit Merge Alignment").click_input()
        time.sleep(2)
    except LookupError:
        pass
    ui.entrar_grupo(a.grupo)
    n_esperado = ui._contagem()
    ui._botao("Merge Alignment").click_input()
    ui._esperar(lambda: ui._botao("Preview the Alignment"), 60, 0.5, "tela de merge nao abriu")
    time.sleep(2)
    ui._botao("Feature").click_input()
    time.sleep(1)
    r = ui.principal.rectangle()
    img = np.asarray(ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom), all_screens=True).convert("RGB"))
    blocos = detectar(img)
    print(
        "scans na lista:",
        len(blocos),
        "| no grupo:",
        n_esperado,
        "| caixas por scan:",
        [len(x) for x in blocos],
        flush=True,
    )
    if len(blocos) != n_esperado:
        sys.exit("numero de scans na lista difere do grupo: nao marquei nada")
    if len(blocos) > 9:
        sys.exit("Feature aceita ate 9 modelos")

    def marcada(x, y):
        """Caixa marcada = interior colorido/numerado (o Revo pinta a caixa com a cor do modelo e o numero)."""
        im = np.asarray(
            ImageGrab.grab(
                bbox=(r.left + x - 6, r.top + y - 6, r.left + x + 7, r.top + y + 7), all_screens=True
            ).convert("RGB")
        ).astype(int)
        return im.mean() > 90

    for k, b in enumerate(blocos):
        x, y = b[-1]
        for tentativa in range(3):  # o Revo ignora cliques enquanto carrega o modelo anterior
            mouse.click(coords=(r.left + x, r.top + y))
            t0 = time.time()
            while time.time() - t0 < 90 and not marcada(x, y):
                time.sleep(1)
            if marcada(x, y):
                break
        else:
            sys.exit(f"nao consegui marcar a caixa do scan {k + 1} (y={y})")
        print(f"scan {k + 1}/{len(blocos)} marcado ({time.time() - t0:.0f} s)", flush=True)
        time.sleep(1)
    R.LogRevo()
    ui._botao("Preview the Alignment").click_input()
    t0 = time.time()
    # fim do preview: Generate Model habilita
    ui._esperar(lambda: ui._botao("Generate Model").is_enabled(), a.espera_preview, 2.0, "preview nao terminou")
    print(f"preview em {time.time() - t0:.0f} s", flush=True)
    ui.captura(os.path.join(os.environ.get("TEMP", "."), "revo_merge_preview.png"))
    antes = set(os.listdir(R.BIBLIOTECA))
    ui._botao("Generate Model").click_input()
    ui._esperar(
        lambda: any(
            f.endswith(".revox") and f.startswith("Merge")
            for d in set(os.listdir(R.BIBLIOTECA)) - antes
            for f in os.listdir(os.path.join(R.BIBLIOTECA, d))
        ),
        600,
        2.0,
        "merge nao foi gravado",
    )
    time.sleep(3)
    novo = [os.path.join(R.BIBLIOTECA, d) for d in set(os.listdir(R.BIBLIOTECA)) - antes]
    print("merge:", novo)
    R.restaurar_janelas()


if __name__ == "__main__":
    main()
