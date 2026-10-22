"""Abre uma malha no MeshInspector (GUI) e salva um print da janela, para conferencia visual.

    python pipeline/meshinspector_ui.py <malha.ply> --saida print.png [--espera 25]

No MeshInspector a face de FRENTE aparece azul e o VERSO amarelo-esverdeado: manchas amarelas = faces invertidas ou furo
mostrando a parede de tras. Os NUMEROS (furos, componentes) vem de `malha_meshlib.py`, que usa o mesmo motor e bate com o painel.
Fecha uma instancia anterior do MeshInspector (so a dele). A tela precisa estar ligada; o aviso inicial e dispensado com ESC.
"""
import argparse
import ctypes
import os
import subprocess
import time
from ctypes import wintypes

from PIL import ImageGrab

EXE = r"C:\Program Files\MeshInspector\MeshInspector\MeshInspector.exe"
user32 = ctypes.windll.user32


def _janela():
    achada = []
    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(h, _):
        n = user32.GetWindowTextLengthW(h)
        if n and user32.IsWindowVisible(h):
            b = ctypes.create_unicode_buffer(n + 1); user32.GetWindowTextW(h, b, n + 1)
            if b.value.startswith("MeshInspector"):
                achada.append(h)
        return True
    user32.EnumWindows(cb, 0)
    return achada[0] if achada else None


def _tecla(vk):
    user32.keybd_event(vk, 0, 0, 0); user32.keybd_event(vk, 0, 2, 0)


def capturar(malha, saida, espera=25):
    subprocess.run(["taskkill", "/IM", "MeshInspector.exe", "/F"], capture_output=True)
    time.sleep(2)
    subprocess.Popen([EXE, os.path.abspath(malha)])
    t0 = time.time(); h = None
    while time.time() - t0 < 120 and not h:
        time.sleep(2); h = _janela()
    if not h:
        raise RuntimeError("janela do MeshInspector nao apareceu")
    time.sleep(espera)
    user32.ShowWindow(h, 3); user32.SetForegroundWindow(h); time.sleep(1)
    _tecla(0x1B)                      # ESC: dispensa o aviso inicial
    time.sleep(2)
    r = wintypes.RECT(); user32.GetWindowRect(h, ctypes.byref(r))
    ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom), all_screens=True).save(saida)
    return saida


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("malha"); ap.add_argument("--saida", required=True); ap.add_argument("--espera", type=int, default=25)
    a = ap.parse_args()
    print("print salvo em", capturar(a.malha, a.saida, a.espera))
