"""Automacao do Revo Scan 6 pela interface (UI Automation do Windows via pywinauto).

O Revo nao tem linha de comando nem API: por argumento abre so .revox, e o fluxo de projeto passa pela
biblioteca interna (%APPDATA%/Revopoint/RevoScan6/Projects). Este modulo dirige a propria janela:

    ui = RevoUI.abrir()                              # inicia ou anexa ao RevoScan6.exe
    ui.carregar_projeto(r"D:\\x\\proj.revo")          # File > Load Project (dialogo nativo): importa os scans
    ui.abrir_scan("20260613_162307")                 # duplo clique no cartao da biblioteca
    ui.fundir(); ui.malha(); ui.exportar(pasta)      # Fusion > Apply, Meshing > Apply, Export Model (.ply)
    ui.biblioteca()                                  # volta a lista

Medido (scan de 383 quadros, 1 M de pontos, GPU): fusao 6 s, malha 10 s, exportacao 2 s.
Fim de cada etapa: fusao = linha "Fused ..." na lista do modelo; malha = linha "Mesh ..."; exportacao =
janela RvMessageBox "Model Export Complete". Os paineis de Fusion/Meshing sao QMenu cujo conteudo nao
aparece na arvore UIA, entao o Apply e clicado por posicao relativa ao menu (rodape - 32 px).
Requer: pip install pywinauto pillow. Usa o mouse real (click_input): nao mexa no PC durante a execucao.
"""
import os
import subprocess
import time

from pywinauto import Desktop, mouse

EXE = r"C:\Program Files\Revo Scan 6\RevoScan6.exe"


class RevoUI:
    def __init__(self, pid):
        self.pid = pid
        self.ds = Desktop(backend="uia")
        self.w32 = Desktop(backend="win32")

    # ---------------------------------------------------------------- janelas
    @classmethod
    def abrir(cls, exe=EXE, espera=40):
        import psutil
        for p in psutil.process_iter(["name"]):
            if p.info["name"] == "RevoScan6.exe":
                return cls(p.pid)
        proc = subprocess.Popen([exe], cwd=os.path.dirname(exe))
        ui = cls(proc.pid)
        t0 = time.time()
        while time.time() - t0 < espera:
            if ui._janelas("MainWindow"):
                time.sleep(3)
                return ui
            time.sleep(1)
        raise RuntimeError("Revo Scan nao abriu a janela principal")

    def _janelas(self, classe):
        return [w for w in self.ds.windows(process=self.pid) if w.class_name() == classe]

    @property
    def principal(self):
        return self._janelas("MainWindow")[0]

    def _botao(self, texto, raiz=None):
        for d in (raiz or self.principal).descendants(control_type="Button"):
            if d.window_text() == texto and d.is_visible():
                return d
        raise LookupError(f"botao {texto!r} nao encontrado")

    def _fechar_avisos(self):
        """Fecha RvMessageBox/CommonDialog pendentes (ex.: 'Model Export Complete')."""
        for cls in ("RvMessageBox", "CommonDialog"):
            for w in self._janelas(cls):
                self._botao("Confirm", w).click_input()
                time.sleep(1)

    def _esperar(self, cond, timeout, passo=1.0, msg="tempo esgotado"):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if cond():
                    return time.time() - t0
            except Exception:
                pass
            time.sleep(passo)
        raise TimeoutError(msg)

    def _linhas_modelo(self):
        return [d.window_text() for d in self.principal.descendants(control_type="ListItem")]

    # ---------------------------------------------------------------- biblioteca
    def biblioteca(self):
        self._fechar_avisos()
        for d in self.principal.descendants(control_type="TabItem"):
            if d.window_text() == "File":
                d.click_input()
        self._esperar(lambda: any(d.window_text() == "Load Project" for d in self.principal.descendants(control_type="Text")),
                      15, msg="biblioteca nao apareceu")

    def _contagem(self):
        for d in self.principal.descendants(control_type="Text"):
            if d.window_text().startswith("Scan(") and d.is_visible():
                return int(d.window_text()[5:-1])
        return -1

    def _esperar_dialogo(self, titulo, timeout=20):
        self._esperar(lambda: self.w32.window(title=titulo, class_name="#32770").exists(), timeout, msg=f"dialogo {titulo!r} nao abriu")
        return self.w32.window(title=titulo, class_name="#32770")

    def carregar_projeto(self, revo, timeout=120):
        """File > Load Project: importa os scans do .revo para a biblioteca do Revo (copia os dados)."""
        self.biblioteca()
        antes = self._contagem()
        next(d for d in self.principal.descendants() if d.window_text() == "Load Project").click_input()
        dlg = self._esperar_dialogo("Load Project")
        dlg.child_window(class_name="ComboBoxEx32").child_window(class_name="Edit").set_edit_text(os.path.abspath(revo))
        dlg.child_window(class_name="Button", title_re=r"&?Abrir|&?Open").click()
        self._esperar(lambda: self._contagem() > antes, timeout, msg="scans nao entraram na biblioteca")
        return self._contagem() - antes

    def abrir_scan(self, nome, timeout=60):
        """Filtra a biblioteca pelo nome (caixa de busca) e abre o primeiro cartao (duplo clique).
        Os cartoes sao desenhados sem texto na arvore UIA, entao a busca e o caminho confiavel."""
        self.biblioteca()
        caixa = next(d for d in self.principal.descendants(control_type="Edit") if d.is_visible())
        caixa.set_edit_text("")
        caixa.set_edit_text(nome)
        time.sleep(2)
        r = self.principal.rectangle()
        mouse.double_click(coords=(r.left + 580, r.top + 420))                 # miniatura do 1o cartao
        self._esperar(lambda: any(d.window_text() == "Fusion" for d in self.principal.descendants(control_type="Button")),
                      timeout, msg="scan nao abriu")

    # ---------------------------------------------------------------- etapas
    # Apply de cada painel, em px relativos a janela principal maximizada (1918 x 999); o conteudo dos
    # QMenu nao aparece na arvore UIA e o proprio QMenu nem sempre e listado.
    APPLY = {"Fusion": (497, 324), "Meshing": (950, 433)}

    def _apply_do_painel(self, botao, timeout=15):
        from pywinauto.keyboard import send_keys
        send_keys("{ESC}")                                     # fecha painel que tenha ficado aberto
        time.sleep(0.5)
        self._botao(botao).click_input()
        time.sleep(2)
        r = self.principal.rectangle()
        dx, dy = self.APPLY[botao]
        mouse.click(coords=(r.left + dx, r.top + dy))

    def fundir(self, timeout=1800):
        """Edit > Fusion > Apply (parametros como estao no painel: distancia de pontos, remover isolados)."""
        n = len(self._linhas_modelo())
        self._apply_do_painel("Fusion")
        return self._esperar(lambda: len(self._linhas_modelo()) > n, timeout, 1.0, "fusao nao terminou")

    def malha(self, timeout=3600):
        """Edit > Meshing > Apply (Fast Meshing / Optimize Low, ou o que estiver no painel)."""
        n = len(self._linhas_modelo())
        self._apply_do_painel("Meshing")
        return self._esperar(lambda: len(self._linhas_modelo()) > n, timeout, 1.0, "malha nao terminou")

    def exportar(self, pasta, prefixos=("Fused", "Mesh"), timeout=600):
        """Export Model (.ply): marca na lista do dialogo so os itens cujo nome comeca por `prefixos`
        (os nomes que o Revo da as etapas novas: 'Fused 1021_1 (0.41mm)', 'Mesh 1021_1'). Itens antigos
        importados do projeto ('fuse', 'fuse_mesh') ficam de fora. Devolve os arquivos criados."""
        os.makedirs(pasta, exist_ok=True)
        antes = set(os.listdir(pasta))
        self._botao("Export Model").click_input()
        self._esperar(lambda: self._botao("Confirm"), 10, 0.3, "dialogo de exportacao nao abriu")
        time.sleep(0.5)
        marcados = []
        for d in self.principal.descendants(control_type="ListItem"):
            if d.is_visible() and "TextureExportDialog" in d.element_info.automation_id and d.window_text().startswith(prefixos):
                r = d.rectangle()
                mouse.click(coords=(r.left + 12, (r.top + r.bottom) // 2))       # caixa de marcar, borda esquerda da linha
                marcados.append(d.window_text())
                time.sleep(0.4)
        if len(marcados) != len(prefixos):
            raise RuntimeError(f"esperava {len(prefixos)} itens a exportar e achei {marcados}")
        self._botao("Confirm").click_input()
        dlg = self._esperar_dialogo("Export Model").wrapper_object()
        ed = next(d for d in dlg.descendants() if d.class_name() == "Edit" and d.is_visible())
        ed.set_edit_text(os.path.abspath(pasta))
        next(d for d in dlg.descendants() if d.class_name() == "Button" and d.window_text() == "Selecionar pasta").click_input()
        self._esperar(lambda: self._janelas("RvMessageBox"), timeout, 1.0, "exportacao nao terminou")
        self._fechar_avisos()
        return sorted(set(os.listdir(pasta)) - antes)
