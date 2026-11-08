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
import json
import os
import re
import subprocess
import time

from pywinauto import Desktop, mouse

EXE = r"C:\Program Files\Revo Scan 6\RevoScan6.exe"
LOG = os.path.expandvars(r"%LOCALAPPDATA%\Revopoint\RevoScan6\logs\RevoScan6.log")
BIBLIOTECA = os.path.expandvars(r"%APPDATA%\Revopoint\RevoScan6\Projects")


_MINIMIZOU = False


def manter_tela_ativa():
    """Deixa a maquina pronta para clicar: acorda a tela (se o Windows a desligou por tempo), a mantem
    ligada enquanto este processo viver e garante que o cursor responde.
    Janela ELEVADA em primeiro plano (ex.: Gerenciador de Tarefas) bloqueia o mouse de processos sem
    elevacao (UIPI): SetCursorPos falha ("no active desktop"). Nesse caso minimiza as janelas pelo shell
    (como Win+D); `restaurar_janelas()` desfaz. Com tela de bloqueio com senha nada disso basta."""
    global _MINIMIZOU
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM, ES_DISPLAY = 0x80000000, 0x1, 0x2
    u = ctypes.windll.user32
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM | ES_DISPLAY)
    for dx in (1, -1):                                         # movimento de 1 px e volta: liga o monitor
        u.mouse_event(0x0001, dx, 0, 0, 0)                     # MOUSEEVENTF_MOVE relativo
        time.sleep(0.3)
    time.sleep(2)
    if not u.SetCursorPos(*_posicao_cursor()):
        subprocess.run(["powershell", "-NoProfile", "-c", "(New-Object -ComObject Shell.Application).MinimizeAll()"],
                       capture_output=True)
        _MINIMIZOU = True
        time.sleep(2)
        if not u.SetCursorPos(*_posicao_cursor()):
            raise RuntimeError("o mouse nao responde (tela de bloqueio ou janela elevada em foco); "
                               "desbloqueie/feche a janela e rode de novo")


def _posicao_cursor():
    import ctypes
    from ctypes import wintypes
    p = wintypes.POINT()
    ctypes.windll.user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


def restaurar_janelas():
    """Desfaz o MinimizeAll feito por `manter_tela_ativa`."""
    global _MINIMIZOU
    if _MINIMIZOU:
        subprocess.run(["powershell", "-NoProfile", "-c", "(New-Object -ComObject Shell.Application).UndoMinimizeAll()"],
                       capture_output=True)
        _MINIMIZOU = False


class LogRevo:
    """Le so o que o Revo escreveu no log depois de criada (posicao em bytes; trata a rotacao do arquivo)."""

    def __init__(self, caminho=LOG):
        self.caminho = caminho
        self.pos = os.path.getsize(caminho) if os.path.exists(caminho) else 0

    def esperar(self, regex, timeout, msg="tempo esgotado", passo=0.5):
        rx = re.compile(regex)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if os.path.exists(self.caminho):
                if os.path.getsize(self.caminho) < self.pos:                  # rotacionou
                    self.pos = 0
                with open(self.caminho, "rb") as f:
                    f.seek(self.pos)
                    novo = f.read()
                texto = novo.decode("utf-8", "replace")
                m = rx.search(texto)
                if m:
                    self.pos += len(texto[:m.end()].encode("utf-8"))      # proxima espera comeca depois deste achado
                    return m
            time.sleep(passo)
        raise TimeoutError(msg)


def revox_do_scan(guid):
    """Caminho do .revox do scan `guid` dentro da biblioteca do Revo (criado na importacao)."""
    for d in os.listdir(BIBLIOTECA):
        p = os.path.join(BIBLIOTECA, d, guid + ".revox")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"scan {guid} nao esta na biblioteca do Revo")


GRUPOS_CFG = os.path.expandvars(r"%LOCALAPPDATA%\Revopoint\RevoScan6\config\group.cfg")


def revox_no_grupo(grupo, nome):
    """Caminho do .revox `nome` dentro do `grupo` da biblioteca (group.cfg: {"group": {nome: [revox...]}})."""
    cfg = json.load(open(GRUPOS_CFG, encoding="utf-8"))
    for c in cfg.get("group", {}).get(grupo, []):
        if os.path.basename(c)[:-6] == nome and os.path.exists(c):
            return c
    raise FileNotFoundError(f"{nome} nao esta no grupo {grupo!r}")


def pasta_mais_recente(guid):
    """Pasta da biblioteca com o .revox mais novo para `guid` (a importacao mais recente)."""
    achados = [os.path.join(BIBLIOTECA, d) for d in os.listdir(BIBLIOTECA)
               if os.path.exists(os.path.join(BIBLIOTECA, d, guid + ".revox"))]
    if not achados:
        raise FileNotFoundError(f"scan {guid} nao esta na biblioteca do Revo")
    return max(achados, key=lambda p: os.path.getmtime(os.path.join(p, guid + ".revox")))


MALHA_MODO = {"rapida": 0, "alta": 1}
OTIMIZAR = {"nenhum": 0, "baixo": 1, "alto": 2}


def parametros(guid, ponto_mm=None, malha=None, otimizar=None, grade_mm=None, furos=None,
               marcadores=None, isolados=None, revox=None):
    """Grava, no .revox da biblioteca, os valores que o painel de Fusion/Meshing mostra ao abrir o scan.
    O Revo le o arquivo na 1a vez que carrega o scan na sessao e reaproveita o estado se o scan for
    reaberto na mesma sessao; grave antes de abrir e nao regrave um scan ja aberto.
      ponto_mm    Point Distance da fusao (isPointPitchSetted=true impede o Revo de recalcular o automatico)
      malha       'rapida' (Fast Meshing) ou 'alta' (High-Quality Meshing, Poisson)
      otimizar    so na rapida: 'nenhum' | 'baixo' | 'alto'  (optimizeLevel 0/1/2)
      grade_mm    so na alta: Grid Size (meshEdgeLength)
      furos       Hole Filling
      marcadores  Remove marker points
      isolados    Remove Isolation
    `revox`: caminho explicito (use quando a biblioteca tem varios scans com o mesmo guid).
    Devolve o caminho do .revox."""
    p = revox or revox_do_scan(guid)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    t = d["param_template"]["postProcessParam"]
    if ponto_mm is not None:
        t["pointPitch"] = ponto_mm
        t["isPointPitchSetted"] = True
    if malha is not None:
        t["meshMode"] = MALHA_MODO[malha]
    if otimizar is not None:
        t["optimizeLevel"] = OTIMIZAR[otimizar]
    if grade_mm is not None:
        t["meshEdgeLength"] = grade_mm
        t["isMeshLenSetted"] = True
    if furos is not None:
        t["enableAutoFillHoles"] = bool(furos)
    if marcadores is not None:
        t["enableClearMarker"] = bool(marcadores)
    if isolados is not None:
        t["enableExtractIsolation"] = bool(isolados)
        t["enableEraseIsolatedPoint"] = bool(isolados)     # os dois: com so um deles o interruptor do painel continua ligado
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=4, ensure_ascii=False)
    return p


class RevoUI:
    def __init__(self, pid):
        self.pid = pid
        self.ds = Desktop(backend="uia")
        self.w32 = Desktop(backend="win32")

    # ---------------------------------------------------------------- janelas
    @classmethod
    def abrir(cls, exe=EXE, espera=40):
        import psutil
        manter_tela_ativa()
        for p in psutil.process_iter(["name"]):
            if p.info["name"] == "RevoScan6.exe":
                if cls._travado(p.pid):
                    raise RuntimeError(f"Revo Scan (pid {p.pid}) nao responde; feche-o (taskkill /F /PID {p.pid}) e rode de novo")
                ui = cls(p.pid)
                abas = [d.window_text() for d in ui.principal.descendants(control_type="TabItem")]
                if abas and "File" not in abas:     # sessao aberta pelo usuario em outra lingua: nao mexer nela
                    raise RuntimeError("ja ha um Revo Scan aberto fora do ingles (sessao do usuario?); "
                                       "feche-o para a automacao iniciar o seu proprio")
                return ui
        # a lingua do Revo vem do ambiente de quem o inicia (QLocale): os rotulos procurados aqui sao os em ingles
        env = dict(os.environ, LANG="en_US.UTF-8", LC_ALL="en_US.UTF-8")
        proc = subprocess.Popen([exe], cwd=os.path.dirname(exe), env=env)
        ui = cls(proc.pid)
        t0 = time.time()
        while time.time() - t0 < espera:
            if ui._janelas("MainWindow"):
                time.sleep(3)
                return ui
            time.sleep(1)
        raise RuntimeError("Revo Scan nao abriu a janela principal")

    def vivo(self):
        import psutil
        return psutil.pid_exists(self.pid)

    @staticmethod
    def _travado(pid):
        """True se alguma janela do processo esta 'Nao respondendo' (IsHungAppWindow)."""
        import ctypes
        from ctypes import wintypes
        achou = []
        enum = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def cb(h, _):
            p = wintypes.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(h, ctypes.byref(p))
            if p.value == pid and ctypes.windll.user32.IsHungAppWindow(h):
                achou.append(h)
            return True
        ctypes.windll.user32.EnumWindows(enum(cb), 0)
        return bool(achou)

    def fechar(self, espera=40):
        """Fecha o Revo como o botao X (WM_CLOSE na janela principal), para ele gravar o que tem pendente.
        `taskkill` sem /F nao fecha com scan aberto; o close() da janela fecha."""
        import psutil
        try:
            self.principal.close()
        except Exception:
            subprocess.run(["taskkill", "/PID", str(self.pid)], capture_output=True)
        try:
            psutil.Process(self.pid).wait(espera)
        except (psutil.NoSuchProcess, psutil.TimeoutExpired):
            pass
        if psutil.pid_exists(self.pid):
            raise RuntimeError("Revo nao fechou (talvez haja um dialogo pedindo confirmacao)")

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
        from pywinauto.keyboard import send_keys
        send_keys("{ESC}")                                     # painel de Fusion/Meshing aberto
        time.sleep(0.5)
        self._fechar_avisos()
        for d in self.principal.descendants(control_type="TabItem"):
            if d.window_text() == "File":
                d.click_input()
        try:
            self._esperar(lambda: any(d.window_text() == "Load Project" for d in self.principal.descendants(control_type="Text")),
                          15, msg="biblioteca nao apareceu")
        except TimeoutError:
            if any(d.window_text() == "Carregar projeto" for d in self.principal.descendants(control_type="Text")):
                raise RuntimeError("Revo esta em portugues; feche-o e deixe a automacao inicia-lo (LANG=en_US)")
            raise

    def novo_grupo(self, nome, timeout=20):
        """Regra do usuario: cada projeto importado vai para um GRUPO NOVO da biblioteca (+ New Group ->
        nome -> Confirm -> clicar no grupo). Assim nomes de scan repetidos de outras importacoes nao se misturam."""
        self.biblioteca()
        self.limpar_busca()
        self._botao("New Group").click_input()
        edit = None
        t0 = time.time()
        while edit is None and time.time() - t0 < timeout:
            for d in self.principal.descendants(control_type="Edit"):
                if d.is_visible() and d.window_text() == "Untitled":
                    edit = d
            time.sleep(0.5)
        if edit is None:
            raise RuntimeError("dialogo New Group nao apareceu")
        edit.set_edit_text(nome)
        time.sleep(0.5)
        self._botao("Confirm").click_input()
        # GroupListView virtualiza os itens fora da tela. O grupo novo pode
        # existir no modelo sem aparecer na arvore de acessibilidade.
        for lista in self.principal.descendants(control_type="List"):
            if lista.element_info.automation_id.endswith(".GroupListView"):
                from pywinauto.keyboard import send_keys
                lista.click_input()
                send_keys("{END}")
                break
        self._esperar(lambda: any(d.window_text() == nome for d in self.principal.descendants(control_type="ListItem")),
                      timeout, 0.5, f"grupo {nome!r} nao apareceu")
        # A lista se reorganiza depois de criar o grupo; um clique unico pode
        # atingir outro item. Reusar a selecao com verificacao pelo log.
        if self.grupo_atual()!=nome:
            self.entrar_grupo(nome, timeout=timeout)
        self._esperar(lambda: self._contagem() == 0, timeout, 0.5, f"grupo {nome!r} nao abriu vazio")

    def grupos(self):
        """Nomes dos grupos da biblioteca (coluna da esquerda). Nao usa o mouse."""
        return [d.window_text() for d in self.principal.descendants(control_type="ListItem")
                if d.window_text() not in ("false", "true") and d.is_visible()]

    def limpar_busca(self):
        """Esvazia a caixa 'Search Scan Files'. O Revo GUARDA o texto da busca entre sessoes: depois de reiniciar,
        o grupo mostraria so o scan pesquisado antes (observado pelo usuario)."""
        for d in self.principal.descendants(control_type="Edit"):
            if d.is_visible() and d.rectangle().top < self.principal.rectangle().top + 400:
                if d.window_text():
                    d.set_edit_text("")
                    time.sleep(1)
                return

    @staticmethod
    def grupo_atual():
        """Grupo selecionado na biblioteca, pela ultima linha 'slotCurrentGroupChanged ... curName=' do log."""
        try:
            with open(LOG, "rb") as f:
                f.seek(max(0, os.path.getsize(LOG) - 4_000_000))
                txt = f.read().decode("utf-8", "replace")
        except OSError:
            return None
        m = re.findall(r"slotCurrentGroupChanged,curRow=\d+,curName=(.*?),previous", txt)
        return m[-1] if m else None

    def entrar_grupo(self, nome, timeout=20, tentativas=4):
        """Clica no grupo da biblioteca (o Revo sai do grupo quando reinicia) e confere pelo log que ficou nele:
        logo apos reiniciar, a lista de grupos ainda se reorganiza e o clique pode cair em outro grupo
        (Projeto 9: clicou Projeto9_v3 e 0,2 s depois o Revo trocou para 'peca_limpa')."""
        self.biblioteca()
        self.limpar_busca()
        for _ in range(tentativas):
            time.sleep(2)                          # deixa a lista assentar
            item = next((d for d in self.principal.descendants(control_type="ListItem") if d.window_text() == nome), None)
            if item is None:
                raise LookupError(f"grupo {nome!r} nao existe; grupos: {self.grupos()}")
            try:
                item.iface_selection_item.Select()
            except Exception:
                item.click_input()
            time.sleep(2)
            if self.grupo_atual() == nome:
                return
        raise RuntimeError(f"nao consegui ficar no grupo {nome!r} (atual: {self.grupo_atual()!r})")

    def captura(self, arquivo):
        """Salva a imagem da janela do Revo (para conferir o estado sem perguntar ao usuario)."""
        from PIL import ImageGrab
        r = self.principal.rectangle()
        ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom), all_screens=True).save(arquivo)
        return arquivo

    def _contagem(self):
        for d in self.principal.descendants(control_type="Text"):
            if d.window_text().startswith("Scan(") and d.is_visible():
                return int(d.window_text()[5:-1])
        return -1

    def _limpar_dialogos(self, titulo, fechar_seletor=True):
        """Dispensa caixas de erro (botao OK) e, se pedido, o seletor de arquivo que sobrou de uma falha."""
        for w in self.w32.windows(process=self.pid, class_name="#32770", title=titulo):
            ok = [b for b in w.children(class_name="Button") if b.window_text() in ("OK", "&OK")]
            if ok:
                ok[0].click()
                time.sleep(0.5)
            elif fechar_seletor:
                w.close()
                time.sleep(1)

    def _esperar_dialogo(self, titulo, timeout=20):
        self._esperar(lambda: self.w32.window(title=titulo, class_name="#32770").exists(), timeout, msg=f"dialogo {titulo!r} nao abriu")
        return self.w32.window(title=titulo, class_name="#32770")

    def carregar_projeto(self, revo, timeout=120):
        """File > Load Project: importa os scans do .revo para a biblioteca do Revo (copia os dados).
        Devolve {guid: pasta do scan na biblioteca} so para o que acabou de entrar (a biblioteca pode ter
        outros scans com o mesmo guid, de importacoes anteriores ou do original)."""
        self.biblioteca()
        next(d for d in self.principal.descendants(control_type="Edit") if d.is_visible()).set_edit_text("")  # filtro antigo
        dados = os.path.join(os.path.dirname(os.path.abspath(revo)), "data")
        # a pasta de marcadores (no type 5, ex.: Frame...) nao vira scan na biblioteca: nao conta
        nos = json.load(open(os.path.abspath(revo), encoding="utf-8")).get("nodes", [])
        esperados = sum(1 for n in nos if n.get("type") != 5 and os.path.isdir(os.path.join(dados, n["guid"])))
        pastas = set(os.listdir(BIBLIOTECA))
        self._limpar_dialogos("Load Project")
        next(d for d in self.principal.descendants() if d.window_text() == "Load Project").click_input()
        dlg = self._esperar_dialogo("Load Project")
        caminho = os.path.abspath(revo)
        for _ in range(4):
            time.sleep(1)
            edit = dlg.child_window(class_name="ComboBoxEx32").child_window(class_name="Edit")
            edit.set_edit_text(caminho)
            time.sleep(0.7)
            if edit.window_text() != caminho:          # o autocompletar do Windows ja alterou o texto
                continue
            dlg.child_window(class_name="Button", title_re=r"&?Abrir|&?Open").click()
            try:
                self._esperar(lambda: not dlg.exists(), 5, 0.3)
                break
            except TimeoutError:
                self._limpar_dialogos("Load Project", fechar_seletor=False)     # "arquivo nao encontrado"
        else:
            raise RuntimeError("nao consegui abrir o arquivo no dialogo Load Project")

        def novos():
            achados = {}
            for d in set(os.listdir(BIBLIOTECA)) - pastas:
                for f in os.listdir(os.path.join(BIBLIOTECA, d)):
                    if f.endswith(".revox"):
                        achados[f[:-6]] = os.path.join(BIBLIOTECA, d)
            return achados
        self._esperar(lambda: len(novos()) >= esperados, timeout, msg="scans nao entraram na biblioteca")
        time.sleep(3)                      # o Revo ainda grava miniatura/indice depois de criar a pasta
        return novos()

    def abrir_scan(self, nome, pasta=None, timeout=1200, grupo=None):
        """Filtra a biblioteca pelo nome (caixa de busca) e abre o primeiro cartao (duplo clique).
        Os cartoes sao desenhados sem texto na arvore UIA, entao a busca e o caminho confiavel.
        Com `pasta` (a que `carregar_projeto` devolveu), confere pelo log do Revo que foi essa que abriu.
        Com `grupo`, entra nele antes (nomes de scan se repetem entre importacoes)."""
        if grupo:
            self.entrar_grupo(grupo)               # ja limpa a busca antes de clicar no grupo
        else:
            self.biblioteca()
            self.limpar_busca()
        caixa = next(d for d in self.principal.descendants(control_type="Edit") if d.is_visible())
        caixa.set_edit_text(nome)
        # sem esperar o filtro, o duplo clique cai no 1o cartao da lista inteira (scan errado)
        self._esperar(lambda: self._contagem() >= 1, 30, 0.5, f"busca por {nome!r} nao achou cartao")
        time.sleep(2)
        if grupo and self.grupo_atual() != grupo:
            raise RuntimeError(f"saiu do grupo {grupo!r} antes de abrir (atual: {self.grupo_atual()!r})")
        log = LogRevo()
        # 1o cartao da grade (ListItem com a miniatura); a posicao fixa antiga dependia do tamanho da janela
        cartoes = [d for d in self.principal.descendants(control_type="ListItem")
                   if d.is_visible() and d.window_text() in ("false", "true")]
        if cartoes:
            c = min(cartoes, key=lambda d: (d.rectangle().top, d.rectangle().left)).rectangle()
            alvo = ((c.left + c.right) // 2, c.top + (c.bottom - c.top) // 3)
        else:
            r = self.principal.rectangle()
            alvo = (r.left + 580, r.top + 420)
        mouse.double_click(coords=alvo)                                         # miniatura do 1o cartao
        aberto = log.esperar(r"Project opened:(.+?\.revox)", timeout, "scan nao abriu").group(1)
        if pasta and os.path.normcase(os.path.dirname(aberto)) != os.path.normcase(os.path.abspath(pasta)):
            raise RuntimeError(f"abriu {aberto}, esperava a pasta {pasta}")
        self._esperar(lambda: any(d.window_text() == "Fusion" for d in self.principal.descendants(control_type="Button")),
                      timeout, msg="scan nao abriu")
        return os.path.dirname(aberto)

    # ---------------------------------------------------------------- etapas
    # O conteudo dos paineis (QMenu) nao aparece na arvore UIA, e a altura muda com as opcoes (malha rapida
    # x alta, Hole Filling, remover marcadores). Entao o Apply e achado na captura da janela: e o unico
    # retangulo de ~200 x 28 px no azul de destaque do Revo (0,136,255); a nuvem usa (1,93,158).
    AZUL = (0, 136, 255)

    def _posicao_apply(self, tentativas=5):
        import numpy as np
        from PIL import ImageGrab
        for _ in range(tentativas):
            r = self.principal.rectangle()
            a = np.asarray(ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom), all_screens=True).convert("RGB")).astype(int)
            m = (abs(a[:, :, 0] - self.AZUL[0]) < 20) & (abs(a[:, :, 1] - self.AZUL[1]) < 15) & (abs(a[:, :, 2] - self.AZUL[2]) < 15)
            linhas = np.flatnonzero(m.sum(axis=1) >= 150)
            if len(linhas):
                grupos = np.split(linhas, np.flatnonzero(np.diff(linhas) > 1) + 1)
                for g in grupos:
                    if 20 <= len(g) <= 45:
                        cols = np.flatnonzero(m[g[0]:g[-1] + 1].any(axis=0))
                        return r.left + (cols[0] + cols[-1]) // 2, r.top + (g[0] + g[-1]) // 2
            time.sleep(0.5)
        raise LookupError("botao Apply nao apareceu na tela")

    def _apply_do_painel(self, botao):
        from pywinauto.keyboard import send_keys
        for _ in range(3):                 # o 1o clique pode cair enquanto o Revo ainda carrega o scan
            send_keys("{ESC}")                                 # fecha painel que tenha ficado aberto
            time.sleep(1)
            b = self._botao(botao)
            if not b.is_enabled():
                raise RuntimeError(f"{botao} desabilitado: selecione uma nuvem fundida (Meshing) ou o scan (Fusion)")
            b.click_input()
            time.sleep(1.5)
            try:
                mouse.click(coords=self._posicao_apply(2))
                return
            except LookupError:
                continue
        raise LookupError(f"painel de {botao} nao abriu")

    # O fim de cada etapa vem do log do Revo, que a lista de modelos (virtualizada: so as linhas visiveis
    # entram na arvore UIA) nao entrega quando o scan ja tem muitos itens. O log tambem registra os
    # parametros realmente usados, o que serve de conferencia.
    def fundir(self, timeout=1800):
        """Edit > Fusion > Apply com o que estiver no painel (veja `parametros`). Devolve a linha do log
        com o passo de pontos usado e a duracao em ms."""
        log = LogRevo()
        self._apply_do_painel("Fusion")
        usado = log.esperar(r"point cloud fuse with pointPitch: ([\d.]+)", 30, "fusao nao comecou")
        fim = log.esperar(r"point cloud fuse end, duration: (\d+)", timeout, "fusao nao terminou")
        linha=usado.string[usado.start():].splitlines()[0]
        otim=re.search(r"optimizePose:\s*(\d+)",linha)
        return {"ponto_mm": float(usado.group(1)), "ms": int(fim.group(1)),
                "otimizar_poses":int(otim.group(1)) if otim else None}

    def malha(self, timeout=3600):
        """Edit > Meshing > Apply com o modo/valores que o painel mostra (veja `parametros`)."""
        log = LogRevo()
        self._apply_do_painel("Meshing")
        usado = log.esperar(r"meshingHelper\] - meshMode:(\d), optimizeLevel:(\d), fillHoleEnabled:(\d), "
                            r"maxHoleAreaRatio:[\d.]+, meshEdgeLength:([\d.]+)", 30, "malha nao comecou")
        fim = log.esperar(r"triMesh meshing end, duration: (\d+), result: (-?\d+)", timeout, "malha nao terminou")
        if int(fim.group(2)) != 0:
            raise RuntimeError(f"malha falhou (result {fim.group(2)})")
        modo, otim, furos, grade = usado.groups()
        return {"modo": int(modo), "otimizar": int(otim), "furos": bool(int(furos)), "grade_mm": float(grade),
                "ms": int(fim.group(1))}

    def exportar(self, pasta, prefixos=("Fused", "Mesh"), timeout=600):
        """Export Model (.ply): marca na lista do dialogo, para cada prefixo, o item mais recente (maior
        contador no nome do Revo: 'Fused 1021_3 (0.50mm)', 'Mesh 1021_2'). Itens importados do projeto
        ('fuse', 'fuse_mesh') ficam de fora. Devolve os arquivos criados."""
        from pywinauto.keyboard import send_keys
        pasta = os.path.abspath(pasta)
        os.makedirs(pasta, exist_ok=True)
        antes = set(os.listdir(pasta))
        send_keys("{ESC}")                                     # painel de malha/fusao pode ter ficado aberto
        self._esperar(lambda: not self._travado(self.pid), 300, 2.0, "Revo nao voltou a responder")   # malha grande: a tela demora
        time.sleep(2)
        self._botao("Export Model").click_input()
        self._esperar(lambda: self._botao("Confirm"), 90, 0.5, "dialogo de exportacao nao abriu")
        time.sleep(0.5)
        melhor = {}
        for d in self.principal.descendants(control_type="ListItem"):
            nome = d.window_text()
            if d.is_visible() and "TextureExportDialog" in d.element_info.automation_id and nome.startswith(prefixos):
                n = re.search(r"_(\d+)", nome)
                chave = (int(n.group(1)) if n else 0)
                pref = next(p for p in prefixos if nome.startswith(p))
                if pref not in melhor or chave > melhor[pref][0]:
                    melhor[pref] = (chave, d)
        if len(melhor) != len(prefixos):
            raise RuntimeError(f"esperava itens {prefixos} para exportar e achei {[v[1].window_text() for v in melhor.values()]}")
        marcados = []
        for _, d in melhor.values():
            r = d.rectangle()
            mouse.click(coords=(r.left + 12, (r.top + r.bottom) // 2))           # caixa de marcar, borda esquerda da linha
            marcados.append(d.window_text())
            time.sleep(0.4)
        self._botao("Confirm").click_input()
        dlg = self._esperar_dialogo("Export Model").wrapper_object()
        ed = next(d for d in dlg.descendants() if d.class_name() == "Edit" and d.is_visible())
        botao = next(d for d in dlg.descendants() if d.class_name() == "Button" and d.window_text() == "Selecionar pasta")
        for _ in range(4):
            time.sleep(1)
            ed.set_edit_text(pasta)
            time.sleep(0.7)
            if ed.window_text() != pasta:                      # autocompletar do Windows alterou o texto
                continue
            for clique in (botao.click_input, botao.click):
                clique()
                try:
                    self._esperar(lambda: not self.w32.window(title="Export Model", class_name="#32770").exists(), 5, 0.3)
                    break
                except TimeoutError:
                    continue
            else:
                continue
            break
        else:
            raise RuntimeError("nao consegui confirmar a pasta no seletor de exportacao")
        self._esperar(lambda: self._janelas("RvMessageBox"), timeout, 1.0, "exportacao nao terminou")
        self._fechar_avisos()
        return {"itens": marcados, "arquivos": sorted(set(os.listdir(pasta)) - antes)}
