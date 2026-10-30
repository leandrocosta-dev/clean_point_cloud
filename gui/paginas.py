"""Paginas da interface."""
import datetime
import json
import os
import shutil
import subprocess
import threading
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QFrame, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QMenu,
                               QMessageBox, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

import runner as RN
from core import (PRESETS_DIR, cabecalho_ply, citar, dividir, humano, ler_projeto, listar_projetos, substituir, tamanho_pasta)
from receitas import RECEITAS
from widgets import CampoTexto, Combo, FormScript, ScanPicker, SpinDbl, SpinInt, cartao


def titulo_pagina(titulo, sub):
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 8)
    lay.setSpacing(2)
    a = QLabel(titulo)
    a.setObjectName("h1")
    b = QLabel(sub)
    b.setObjectName("suave")
    b.setWordWrap(True)
    lay.addWidget(a)
    lay.addWidget(b)
    return w


def rolavel(widget):
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setFrameShape(QFrame.NoFrame)
    sc.setWidget(widget)
    return sc


def pagina_base(titulo, sub, margens=(28, 22, 28, 12)):
    w = QWidget()
    w.setObjectName("pagina")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(*margens)
    lay.setSpacing(12)
    lay.addWidget(titulo_pagina(titulo, sub))
    return w, lay


# =============================================================================== console
class Console(QPlainTextEdit):
    def __init__(self, pal):
        super().__init__()
        self.setObjectName("console")
        self.setReadOnly(True)
        self.setMaximumBlockCount(40000)
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.pal = pal

    def escrever(self, texto, tipo="out"):
        cor = {"out": self.pal["texto"], "info": self.pal["acento"], "erro": self.pal["erro"], "ok": self.pal["ok"]}.get(tipo, self.pal["texto"])
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(cor))
        bar = self.verticalScrollBar()
        no_fim = bar.value() >= bar.maximum() - 4
        cur = QTextCursor(self.document())
        cur.movePosition(QTextCursor.End)
        cur.insertText(texto + "\n", fmt)
        if no_fim:
            bar.setValue(bar.maximum())


# =============================================================================== projeto
class _Sinal(QObject):
    tamanho = Signal(int, int)


class PaginaProjeto(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.sinal = _Sinal()
        self.sinal.tamanho.connect(self._tamanho)
        self.scans = []
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 12)
        lay.setSpacing(12)
        lay.addWidget(titulo_pagina("Projeto e contexto", "Escolha o projeto do Revo Scan e as pastas de trabalho. Os valores viram variáveis "
                                    "({PROJETO}, {SAIDA}, {TRABALHO}, {NOME}) que podem ser usadas em qualquer campo de caminho."))
        c = app.estado.ctx
        f, fl = cartao("Pastas")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.raiz = CampoTexto(app=app, placeholder="pasta que contém os projetos (ex.: D:\\Projects)")
        self.raiz.ed.setText(c["raiz"])
        self.combo = Combo()
        self.combo.setEditable(True)
        self.combo.lineEdit().setPlaceholderText("nenhum projeto selecionado")
        self.combo.setMinimumWidth(420)
        bt = QPushButton("Atualizar")
        bt.setObjectName("mini")
        bt.clicked.connect(self.recarregar)
        ln = QHBoxLayout()
        bp = QPushButton("Escolher projeto…")
        bp.setObjectName("mini")
        bp.clicked.connect(self._escolher_projeto)
        ln.addWidget(self.combo, 1)
        ln.addWidget(bp)
        ln.addWidget(bt)
        self.saida = CampoTexto(app=app, placeholder="onde salvar os entregáveis")
        self.saida.ed.setText(c["saida"])
        self.trab = CampoTexto(app=app, placeholder="onde ficam as pastas intermediárias")
        self.trab.ed.setText(c["trabalho"])
        form.addRow("Pasta dos projetos", self.raiz)
        form.addRow("Projeto (.revo)", ln)
        form.addRow("Salvar entregáveis em  {SAIDA}", self.saida)
        form.addRow("Pasta de trabalho  {TRABALHO}", self.trab)
        fl.addLayout(form)
        self.aviso = QLabel("")
        self.aviso.setObjectName("ajuda")
        fl.addWidget(self.aviso)
        lay.addWidget(f)

        f2, fl2 = cartao("Scans do projeto", "Marque os scans que as ferramentas devem usar. A seleção alimenta os campos de scans e as receitas.")
        self.tab = QTableWidget(0, 6)
        self.tab.setHorizontalHeaderLabels(["", "Scan", "Formato", "fuse.ply", "fuse_mesh.ply", "Tamanho"])
        self.tab.verticalHeader().setVisible(False)
        self.tab.setAlternatingRowColors(True)
        self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        h = self.tab.horizontalHeader()
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        self.tab.setColumnWidth(0, 36)
        for i in (2, 3, 4, 5):
            self.tab.setColumnWidth(i, 120)
        self.tab.itemChanged.connect(self._marcou)
        fl2.addWidget(self.tab, 1)
        bl = QHBoxLayout()
        for txt, fn in (("Todos", lambda: self._todos(True)), ("Nenhum", lambda: self._todos(False)),
                        ("Só frames.dataset", self._so_novos), ("Abrir pasta do projeto", self._abrir)):
            b = QPushButton(txt)
            b.setObjectName("mini")
            b.clicked.connect(fn)
            bl.addWidget(b)
        bl.addStretch()
        self.resumo = QLabel("")
        self.resumo.setObjectName("suave")
        bl.addWidget(self.resumo)
        fl2.addLayout(bl)
        lay.addWidget(f2, 1)

        for w in (self.raiz.ed, self.saida.ed, self.trab.ed):
            w.textChanged.connect(self._salvar)
        self.raiz.ed.editingFinished.connect(self.recarregar)
        self._t_raiz = QTimer(self)
        self._t_raiz.setSingleShot(True)
        self._t_raiz.timeout.connect(self.recarregar)
        self.raiz.ed.textChanged.connect(lambda _: self._t_raiz.start(500))
        self.combo.currentTextChanged.connect(self._projeto)
        self.recarregar()

    def _salvar(self):
        c = self.app.estado.ctx
        c["raiz"], c["saida"], c["trabalho"] = self.raiz.ed.text().strip(), self.saida.ed.text().strip(), self.trab.ed.text().strip()
        self.app.salvar_depois()

    def _escolher_projeto(self):
        p = QFileDialog.getExistingDirectory(self, "Pasta do projeto (contém o .revo)", self.raiz.ed.text() or "")
        if p:
            self.raiz.ed.setText(os.path.normpath(p))      # a recarga automatica seleciona o projeto

    def recarregar(self):
        projs = listar_projetos(self.raiz.ed.text())
        atual = os.path.normpath(self.app.estado.ctx.get("projeto", "") or ".")
        self.combo.blockSignals(True)
        self.combo.clear()
        for p in projs:
            self.combo.addItem(p)
        escolha = atual if atual in projs else (projs[0] if len(projs) == 1 else "")
        self.combo.setCurrentText(escolha)
        self.combo.blockSignals(False)
        if projs:
            self.aviso.setText(f"{len(projs)} projeto(s) encontrado(s)" + ("" if escolha else " — escolha um na lista."))
        else:
            self.aviso.setText("Nenhum .revo encontrado nessa pasta nem até 2 níveis abaixo dela. Confira o caminho.")
        self._projeto(escolha)

    def _projeto(self, caminho):
        caminho = caminho.strip()
        self.app.estado.ctx["projeto"] = caminho
        revo, scans = ler_projeto(caminho) if caminho and os.path.isdir(caminho) else (None, [])
        self.scans = scans
        sel = set(self.app.estado.ctx.get("scans", []))
        self.tab.blockSignals(True)
        self.tab.setRowCount(len(scans))
        for r, s in enumerate(scans):
            it = QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Checked if s["guid"] in sel else Qt.Unchecked)
            self.tab.setItem(r, 0, it)
            for c, v in ((1, s["guid"]), (2, s["formato"]), (3, "sim" if s["fuse"] else "não"), (4, "sim" if s["malha"] else "não"), (5, "…")):
                self.tab.setItem(r, c, QTableWidgetItem(v))
        self.tab.blockSignals(False)
        self.resumo.setText(f"{len(scans)} scan(s)" + (f"  ·  {revo}" if revo else ""))
        self.app.salvar_depois()
        if scans:
            threading.Thread(target=self._calcula_tamanhos, args=(list(scans),), daemon=True).start()
        self.app.contexto_mudou()

    def _calcula_tamanhos(self, scans):
        for r, s in enumerate(scans):
            self.sinal.tamanho.emit(r, tamanho_pasta(s["pasta"]))

    def _tamanho(self, r, n):
        if r < self.tab.rowCount():
            self.tab.item(r, 5).setText(humano(n))

    def _marcou(self, _):
        self.app.estado.ctx["scans"] = [self.tab.item(r, 1).text() for r in range(self.tab.rowCount())
                                        if self.tab.item(r, 0).checkState() == Qt.Checked]
        self.resumo.setText(f"{len(self.scans)} scan(s)  ·  {len(self.app.estado.ctx['scans'])} selecionado(s)")
        self.app.salvar_depois()

    def marcar(self, guids):
        """Marca na tabela so estes scans (lista vazia = todos). Usado pela pagina Inicio."""
        self.tab.blockSignals(True)
        for r in range(self.tab.rowCount()):
            self.tab.item(r, 0).setCheckState(Qt.Checked if (not guids or self.tab.item(r, 1).text() in guids) else Qt.Unchecked)
        self.tab.blockSignals(False)
        self.app.estado.ctx["scans"] = list(guids)
        self.resumo.setText(f"{len(self.scans)} scan(s)  ·  {len(guids) or len(self.scans)} selecionado(s)")
        self.app.salvar_depois()

    def _todos(self, on):
        self.tab.blockSignals(True)
        for r in range(self.tab.rowCount()):
            self.tab.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.tab.blockSignals(False)
        self._marcou(None)

    def _so_novos(self):
        self.tab.blockSignals(True)
        for r, s in enumerate(self.scans):
            self.tab.item(r, 0).setCheckState(Qt.Checked if s["formato"] == "frames.dataset" else Qt.Unchecked)
        self.tab.blockSignals(False)
        self._marcou(None)

    def _abrir(self):
        p = self.app.estado.ctx.get("projeto")
        if p and os.path.isdir(p):
            os.startfile(p)


# =============================================================================== script
class PaginaScript(QWidget):
    """Formulario completo de um script + barra de comando/acoes."""

    def __init__(self, item, app, modelos=None):
        super().__init__()
        self.item, self.app = item, app
        self.modelos = modelos or {}
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(10)

        cab = QHBoxLayout()
        t = QLabel(item["titulo"])
        t.setObjectName("h1")
        chip = QLabel(item["categoria"])
        chip.setObjectName("chip")
        cab.addWidget(t)
        cab.addWidget(chip, 0, Qt.AlignVCenter)
        cab.addStretch()
        raiz.addLayout(cab)
        sub = QLabel(item["rel"] + ("   ·   parâmetros descritos manualmente (lê sys.argv)" if item["manual"] else f"   ·   {len(item['params'])} parâmetros lidos do script"))
        sub.setObjectName("ajuda")
        raiz.addWidget(sub)
        doc = (item["descricao"] or "").strip()
        if doc:
            self.doc = QLabel(doc)
            self.doc.setObjectName("suave")
            self.doc.setWordWrap(True)
            self.doc.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.curto = len(doc.splitlines()) > 4 or len(doc) > 420
            self.mais = QPushButton("mostrar tudo")
            self.mais.setObjectName("fantasma")
            self.mais.clicked.connect(self._alterna_doc)
            self._doc_completo = doc
            self._alterna_doc(forcar=False)
            raiz.addWidget(self.doc)
            if self.curto:
                raiz.addWidget(self.mais, 0, Qt.AlignLeft)

        self.form = FormScript(item, app)
        raiz.addWidget(rolavel(self.form), 1)

        f, fl = cartao()
        fl.setContentsMargins(14, 10, 14, 10)
        fl.setSpacing(8)
        self.previa = QPlainTextEdit()
        self.previa.setReadOnly(True)
        self.previa.setFixedHeight(66)
        self.previa.setStyleSheet("font-family:'Cascadia Mono',Consolas,monospace;font-size:12px")
        fl.addWidget(self.previa)
        bl = QHBoxLayout()
        self.b_run = QPushButton("▶  Executar agora")
        self.b_run.setObjectName("primario")
        self.b_run.clicked.connect(self.executar)
        b_fila = QPushButton("＋ Adicionar à fila")
        b_fila.clicked.connect(self.para_fila)
        b_copia = QPushButton("Copiar comando")
        b_copia.clicked.connect(self.copiar)
        b_pre = QPushButton("Presets  ▾")
        self.menu_pre = QMenu(self)
        b_pre.setMenu(self.menu_pre)
        self.menu_pre.aboutToShow.connect(self._monta_menu)
        b_ctx = QPushButton("Preencher do contexto")
        b_ctx.setToolTip("Sugere projeto e pastas de saída nos campos vazios")
        b_ctx.clicked.connect(self.form.preencher_contexto)
        b_rst = QPushButton("Restaurar padrões")
        b_rst.clicked.connect(self.form.restaurar)
        for b in (self.b_run, b_fila, b_copia, b_pre, b_ctx, b_rst):
            bl.addWidget(b)
        bl.addStretch()
        fl.addLayout(bl)
        raiz.addWidget(f)

        self.form.mudou.connect(self._atualiza)
        salvo = app.estado.d["form"].get(item["id"])
        if salvo:
            self.form.aplicar(salvo)
        else:
            self.form.preencher_contexto()
        self._atualiza()

    def _alterna_doc(self, forcar=None):
        if forcar is False or (forcar is None and self.mais.text() == "ocultar"):
            linhas = self._doc_completo.splitlines()
            self.doc.setText("\n".join(linhas[:4]) + (" …" if len(linhas) > 4 else ""))
            if hasattr(self, "mais"):
                self.mais.setText("mostrar tudo")
        else:
            self.doc.setText(self._doc_completo)
            self.mais.setText("ocultar")

    # ---------------------------------------------------------------- comando
    def comando(self):
        return [self.app.estado.env["python"], "-u", self.item["caminho"], *self.form.argv()]

    def _atualiza(self):
        try:
            self.previa.setPlainText(citar(self.comando()))
            self.previa.setStyleSheet("font-family:'Cascadia Mono',Consolas,monospace;font-size:12px")
            ok = True
        except ValueError as e:
            self.previa.setPlainText("⚠ " + str(e))
            self.previa.setStyleSheet(f"font-family:'Cascadia Mono',Consolas,monospace;font-size:12px;color:{self.app.pal['aviso']}")
            ok = False
        self.b_run.setEnabled(True)
        self.app.estado.d["form"][self.item["id"]] = self.form.valores()
        self.app.salvar_depois()
        return ok

    def _job(self):
        try:
            argv = self.comando()
        except ValueError as e:
            QMessageBox.warning(self, "Parâmetros pendentes", str(e))
            return None
        # os scripts recusam destino que ja tem conteudo (nunca sobrescrevem): avisa antes de gastar a execucao
        ocupados = [argv[i + 1] for i, a in enumerate(argv[:-1]) if a in ("--saida", "--saida-limpo", "--saida-uniao")
                    and os.path.exists(argv[i + 1]) and (not os.path.isdir(argv[i + 1]) or os.listdir(argv[i + 1]))]
        if ocupados:
            QMessageBox.warning(self, "Destino já existe", "Estas pastas já existem e têm conteúdo; o script se recusa a sobrescrever.\n"
                                "Escolha um nome novo ou apague/mova a pasta:\n\n" + "\n".join(ocupados))
            return None
        return {"nome": self.item["titulo"], "argv": argv}

    def executar(self):
        j = self._job()
        if j:
            self.app.executar([j])

    def para_fila(self):
        j = self._job()
        if j:
            self.app.adicionar_fila([j])

    def copiar(self):
        QApplication.clipboard().setText(self.previa.toPlainText())
        self.app.aviso("Comando copiado")

    # ---------------------------------------------------------------- presets
    def _pasta_presets(self):
        return os.path.join(PRESETS_DIR, self.item["id"].replace("/", "_").replace(".py", ""))

    def _monta_menu(self):
        m = self.menu_pre
        m.clear()
        if self.modelos:
            m.addSection("Modelos prontos")
            for nome, mapa in self.modelos.items():
                m.addAction(nome, lambda mapa=mapa: self.form.aplicar_valores(mapa))
            m.addSeparator()
        m.addAction("Salvar valores atuais como…", self._salvar_preset)
        pasta = self._pasta_presets()
        nomes = sorted(f[:-5] for f in os.listdir(pasta) if f.endswith(".json")) if os.path.isdir(pasta) else []
        if nomes:
            m.addSection("Meus presets")
            for n in nomes:
                sub = m.addMenu(n)
                sub.addAction("Carregar", lambda n=n: self._carregar_preset(n))
                sub.addAction("Excluir", lambda n=n: self._excluir_preset(n))
        m.addSeparator()
        m.addAction("Exportar para arquivo…", self._exportar_preset)
        m.addAction("Importar de arquivo…", self._importar_preset)

    def _salvar_preset(self):
        nome, ok = QInputDialog.getText(self, "Salvar preset", "Nome do preset:")
        if ok and nome.strip():
            os.makedirs(self._pasta_presets(), exist_ok=True)
            with open(os.path.join(self._pasta_presets(), nome.strip() + ".json"), "w", encoding="utf-8") as f:
                json.dump(self.form.valores(), f, indent=1, ensure_ascii=False)
            self.app.aviso(f"Preset '{nome}' salvo")

    def _carregar_preset(self, n):
        with open(os.path.join(self._pasta_presets(), n + ".json"), encoding="utf-8") as f:
            self.form.aplicar(json.load(f))

    def _excluir_preset(self, n):
        os.remove(os.path.join(self._pasta_presets(), n + ".json"))

    def _exportar_preset(self):
        p, _ = QFileDialog.getSaveFileName(self, "Exportar valores", self.item["id"].replace("/", "_") + ".json", "JSON (*.json)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.form.valores(), f, indent=1, ensure_ascii=False)

    def _importar_preset(self):
        p, _ = QFileDialog.getOpenFileName(self, "Importar valores", "", "JSON (*.json)")
        if p:
            with open(p, encoding="utf-8") as f:
                self.form.aplicar(json.load(f))


class PaginaScripts(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.paginas = {}
        lay = QHBoxLayout(self)
        lay.setContentsMargins(28, 22, 0, 12)
        lay.setSpacing(16)
        esq = QVBoxLayout()
        esq.addWidget(titulo_pagina("Ferramentas", f"{len(app.itens)} scripts do pipeline, com todos os parâmetros."))
        self.busca = QLineEdit()
        self.busca.setPlaceholderText("Buscar script ou parâmetro…")
        self.busca.setClearButtonEnabled(True)
        self.busca.textChanged.connect(self._filtra)
        esq.addWidget(self.busca)
        self.arvore = QTreeWidget()
        self.arvore.setHeaderHidden(True)
        self.arvore.setIndentation(14)
        self.arvore.setFixedWidth(320)
        cats = {}
        for it in app.itens:
            if it["categoria"] not in cats:
                n = QTreeWidgetItem([it["categoria"]])
                n.setFlags(Qt.ItemIsEnabled)
                f = n.font(0)
                f.setBold(True)
                n.setFont(0, f)
                self.arvore.addTopLevelItem(n)
                cats[it["categoria"]] = n
            c = QTreeWidgetItem([it["titulo"]])
            c.setData(0, Qt.UserRole, it["id"])
            c.setToolTip(0, it["rel"])
            cats[it["categoria"]].addChild(c)
        self.arvore.expandAll()
        self.arvore.currentItemChanged.connect(self._escolhe)
        esq.addWidget(self.arvore, 1)
        lay.addLayout(esq)
        self.pilha = QStackedWidget()
        vazio = QLabel("Escolha uma ferramenta na lista.")
        vazio.setObjectName("suave")
        vazio.setAlignment(Qt.AlignCenter)
        self.pilha.addWidget(vazio)
        dire = QVBoxLayout()
        dire.setContentsMargins(0, 0, 28, 0)
        dire.addWidget(self.pilha)
        lay.addLayout(dire, 1)

    def _filtra(self, txt):
        txt = txt.lower().strip()
        achados = {i["id"] for i in self.app.itens if not txt or txt in i["titulo"].lower() or txt in i["rel"].lower()
                   or any(txt in (p["flag"] or p["dest"]).lower() or txt in p["ajuda"].lower() for p in i["params"])}
        for i in range(self.arvore.topLevelItemCount()):
            cat = self.arvore.topLevelItem(i)
            vis = 0
            for j in range(cat.childCount()):
                c = cat.child(j)
                oculto = c.data(0, Qt.UserRole) not in achados
                c.setHidden(oculto)
                vis += not oculto
            cat.setHidden(vis == 0)

    def _escolhe(self, atual, _):
        if not atual or not atual.data(0, Qt.UserRole):
            return
        self.abrir(atual.data(0, Qt.UserRole))

    def abrir(self, id_):
        item = next(i for i in self.app.itens if i["id"] == id_)
        if id_ not in self.paginas:
            self.paginas[id_] = PaginaScript(item, self.app)
            self.pilha.addWidget(self.paginas[id_])
        self.pilha.setCurrentWidget(self.paginas[id_])
        for i in range(self.arvore.topLevelItemCount()):
            for j in range(self.arvore.topLevelItem(i).childCount()):
                c = self.arvore.topLevelItem(i).child(j)
                if c.data(0, Qt.UserRole) == id_ and self.arvore.currentItem() is not c:
                    self.arvore.setCurrentItem(c)


# =============================================================================== pipeline
MODELOS_PIPELINE = {
    "Só limpar → entregar .revo limpo (sem Revo)": {"so_limpar": True, "periferia_mm": 5.0, "periferia_banda": 12.0, "periferia_alargue": 10.0},
    "Rápido (malha rápida, ponto automático)": {"malha": "rapida", "otimizar": "baixo"},
    "Alta resolução 0,1 mm": {"ponto": 0.1, "malha": "alta", "grade": 0.1, "furos": True},
    "Resolução média 0,4 mm": {"ponto": 0.4, "malha": "alta", "grade": 0.4},
    "Remover mesa (receita validada 5/12/10)": {"periferia_mm": 5.0, "periferia_banda": 12.0, "periferia_alargue": 10.0},
    "Só nuvem (sem malha)": {"entregaveis": "nuvem"},
    "Duas partes viradas (lados opostos)": {"lados": "opostos"},
}


def pagina_pipeline(app):
    item = next(i for i in app.itens if i["id"] == "rodar_tudo.py")
    w = QWidget()
    lay = QVBoxLayout(w)
    lay.setContentsMargins(28, 22, 28, 12)
    lay.addWidget(PaginaScript({**item, "titulo": "Pipeline completo", "categoria": "Receitas A e B"}, app, MODELOS_PIPELINE), 1)
    return w


# =============================================================================== fila + console
class PaginaFila(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 12)
        lay.setSpacing(10)
        lay.addWidget(titulo_pagina("Fila e execução", "As etapas rodam uma de cada vez, em prioridade baixa. Em caso de crash do Open3D a etapa é repetida "
                                    "(Ambiente → tentativas)."))
        sp = QSplitter(Qt.Vertical)
        topo = QWidget()
        tl = QVBoxLayout(topo)
        tl.setContentsMargins(0, 0, 0, 0)
        self.abas = QTabWidget()
        self.tab = QTableWidget(0, 3)
        self.tab.setHorizontalHeaderLabels(["#", "Etapa", "Comando"])
        self.tab.verticalHeader().setVisible(False)
        self.tab.setAlternatingRowColors(True)
        self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab.setColumnWidth(0, 40)
        self.tab.setColumnWidth(1, 300)
        self.tab.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.tab.cellDoubleClicked.connect(self.editar)
        self.hist = QTableWidget(0, 5)
        self.hist.setHorizontalHeaderLabels(["Quando", "Etapa", "Código", "Duração", "Comando"])
        self.hist.verticalHeader().setVisible(False)
        self.hist.setAlternatingRowColors(True)
        self.hist.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.hist.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.hist.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.hist.setColumnWidth(0, 150)
        self.hist.setColumnWidth(1, 240)
        self.hist.setColumnWidth(2, 70)
        self.hist.setColumnWidth(3, 90)
        caixa_f = QWidget()
        cf = QVBoxLayout(caixa_f)
        cf.setContentsMargins(0, 10, 0, 0)
        cf.addWidget(self.tab)
        bl = QHBoxLayout()
        self.b_run = QPushButton("▶  Executar fila")
        self.b_run.setObjectName("primario")
        self.b_run.clicked.connect(self.rodar)
        self.b_stop = QPushButton("■  Parar")
        self.b_stop.setObjectName("perigo")
        self.b_stop.clicked.connect(app.runner.parar)
        self.b_stop.setEnabled(False)
        bl.addWidget(self.b_run)
        bl.addWidget(self.b_stop)
        for txt, fn in (("↑", lambda: self.mover(-1)), ("↓", lambda: self.mover(1)), ("Editar", lambda: self.editar(self.tab.currentRow(), 0)),
                        ("Remover", self.remover), ("Limpar", self.limpar)):
            b = QPushButton(txt)
            b.setObjectName("mini")
            b.clicked.connect(fn)
            bl.addWidget(b)
        bl.addSpacing(12)
        bm = QPushButton("Modelos de receita  ▾")
        m = QMenu(self)
        for nome, fn in RECEITAS:
            m.addAction(nome, lambda fn=fn: self.modelo(fn))
        bm.setMenu(m)
        bl.addWidget(bm)
        bs = QPushButton("Salvar fila…")
        bs.clicked.connect(self.salvar_fila)
        bc = QPushButton("Carregar fila…")
        bc.clicked.connect(self.carregar_fila)
        bl.addWidget(bs)
        bl.addWidget(bc)
        bl.addStretch()
        self.parar_erro = QCheckBox("parar no primeiro erro")
        self.parar_erro.setChecked(bool(app.estado.env.get("parar_no_erro", True)))
        self.parar_erro.toggled.connect(lambda v: app.estado.env.__setitem__("parar_no_erro", v))
        bl.addWidget(self.parar_erro)
        cf.addLayout(bl)
        caixa_h = QWidget()
        ch = QVBoxLayout(caixa_h)
        ch.setContentsMargins(0, 10, 0, 0)
        ch.addWidget(self.hist)
        bh = QHBoxLayout()
        b1 = QPushButton("Reenviar para a fila")
        b1.clicked.connect(self.reenviar)
        b2 = QPushButton("Apagar histórico")
        b2.clicked.connect(self.apagar_hist)
        bh.addWidget(b1)
        bh.addWidget(b2)
        bh.addStretch()
        ch.addLayout(bh)
        self.abas.addTab(caixa_f, "Fila")
        self.abas.addTab(caixa_h, "Histórico")
        self.abas.currentChanged.connect(lambda i: self.carregar_hist() if i == 1 else None)
        tl.addWidget(self.abas)
        sp.addWidget(topo)
        self.console = Console(app.pal)
        base = QWidget()
        bl2 = QVBoxLayout(base)
        bl2.setContentsMargins(0, 8, 0, 0)
        cab = QHBoxLayout()
        lbl = QLabel("Saída")
        lbl.setObjectName("h2")
        cab.addWidget(lbl)
        cab.addStretch()
        bcl = QPushButton("Limpar console")
        bcl.setObjectName("mini")
        bcl.clicked.connect(self.console.clear)
        bcp = QPushButton("Copiar tudo")
        bcp.setObjectName("mini")
        bcp.clicked.connect(lambda: QApplication.clipboard().setText(self.console.toPlainText()))
        cab.addWidget(bcl)
        cab.addWidget(bcp)
        bl2.addLayout(cab)
        bl2.addWidget(self.console, 1)
        sp.addWidget(base)
        sp.setSizes([320, 360])
        lay.addWidget(sp, 1)
        self.refresh()

    # ------------------------------------------------------------ fila
    @property
    def fila(self):
        return self.app.estado.d["fila"]

    def refresh(self):
        self.tab.setRowCount(len(self.fila))
        for r, j in enumerate(self.fila):
            self.tab.setItem(r, 0, QTableWidgetItem(str(r + 1)))
            self.tab.setItem(r, 1, QTableWidgetItem(j["nome"]))
            self.tab.setItem(r, 2, QTableWidgetItem(citar(j["argv"][2:])))
        self.app.salvar_depois()
        self.app.atualiza_contagem_fila(len(self.fila))

    def adicionar(self, jobs):
        self.fila.extend(jobs)
        self.refresh()

    def marcar(self, i, estado):
        cor = {"rodando": self.app.pal["acento"], "ok": self.app.pal["ok"], "erro": self.app.pal["erro"]}.get(estado)
        for c in range(3):
            it = self.tab.item(i, c)
            if it:
                it.setForeground(QColor(cor) if cor else QColor(self.app.pal["texto"]))
        if estado == "rodando":
            self.tab.selectRow(i)

    def rodar(self):
        if not self.fila:
            self.app.aviso("A fila está vazia")
            return
        for i in range(len(self.fila)):
            self.marcar(i, None)
        self.app.executar(list(self.fila), na_pagina_fila=True)

    def mover(self, d):
        r = self.tab.currentRow()
        n = r + d
        if r < 0 or not 0 <= n < len(self.fila):
            return
        self.fila[r], self.fila[n] = self.fila[n], self.fila[r]
        self.refresh()
        self.tab.selectRow(n)

    def remover(self):
        r = self.tab.currentRow()
        if r >= 0:
            del self.fila[r]
            self.refresh()

    def limpar(self):
        self.fila.clear()
        self.refresh()

    def editar(self, r, _c):
        if r < 0 or r >= len(self.fila):
            return
        j = self.fila[r]
        d = QDialog(self)
        d.setWindowTitle("Editar etapa")
        d.resize(900, 320)
        ly = QVBoxLayout(d)
        nome = QLineEdit(j["nome"])
        ly.addWidget(nome)
        ed = QPlainTextEdit(citar(j["argv"]))
        ed.setStyleSheet("font-family:'Cascadia Mono',Consolas,monospace;font-size:12px")
        ly.addWidget(ed, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(d.accept)
        bb.rejected.connect(d.reject)
        ly.addWidget(bb)
        if d.exec():
            j["nome"] = nome.text()
            j["argv"] = [a.strip('"') for a in dividir(ed.toPlainText())]
            self.refresh()

    def modelo(self, fn):
        try:
            jobs = fn(self.app)
        except ValueError as e:
            QMessageBox.information(self, "Modelo", str(e))
            return
        self.adicionar(jobs)
        self.app.aviso(f"{len(jobs)} etapa(s) adicionada(s) — confira os caminhos antes de executar")

    def salvar_fila(self):
        p, _ = QFileDialog.getSaveFileName(self, "Salvar fila", "fila.json", "JSON (*.json)")
        if p:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.fila, f, indent=1, ensure_ascii=False)

    def carregar_fila(self):
        p, _ = QFileDialog.getOpenFileName(self, "Carregar fila", "", "JSON (*.json)")
        if p:
            with open(p, encoding="utf-8") as f:
                self.adicionar(json.load(f))

    # ------------------------------------------------------------ historico
    def carregar_hist(self):
        h = list(reversed(RN.carregar_historico()))
        self._hist = h
        self.hist.setRowCount(len(h))
        for r, x in enumerate(h):
            vals = [x["quando"], x["nome"], str(x["codigo"]), RN._dur(x["segundos"]), citar(x["argv"][2:])]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if c == 2:
                    it.setForeground(QColor(self.app.pal["ok"] if x["codigo"] == 0 else self.app.pal["erro"]))
                self.hist.setItem(r, c, it)

    def reenviar(self):
        r = self.hist.currentRow()
        if r >= 0:
            x = self._hist[r]
            self.adicionar([{"nome": x["nome"], "argv": x["argv"]}])
            self.abas.setCurrentIndex(0)

    def apagar_hist(self):
        if QMessageBox.question(self, "Histórico", "Apagar todo o histórico de execuções?") == QMessageBox.Yes:
            try:
                os.remove(RN.ARQ_HISTORICO)
            except OSError:
                pass
            self.carregar_hist()


# =============================================================================== entregaveis
class PaginaEntregaveis(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 12)
        lay.setSpacing(12)
        lay.addWidget(titulo_pagina("Entregáveis", "Veja o que o pipeline gerou, abra no MeshInspector e monte a pasta final de entrega com um LEIA-ME."))
        f, fl = cartao("Pasta de resultados")
        ln = QHBoxLayout()
        self.origem = CampoTexto(app=app, placeholder="pasta com os .ply / .revo gerados")
        self.origem.ed.setText(app.estado.ctx["saida"])
        self.filtro = Combo()
        self.filtro.addItems(["Todos", ".ply", ".revo", ".json", ".md", ".png", ".stl/.obj"])
        self.filtro.currentIndexChanged.connect(self.listar)
        ba = QPushButton("Atualizar")
        ba.clicked.connect(self.listar)
        ln.addWidget(self.origem, 1)
        ln.addWidget(self.filtro)
        ln.addWidget(ba)
        fl.addLayout(ln)
        lay.addWidget(f)
        self.tab = QTableWidget(0, 6)
        self.tab.setHorizontalHeaderLabels(["", "Arquivo", "Tipo", "Tamanho", "Vértices", "Triângulos"])
        self.tab.verticalHeader().setVisible(False)
        self.tab.setAlternatingRowColors(True)
        self.tab.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tab.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tab.setColumnWidth(0, 36)
        self.tab.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        for i, w in ((2, 70), (3, 100), (4, 110), (5, 110)):
            self.tab.setColumnWidth(i, w)
        lay.addWidget(self.tab, 1)
        bl = QHBoxLayout()
        for txt, fn in (("Marcar todos", lambda: self._marca(True)), ("Desmarcar", lambda: self._marca(False)),
                        ("Abrir no MeshInspector", self.mesh), ("Mostrar na pasta", self.mostrar), ("Copiar caminho", self.copiar)):
            b = QPushButton(txt)
            b.setObjectName("mini")
            b.clicked.connect(fn)
            bl.addWidget(b)
        bl.addStretch()
        lay.addLayout(bl)
        f2, fl2 = cartao("Montar pasta de entrega", "Copia os arquivos marcados para uma pasta nova (nunca sobrescreve) e escreve o LEIA-ME.md.")
        l2 = QHBoxLayout()
        self.destino = CampoTexto(app=app, placeholder="pasta de entrega (será criada)")
        self.leia = QCheckBox("LEIA-ME.md")
        self.leia.setChecked(True)
        self.com_hist = QCheckBox("incluir comandos do histórico")
        self.com_hist.setChecked(True)
        bm = QPushButton("Montar entrega")
        bm.setObjectName("primario")
        bm.clicked.connect(self.montar)
        l2.addWidget(self.destino, 1)
        l2.addWidget(self.leia)
        l2.addWidget(self.com_hist)
        l2.addWidget(bm)
        fl2.addLayout(l2)
        lay.addWidget(f2)
        self.arquivos = []
        self.origem.ed.editingFinished.connect(self.listar)
        self.listar()

    def listar(self):
        raiz = substituir(self.origem.ed.text().strip(), self.app.estado.ctx)
        ext = {0: None, 1: (".ply",), 2: (".revo",), 3: (".json",), 4: (".md",), 5: (".png",), 6: (".stl", ".obj")}[self.filtro.currentIndex()]
        self.arquivos = []
        if os.path.isdir(raiz):
            for r, _, fs in os.walk(raiz):
                for n in fs:
                    p = os.path.join(r, n)
                    if ext is None and not n.lower().endswith((".ply", ".revo", ".json", ".md", ".png", ".stl", ".obj", ".txt", ".csv")):
                        continue
                    if ext and not n.lower().endswith(ext):
                        continue
                    self.arquivos.append(p)
                if len(self.arquivos) > 1500:
                    break
        self.arquivos.sort()
        self.raiz = raiz
        self.tab.setRowCount(len(self.arquivos))
        for r, p in enumerate(self.arquivos):
            it = QTableWidgetItem()
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            it.setCheckState(Qt.Unchecked)
            self.tab.setItem(r, 0, it)
            self.tab.setItem(r, 1, QTableWidgetItem(os.path.relpath(p, raiz)))
            self.tab.setItem(r, 2, QTableWidgetItem(os.path.splitext(p)[1]))
            try:
                self.tab.setItem(r, 3, QTableWidgetItem(humano(os.path.getsize(p))))
            except OSError:
                pass
            if p.lower().endswith(".ply"):
                i = cabecalho_ply(p)
                self.tab.setItem(r, 4, QTableWidgetItem(f"{i['vertices']:,}".replace(",", ".") if "vertices" in i else ""))
                self.tab.setItem(r, 5, QTableWidgetItem(f"{i['faces']:,}".replace(",", ".") if "faces" in i else ""))

    def _marca(self, on):
        for r in range(self.tab.rowCount()):
            self.tab.item(r, 0).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _sel(self):
        marc = [self.arquivos[r] for r in range(self.tab.rowCount()) if self.tab.item(r, 0).checkState() == Qt.Checked]
        if not marc and self.tab.currentRow() >= 0:
            marc = [self.arquivos[self.tab.currentRow()]]
        return marc

    def mesh(self):
        exe = self.app.estado.env["meshinspector"]
        sel = [p for p in self._sel() if p.lower().endswith((".ply", ".stl", ".obj"))]
        if not os.path.exists(exe):
            QMessageBox.warning(self, "MeshInspector", f"Não encontrado: {exe}\nAjuste em Ambiente.")
            return
        if len(sel) > 3:
            sel = sel[:3]
            self.app.aviso("Abrindo só os 3 primeiros")
        for p in sel:
            subprocess.Popen([exe, p])

    def mostrar(self):
        for p in self._sel()[:1]:
            subprocess.Popen(["explorer", "/select,", os.path.normpath(p)])

    def copiar(self):
        QApplication.clipboard().setText("\n".join(self._sel()))

    def montar(self):
        sel = self._sel()
        dest = substituir(self.destino.ed.text().strip(), self.app.estado.ctx)
        if not sel or not dest:
            QMessageBox.information(self, "Entrega", "Marque arquivos e informe a pasta de destino.")
            return
        if os.path.exists(dest) and os.listdir(dest):
            QMessageBox.warning(self, "Entrega", "A pasta de destino já existe e não está vazia. Escolha uma nova.")
            return
        os.makedirs(dest, exist_ok=True)
        linhas = []
        for p in sel:
            nome = os.path.basename(p)
            alvo = os.path.join(dest, nome)
            if os.path.exists(alvo):
                alvo = os.path.join(dest, os.path.basename(os.path.dirname(p)) + "_" + nome)
            shutil.copy2(p, alvo)
            i = cabecalho_ply(p) if p.lower().endswith(".ply") else {}
            linhas.append(f"| {os.path.basename(alvo)} | {humano(os.path.getsize(p))} | {i.get('vertices', '')} | {i.get('faces', '')} |")
        if self.leia.isChecked():
            md = [f"# Entrega — {os.path.basename(self.app.estado.ctx.get('projeto', '').rstrip(chr(92)) or dest)}",
                  f"\nGerada em {datetime.datetime.now():%Y-%m-%d %H:%M} pela GUI do pipeline.\n",
                  "| arquivo | tamanho | vértices | triângulos |", "|---|---|---|---|", *linhas]
            if self.com_hist.isChecked():
                h = RN.carregar_historico()[-12:]
                if h:
                    md += ["\n## Comandos executados (mais recentes)\n"] + [f"- `{x['quando']}` {x['nome']} (código {x['codigo']}): `{citar(x['argv'][2:])}`" for x in h]
            md += ["\n## Observações\n", "Preencha aqui o que foi e o que não foi verificado (cobertura, furos, conferência visual)."]
            with open(os.path.join(dest, "LEIA-ME.md"), "w", encoding="utf-8") as f:
                f.write("\n".join(md))
        self.app.aviso(f"Entrega montada: {len(sel)} arquivo(s)")
        os.startfile(dest)


# =============================================================================== json
class PaginaJson(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.getters = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 12)
        lay.setSpacing(12)
        lay.addWidget(titulo_pagina("Configurações JSON", "Edite arquivos de configuração (ex.: config.default.json, config.highres.json) como formulário."))
        ln = QHBoxLayout()
        self.combo = Combo()
        from catalog import PIPE
        self.combo.setMinimumWidth(320)
        self.combo.setEditable(True)
        for f in sorted(os.listdir(PIPE)):
            if f.endswith(".json"):
                self.combo.addItem(os.path.join(PIPE, f))
        bo = QPushButton("Abrir outro…")
        bo.clicked.connect(self.outro)
        bs = QPushButton("Salvar")
        bs.setObjectName("primario")
        bs.clicked.connect(self.salvar)
        bsc = QPushButton("Salvar como…")
        bsc.clicked.connect(lambda: self.salvar(True))
        ln.addWidget(self.combo, 1)
        ln.addWidget(bo)
        ln.addWidget(bsc)
        ln.addWidget(bs)
        lay.addLayout(ln)
        self.arv = QTreeWidget()
        self.arv.setHeaderLabels(["Chave", "Valor"])
        self.arv.setColumnWidth(0, 340)
        lay.addWidget(self.arv, 1)
        self.combo.currentTextChanged.connect(self.carregar)
        self.carregar(self.combo.currentText())

    def outro(self):
        p, _ = QFileDialog.getOpenFileName(self, "Abrir JSON", "", "JSON (*.json)")
        if p:
            self.combo.setCurrentText(p)

    def carregar(self, caminho):
        self.arv.clear()
        self.getters = {}
        try:
            with open(caminho, encoding="utf-8") as f:
                self.dados = json.load(f)
        except (OSError, ValueError):
            return
        self._monta(self.arv.invisibleRootItem(), self.dados, ())
        self.arv.expandAll()

    def _monta(self, pai, obj, caminho):
        for k, v in obj.items():
            it = QTreeWidgetItem(pai, [str(k), ""])
            c = caminho + (k,)
            if isinstance(v, dict):
                f = it.font(0)
                f.setBold(True)
                it.setFont(0, f)
                self._monta(it, v, c)
                continue
            if isinstance(v, bool):
                w = QCheckBox()
                w.setChecked(v)
                g = w.isChecked
            elif isinstance(v, int):
                w = SpinInt()
                w.setRange(-2_000_000_000, 2_000_000_000)
                w.setValue(v)
                g = w.value
            elif isinstance(v, float):
                w = SpinDbl()
                w.setDecimals(8)
                w.setRange(-1e12, 1e12)
                w.setValue(v)
                g = w.value
            elif isinstance(v, str):
                w = QLineEdit(v)
                g = w.text
            else:
                w = QLineEdit(json.dumps(v))
                g = (lambda w=w: json.loads(w.text()))
            self.arv.setItemWidget(it, 1, w)
            self.getters[c] = g

    def _dados(self):
        import copy
        d = copy.deepcopy(self.dados)
        for c, g in self.getters.items():
            o = d
            for k in c[:-1]:
                o = o[k]
            try:
                o[c[-1]] = g()
            except ValueError:
                raise ValueError(".".join(c))
        return d

    def salvar(self, como=False):
        caminho = self.combo.currentText()
        if como is True:
            caminho, _ = QFileDialog.getSaveFileName(self, "Salvar como", caminho, "JSON (*.json)")
            if not caminho:
                return
        try:
            d = self._dados()
        except ValueError as e:
            QMessageBox.warning(self, "JSON", f"Valor inválido em: {e}")
            return
        with open(caminho, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        self.app.aviso(f"Salvo: {caminho}")


# =============================================================================== ambiente
DEPENDENCIAS = ["numpy", "scipy", "open3d", "meshlib", "pywinauto", "psutil", "PIL", "matplotlib", "PySide6"]


class PaginaAmbiente(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        env = app.estado.env
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(28, 22, 28, 12)
        lay.setSpacing(12)
        lay.addWidget(titulo_pagina("Ambiente", "Como os comandos são executados. A máquina é sensível a carga sustentada (guia, seção 8): "
                                    "mantenha prioridade baixa e poucos threads."))
        f, fl = cartao("Execução")
        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)
        self.py = CampoTexto(app=app, multi=False)
        self.py.ed.setText(env["python"])
        self.cwd = CampoTexto(app=app)
        self.cwd.ed.setText(env["cwd"])
        self.omp = SpinInt()
        self.omp.setRange(0, 64)
        self.omp.setValue(int(env["omp_threads"]))
        self.omp.setToolTip("OMP/MKL/OpenBLAS threads por processo; 0 = não definir")
        self.tent = SpinInt()
        self.tent.setRange(0, 10)
        self.tent.setValue(int(env["tentativas_crash"]))
        self.prio = QCheckBox("Prioridade abaixo do normal (BELOW_NORMAL)")
        self.prio.setChecked(bool(env["baixa_prioridade"]))
        self.pe = QCheckBox("Parar a fila no primeiro erro")
        self.pe.setChecked(bool(env["parar_no_erro"]))
        self.mi = CampoTexto(app=app)
        self.mi.ed.setText(env["meshinspector"])
        self.rv = CampoTexto(app=app)
        self.rv.ed.setText(env["revo"])
        self.ee = QPlainTextEdit(env.get("extra_env", ""))
        self.ee.setFixedHeight(64)
        self.ee.setPlaceholderText("NOME=valor (um por linha)")
        form.addRow("Python (.venv)", self.py)
        form.addRow("Pasta de trabalho dos comandos", self.cwd)
        form.addRow("Threads (OMP_NUM_THREADS)", self.omp)
        form.addRow("Repetir em crash (tentativas)", self.tent)
        form.addRow("", self.prio)
        form.addRow("", self.pe)
        form.addRow("MeshInspector.exe", self.mi)
        form.addRow("RevoScan6.exe", self.rv)
        form.addRow("Variáveis de ambiente extras", self.ee)
        fl.addLayout(form)
        lay.addWidget(f)
        f2, fl2 = cartao("Aparência")
        self.tema = Combo()
        self.tema.addItems(["escuro", "claro"])
        self.tema.setCurrentText(app.estado.d.get("tema", "escuro"))
        self.tema.currentTextChanged.connect(app.mudar_tema)
        fl2.addWidget(self.tema)
        lay.addWidget(f2)
        f3, fl3 = cartao("Dependências do Python selecionado")
        self.dep = QLabel("Clique em verificar.")
        self.dep.setTextFormat(Qt.RichText)
        self.dep.setWordWrap(True)
        bv = QPushButton("Verificar dependências")
        bv.clicked.connect(self.verificar)
        fl3.addWidget(self.dep)
        fl3.addWidget(bv, 0, Qt.AlignLeft)
        lay.addWidget(f3)
        lay.addStretch()
        sc = rolavel(w)
        out = QVBoxLayout(self)
        out.setContentsMargins(0, 0, 0, 0)
        out.addWidget(sc)
        for wd in (self.py.ed, self.cwd.ed, self.mi.ed, self.rv.ed):
            wd.textChanged.connect(self.guardar)
        for wd in (self.omp, self.tent):
            wd.valueChanged.connect(self.guardar)
        for wd in (self.prio, self.pe):
            wd.toggled.connect(self.guardar)
        self.ee.textChanged.connect(self.guardar)

    def guardar(self):
        e = self.app.estado.env
        e.update(python=self.py.ed.text().strip(), cwd=self.cwd.ed.text().strip(), omp_threads=self.omp.value(),
                 tentativas_crash=self.tent.value(), baixa_prioridade=self.prio.isChecked(), parar_no_erro=self.pe.isChecked(),
                 meshinspector=self.mi.ed.text().strip(), revo=self.rv.ed.text().strip(), extra_env=self.ee.toPlainText())
        self.app.salvar_depois()

    def verificar(self):
        py = self.app.estado.env["python"]
        res = []
        for m in DEPENDENCIAS:
            try:
                r = subprocess.run([py, "-c", f"import {m}; print(getattr({m}, '__version__', 'ok'))"], capture_output=True, text=True, timeout=60)
                ok = r.returncode == 0
                res.append((m, ok, r.stdout.strip() if ok else "ausente"))
            except Exception as ex:
                res.append((m, False, type(ex).__name__))
        pal = self.app.pal
        self.dep.setText("<br>".join(f"<span style='color:{pal['ok'] if ok else pal['erro']}'>{'●' if ok else '○'}</span> {m} "
                                     f"<span style='color:{pal['suave']}'>{v}</span>" for m, ok, v in res))
