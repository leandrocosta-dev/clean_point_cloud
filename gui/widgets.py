"""Widgets reutilizaveis: cartoes, linha de parametro e formulario gerado da especificacao do script."""
import os

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton,
                               QSpinBox, QVBoxLayout, QWidget)

from catalog import ROTULOS
from core import dividir, substituir

NAO_CAMINHO = {"nome", "lados", "scan", "scans", "apenas", "unir", "pesos", "corte", "referencia", "lado", "modo"}
DE_SCANS = {"scans", "apenas", "unir", "scan", "referencia"}
BIG = 2_000_000_000


def rotulo(d, ctx_lbl=None):
    return ROTULOS.get(d, d.replace("_", " "))


def cartao(titulo=None, sub=None):
    f = QFrame()
    f.setObjectName("cartao")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(18, 14, 18, 14)
    lay.setSpacing(0)
    if titulo:
        h = QLabel(titulo)
        h.setObjectName("h2")
        lay.addWidget(h)
        if sub:
            s = QLabel(sub)
            s.setObjectName("ajuda")
            s.setWordWrap(True)
            lay.addWidget(s)
        lay.addSpacing(6)
    return f, lay


def cartao_recolhivel(titulo, n, aberto=False):
    """Cartao com cabecalho clicavel. Devolve (frame, layout_do_corpo, alternar(bool))."""
    f = QFrame()
    f.setObjectName("cartao")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(18, 10, 18, 10)
    lay.setSpacing(0)
    cab = QPushButton()
    cab.setObjectName("fantasma")
    cab.setCursor(Qt.PointingHandCursor)
    cab.setStyleSheet("text-align:left;font-weight:650;font-size:15px;padding:6px 0")
    corpo = QWidget()
    cl = QVBoxLayout(corpo)
    cl.setContentsMargins(0, 4, 0, 0)
    cl.setSpacing(0)
    lay.addWidget(cab)
    lay.addWidget(corpo)

    def alternar(abrir):
        corpo.setVisible(abrir)
        cab.setText(("▾  " if abrir else "▸  ") + f"{titulo}   ·   {n} parâmetro(s)" + ("" if abrir else "   (clique para abrir)"))

    cab.clicked.connect(lambda: alternar(not corpo.isVisible()))
    alternar(aberto)
    f.alternar = alternar
    return f, cl


def _roda_so_com_foco(base):
    """Rolar a pagina com o ponteiro sobre o campo nao pode mudar o valor: so com o campo em foco."""
    def wheelEvent(self, e):
        if self.hasFocus():
            base.wheelEvent(self, e)
        else:
            e.ignore()                              # a rolagem vai para a pagina
    return wheelEvent


class SpinInt(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    wheelEvent = _roda_so_com_foco(QSpinBox)


class SpinDbl(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    wheelEvent = _roda_so_com_foco(QDoubleSpinBox)


class Combo(QComboBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)

    wheelEvent = _roda_so_com_foco(QComboBox)


class NumDouble(SpinDbl):
    def __init__(self):
        super().__init__()
        self.setDecimals(6)
        self.setRange(-1e12, 1e12)
        self.setSingleStep(0.1)
        self.setStepType(QDoubleSpinBox.AdaptiveDecimalStepType)
        self.setKeyboardTracking(False)

    def textFromValue(self, v):
        t = f"{v:.6f}".rstrip("0").rstrip(".")
        return t if t not in ("", "-0") else "0"

    def valueFromText(self, t):
        try:
            return float(t.replace(",", "."))
        except ValueError:
            return 0.0

    def validate(self, text, pos):
        from PySide6.QtGui import QValidator
        try:
            float(text.replace(",", ".") or "0")
            return QValidator.Acceptable, text, pos
        except ValueError:
            return (QValidator.Intermediate if text in ("", "-", ".", ",", "-.") else QValidator.Invalid), text, pos


class ScanPicker(QDialog):
    def __init__(self, scans, marcados, parent=None, max_n=None):
        super().__init__(parent)
        self.setWindowTitle("Escolher scans")
        self.resize(480, 520)
        lay = QVBoxLayout(self)
        self.max_n = max_n
        self.lista = QListWidget()
        for s in scans:
            it = QListWidgetItem(f"{s['guid']}   ·   {s['formato']}")
            it.setData(Qt.UserRole, s["guid"])
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if s["guid"] in marcados else Qt.Unchecked)
            self.lista.addItem(it)
        lay.addWidget(self.lista)
        linha = QHBoxLayout()
        for txt, fn in (("Todos", lambda: self._todos(Qt.Checked)), ("Nenhum", lambda: self._todos(Qt.Unchecked))):
            b = QPushButton(txt)
            b.setObjectName("mini")
            b.clicked.connect(fn)
            linha.addWidget(b)
        linha.addStretch()
        lay.addLayout(linha)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _todos(self, estado):
        for i in range(self.lista.count()):
            self.lista.item(i).setCheckState(estado)

    def escolhidos(self):
        return [self.lista.item(i).data(Qt.UserRole) for i in range(self.lista.count()) if self.lista.item(i).checkState() == Qt.Checked]


class CampoTexto(QWidget):
    """QLineEdit com botoes opcionais de pasta/arquivo/scans."""
    mudou = Signal()

    def __init__(self, caminho=True, scans=False, app=None, placeholder="", multi=False):
        super().__init__()
        self.app = app
        self.multi = multi
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.ed = QLineEdit()
        self.ed.setPlaceholderText(placeholder)
        self.ed.textChanged.connect(lambda _: self.mudou.emit())
        lay.addWidget(self.ed, 1)
        if caminho:
            for txt, fn, tip in (("Pasta", self._pasta, "Escolher pasta"), ("Arquivo", self._arquivo, "Escolher arquivo")):
                b = QPushButton(txt)
                b.setObjectName("mini")
                b.setToolTip(tip)
                b.clicked.connect(fn)
                lay.addWidget(b)
        if scans:
            b = QPushButton("Scans…")
            b.setObjectName("mini")
            b.setToolTip("Escolher entre os scans do projeto selecionado")
            b.clicked.connect(self._scans)
            lay.addWidget(b)

    def _resolvido(self):
        t = substituir(self.ed.text().strip().strip('"'), self.app.estado.ctx) if self.app else self.ed.text()
        return t if os.path.exists(t) else (os.path.dirname(t) if os.path.exists(os.path.dirname(t) or ".") else "")

    def _pasta(self):
        p = QFileDialog.getExistingDirectory(self, "Escolher pasta", self._resolvido())
        if p:
            if self.multi and self.ed.text().strip():
                self.ed.setText(self.ed.text().strip() + f' "{os.path.normpath(p)}"')
            else:
                self.ed.setText(os.path.normpath(p))

    def _arquivo(self):
        if self.multi:
            ps, _ = QFileDialog.getOpenFileNames(self, "Escolher arquivos", self._resolvido())
            if ps:
                antes = self.ed.text().strip()
                self.ed.setText((antes + " " if antes else "") + " ".join(f'"{os.path.normpath(p)}"' for p in ps))
            return
        p, _ = QFileDialog.getOpenFileName(self, "Escolher arquivo ou digite um novo caminho", self._resolvido())
        if p:
            self.ed.setText(os.path.normpath(p))

    def _scans(self):
        scans = self.app.scans_projeto() if self.app else []
        if not scans:
            self.ed.setFocus()
            return
        d = ScanPicker(scans, set(self.ed.text().split()), self)
        if d.exec():
            self.ed.setText(" ".join(d.escolhidos()))


class LinhaParam(QFrame):
    """Uma linha: [usar] flag/ajuda ........ editor."""
    mudou = Signal()

    def __init__(self, spec, app, rotulo_txt=None, simples=False, ajuda_txt=None):
        super().__init__()
        self.spec, self.app = spec, app
        self.setObjectName("linha")
        s = spec
        self.tipo = s["tipo"]
        lista = s["nargs"] not in (None, "?") or s["tipo"] == "append"
        self.lista = lista
        self.opcional = not s["obrigatorio"]
        grade = QGridLayout(self)
        grade.setContentsMargins(0, 10, 0, 10)
        grade.setHorizontalSpacing(12)
        grade.setVerticalSpacing(2)
        grade.setColumnStretch(1, 1)

        self.usar = QCheckBox()
        self.usar.setToolTip("Marcado: o parâmetro é enviado ao script. Desmarcado: vale o padrão do script.")
        precisa_usar = self.opcional and not s["posicional"] and s["tipo"] != "flag" and (s["tipo"] in ("int", "float") or s["opcoes"])
        self.tem_usar = bool(precisa_usar) and not simples
        self.usar.setVisible(self.tem_usar)
        grade.addWidget(self.usar, 0, 0, 2, 1, Qt.AlignTop)

        nome = QLabel(rotulo_txt or (rotulo(s["dest"]) if s["dest"] in ROTULOS else s["dest"].replace("_", " ")))
        nome.setStyleSheet("font-weight:600")
        topo = QHBoxLayout()
        topo.setSpacing(8)
        topo.addWidget(nome)
        if s["obrigatorio"]:
            r = QLabel("obrigatório")
            r.setObjectName("req")
            r.setStyleSheet("font-size:11px")
            topo.addWidget(r)
        flag = QLabel(s["flag"] or "posicional")
        flag.setObjectName("flag")
        flag.setStyleSheet("color:#8b97a6")
        flag.setVisible(not simples)
        topo.addWidget(flag)
        topo.addStretch()
        cab = QWidget()
        cab.setLayout(topo)
        topo.setContentsMargins(0, 0, 0, 0)
        grade.addWidget(cab, 0, 1)

        txt = ajuda_txt if ajuda_txt is not None else (s["ajuda"] or "")
        if s["padrao"] not in (None, False, "") and s["tipo"] != "flag" and ajuda_txt is None:
            txt = (txt + "  " if txt else "") + f"[padrão: {s['padrao']}]"
        if s["padrao_dinamico"]:
            txt = (txt + "  " if txt else "") + "[padrão calculado pelo script]"
        self.ajuda = QLabel(txt)
        self.ajuda.setObjectName("ajuda")
        self.ajuda.setWordWrap(True)
        self.ajuda.setVisible(bool(txt))
        grade.addWidget(self.ajuda, 1, 1)

        self.editor = self._criar_editor()
        grade.addWidget(self.editor, 0, 2, 2, 1, Qt.AlignVCenter)
        grade.setColumnMinimumWidth(2, 360)
        self.usar.toggled.connect(self._atualiza)
        self._atualiza()

    # -------------------------------------------------------------- editor
    def _criar_editor(self):
        s, t = self.spec, self.tipo
        self.padrao_valor = s["padrao"]
        if t == "flag":
            w = QCheckBox("ligado" if s["acao"] == "store_true" else "desliga o padrão")
            w.toggled.connect(lambda _: self.mudou.emit())
            self._get, self._set = w.isChecked, w.setChecked
            return w
        if s["opcoes"] and not self.lista:
            w = Combo()
            w.addItems([str(o) for o in s["opcoes"]])
            if s["padrao"] is not None and str(s["padrao"]) in [str(o) for o in s["opcoes"]]:
                w.setCurrentText(str(s["padrao"]))
            w.currentIndexChanged.connect(lambda _: self._edit())
            self._get, self._set = w.currentText, w.setCurrentText
            return w
        if t == "int" and not self.lista:
            w = SpinInt()
            w.setRange(-BIG, BIG)
            w.setKeyboardTracking(False)
            if isinstance(s["padrao"], (int, float)) and not isinstance(s["padrao"], bool):
                w.setValue(int(s["padrao"]))
            w.valueChanged.connect(lambda _: self._edit())
            self._get, self._set = w.value, lambda v: w.setValue(int(float(v)))
            return w
        if t == "float" and not self.lista:
            w = NumDouble()
            if isinstance(s["padrao"], (int, float)) and not isinstance(s["padrao"], bool):
                w.setValue(float(s["padrao"]))
            w.valueChanged.connect(lambda _: self._edit())
            self._get, self._set = w.value, lambda v: w.setValue(float(v))
            return w
        d = s["dest"]
        caminho = d not in NAO_CAMINHO
        ph = ""
        if s["padrao"] not in (None, False):
            ph = f"padrão: {s['padrao']}"
        elif s["nargs"] not in (None, "?"):
            ph = "valores separados por espaço"
        elif caminho:
            ph = "use {PROJETO}, {SAIDA}, {TRABALHO}, {NOME}"
        w = CampoTexto(caminho=caminho, scans=d in DE_SCANS, app=self.app, placeholder=ph, multi=self.lista)
        w.mudou.connect(self._edit)
        self._get, self._set = w.ed.text, w.ed.setText
        return w

    def _edit(self):
        if self.tem_usar and not self.usar.isChecked():
            self.usar.setChecked(True)
        self.mudou.emit()

    def _atualiza(self):
        if self.tem_usar:
            self.editor.setEnabled(self.usar.isChecked())
        self.mudou.emit()

    # -------------------------------------------------------------- valores
    def ativo(self):
        s = self.spec
        if self.tipo == "flag":
            return self._get()
        if s["obrigatorio"]:
            return True
        if self.tem_usar:
            return self.usar.isChecked()
        v = self._get()
        return bool(v.strip()) if isinstance(v, str) else True

    def tokens(self, ctx):
        """Argumentos (flag + valores) desta linha ou [] se nao ativo; ValueError se falta obrigatorio."""
        s = self.spec
        v = self._get()
        if s["obrigatorio"] and self.tipo != "flag" and isinstance(v, str) and not v.strip():
            raise ValueError(f"'{s['flag'] or s['dest']}' é obrigatório")
        if not self.ativo():
            return []
        if self.tipo == "flag":
            vals = []
        elif isinstance(v, str):
            v = substituir(v.strip(), ctx)
            if self.tipo == "append":
                return [x for t in dividir(v) for x in (s["flag"], t)]
            vals = dividir(v) if self.lista else [v.strip('"')]
        else:
            vals = [repr(v) if isinstance(v, float) else str(v)]
        return vals if s["posicional"] else [s["flag"], *vals]

    def valor(self):
        return {"ativo": self.usar.isChecked(), "valor": self._get()}

    def aplicar(self, d):
        self.bloquear(True)
        try:
            if "valor" in d and d["valor"] is not None:
                self._set(d["valor"])
            self.usar.setChecked(bool(d.get("ativo", False)))
        finally:
            self.bloquear(False)
        self._atualiza()

    def definir(self, valor):
        """Define e ativa (usado por presets)."""
        self.bloquear(True)
        try:
            self._set(valor)
            if self.tem_usar or self.tipo == "flag":
                if self.tipo != "flag":
                    self.usar.setChecked(True)
        finally:
            self.bloquear(False)
        self._atualiza()

    def restaurar(self):
        s = self.spec
        padrao = s["padrao"]
        self.bloquear(True)
        try:
            if self.tipo == "flag":
                self._set(False)
            elif self.tipo in ("int", "float") and not self.lista:
                self._set(padrao if isinstance(padrao, (int, float)) else 0)
            elif s["opcoes"] and not self.lista:
                self._set(str(padrao) if padrao is not None else str(s["opcoes"][0]))
            else:
                self._set("")
            self.usar.setChecked(False)
        finally:
            self.bloquear(False)
        self._atualiza()

    def bloquear(self, b):
        self.blockSignals(b)
        for w in (self.editor, self.usar):
            w.blockSignals(b)
        if isinstance(self.editor, CampoTexto):
            self.editor.ed.blockSignals(b)

    def preencher_contexto(self):
        """Sugere valores a partir do contexto quando o campo esta vazio."""
        s, d = self.spec, self.spec["dest"]
        if not isinstance(self.editor, CampoTexto) or self._get().strip():
            return
        sug = None
        if d in ("projeto", "project_dir"):
            sug = "{PROJETO}"
        elif d == "saida_limpo":
            sug = "{SAIDA}\\{NOME}_limpo"
        elif d in ("saida_ply", "saida", "out", "out_dir", "output") and (s["obrigatorio"] or d == "saida_ply"):
            sug = "{SAIDA}\\{NOME}_" + d.replace("saida_", "").replace("saida", "resultado")
        if sug:
            self._set(sug)
            self.mudou.emit()


class FormScript(QWidget):
    mudou = Signal()

    def __init__(self, item, app):
        super().__init__()
        self.item, self.app = item, app
        self.linhas = {}
        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        raiz.setSpacing(14)
        params = item["params"]
        if not params:
            f, lay = cartao("Sem parâmetros", "Este script não recebe argumentos (caminhos fixos no código). Use 'Executar' para rodá-lo como está.")
            raiz.addWidget(f)
        grupos = []
        if item.get("grupos"):
            usados = set()
            for titulo, dests in item["grupos"]:
                ls = [p for d in dests for p in params if p["dest"] == d]
                usados |= {p["dest"] for p in ls}
                grupos.append((titulo, ls))
            resto = [p for p in params if p["dest"] not in usados]
            if resto:
                grupos.append(("Outros parâmetros", resto))
        else:
            ent = [p for p in params if p["posicional"] or p["obrigatorio"]]
            opc = [p for p in params if p not in ent]
            if ent:
                grupos.append(("Entrada e saída", ent))
            if opc:
                grupos.append(("Parâmetros", opc))
        self.secoes = []
        tem_ent = any(t == "Entrada e saída" for t, _ in grupos)
        for titulo, ls in grupos:
            if not ls:
                continue
            recolher = titulo in ("Quadros e amostragem", "Outros parâmetros") or (titulo == "Parâmetros" and tem_ent and len(ls) > 4)
            if recolher:
                f, lay = cartao_recolhivel(titulo, len(ls))
                self.secoes.append((f, [p["dest"] for p in ls]))
            else:
                f, lay = cartao(titulo)
            for i, p in enumerate(ls):
                linha = LinhaParam(p, app)
                linha.mudou.connect(self.mudou)
                self.linhas[p["dest"]] = linha
                lay.addWidget(linha)
            raiz.addWidget(f)
        f, lay = cartao("Argumentos extras", "Linha crua anexada ao final do comando (aspas para caminhos com espaço). Garante controle total sobre qualquer flag.")
        self.extra = QPlainTextEdit()
        self.extra.setFixedHeight(64)
        self.extra.setPlaceholderText("--flag valor --outra-flag")
        self.extra.textChanged.connect(self.mudou)
        lay.addWidget(self.extra)
        raiz.addWidget(f)
        raiz.addStretch()
        self._abrir_se_ativo()

    def _abrir_se_ativo(self):
        """Secao recolhida abre sozinha quando algum parametro dela esta em uso (nada fica escondido com efeito)."""
        for f, dests in self.secoes:
            if any(self.linhas[d].ativo() and not self.linhas[d].spec["obrigatorio"] for d in dests if d in self.linhas):
                f.alternar(True)

    def argv(self):
        """Argumentos do script (sem python nem caminho). ValueError com todas as pendencias."""
        ctx = self.app.estado.ctx
        pos, opc, erros = [], [], []
        for p in self.item["params"]:
            try:
                tk = self.linhas[p["dest"]].tokens(ctx)
            except ValueError as e:
                erros.append(str(e))
                continue
            (pos if p["posicional"] else opc).extend(tk)
        if erros:
            raise ValueError("; ".join(erros))
        return pos + opc + dividir(substituir(self.extra.toPlainText().strip(), ctx))

    def valores(self):
        return {"linhas": {d: l.valor() for d, l in self.linhas.items()}, "extra": self.extra.toPlainText()}

    def aplicar(self, v):
        for d, val in v.get("linhas", {}).items():
            if d in self.linhas:
                self.linhas[d].aplicar(val)
        self.extra.setPlainText(v.get("extra", ""))
        self._abrir_se_ativo()
        self.mudou.emit()

    def aplicar_valores(self, mapa):
        """mapa dest -> valor (presets nomeados do codigo)."""
        for d, val in mapa.items():
            if d in self.linhas:
                self.linhas[d].definir(val)
        self._abrir_se_ativo()
        self.mudou.emit()

    def restaurar(self):
        for l in self.linhas.values():
            l.restaurar()
        self.extra.clear()
        self.mudou.emit()

    def preencher_contexto(self):
        for l in self.linhas.values():
            l.preencher_contexto()
