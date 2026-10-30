"""Pagina Inicio: fluxo guiado em 4 passos (projeto -> objetivo -> ajustes -> resultado)."""
import glob
import os
import re
import subprocess

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFileDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QVBoxLayout, QWidget)

import catalog
from core import citar, humano, substituir
from widgets import CampoTexto, LinhaParam, cartao

# cada tarefa: script, argumentos fixos, onde vai a saida e os poucos campos que importam
TAREFAS = [
    dict(id="placa", icone="✂", titulo="Limpar e entregar o .revo",
         sub="Peça apoiada ou suspensa sobre uma placa-gabarito. Fica só a peça; nada é aberto no Revo.",
         script="limpar_por_placa.py", fixos=[], saida_flag="--saida", saida_padrao="{SAIDA}\\{NOME}_limpo",
         extras=[("--diagnostico", "{TRABALHO}\\{NOME}_diagnostico")], entrega="revo",
         campos=[("altura_min", "Altura mínima sobre a placa (mm)", "Tudo abaixo disso é placa, parafuso ou suporte baixo. "
                  "Aumente se sobrar a base; diminua se a limpeza cortar a peça.", 16.0),
                 ("componentes", "Quantas peças manter", "1 mantém só o maior objeto conectado acima da placa.", 1)]),
    dict(id="mesa", icone="🧹", titulo="Limpar a mesa e entregar o .revo",
         sub="Peça pousada na mesa. Remove a mesa ao redor da peça; nada é aberto no Revo.",
         script="rodar_tudo.py", fixos=["--so-limpar"], saida_flag="--saida-limpo", saida_padrao="{SAIDA}\\{NOME}_limpo",
         extras=[], entrega="revo",
         campos=[("periferia_mm", "Remover a mesa além de (mm)", "Distância da peça a partir da qual a mesa é apagada (receita validada: 5).", 5.0),
                 ("periferia_banda", "Espessura da faixa da mesa (mm)", "Receita validada: 12.", 12.0),
                 ("periferia_alargue", "Alargue da base (mm)", "Quanto relevo baixo ainda conta como peça. Receita validada: 10.", 10.0)]),
    dict(id="completo", icone="⚙", titulo="Pipeline completo no Revo",
         sub="Limpa, funde no Revo Scan e exporta nuvem e/ou malha. Usa o mouse: não mexa no PC enquanto roda.",
         script="rodar_tudo.py", fixos=[], saida_flag="--saida-ply", saida_padrao="{SAIDA}\\{NOME}_resultado",
         extras=[("--saida-limpo", "{TRABALHO}\\{NOME}_limpo")], entrega="pasta",
         campos=[("ponto", "Resolução da nuvem (mm)", "Point Distance da fusão. Menor = mais detalhe e mais pesado.", 0.1),
                 ("malha", "Tipo de malha", "alta = Poisson de alta qualidade; rapida = Fast Meshing.", "alta"),
                 ("grade", "Resolução da malha (mm)", "Grid Size da malha alta.", 0.1),
                 ("entregaveis", "O que entregar", "nuvem+malha, só a nuvem (mais rápido) ou só a malha.", "nuvem+malha"),
                 ("furos", "Fechar furos da malha", "Hole Filling do Revo. Pode fechar aberturas reais da peça.", None)]),
]


def _item(id_):
    return next(i for i in catalog.carregar_cache() if i["id"] == id_)


class PaginaInicio(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.tarefa = None
        self.paineis = {}
        self.linhas_run = []
        self.rodando_aqui = False
        self.saida_run = ""
        self.checks = {}

        raiz = QVBoxLayout(self)
        raiz.setContentsMargins(0, 0, 0, 0)
        sc = self.rolagem = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setFrameShape(QFrame.NoFrame)
        raiz.addWidget(sc)
        corpo = QWidget()
        sc.setWidget(corpo)
        lay = QVBoxLayout(corpo)
        lay.setContentsMargins(28, 22, 28, 22)
        lay.setSpacing(14)
        t = QLabel("Início")
        t.setObjectName("h1")
        s = QLabel("Escolha o projeto, o que você quer fazer e clique em Iniciar. As ferramentas completas ficam no menu à esquerda.")
        s.setObjectName("suave")
        s.setWordWrap(True)
        lay.addWidget(t)
        lay.addWidget(s)

        # ---- 1. projeto
        f1, l1 = cartao("1   Projeto", "A pasta do projeto é a que contém o arquivo .revo. Você também pode arrastá-la para esta janela.")
        ln = QHBoxLayout()
        ln.setSpacing(14)
        b = QPushButton("Escolher pasta do projeto…")
        b.setObjectName("primario")
        b.clicked.connect(self.escolher)
        self.lbl_proj = QLabel("Nenhum projeto selecionado")
        self.lbl_proj.setObjectName("suave")
        self.lbl_proj.setWordWrap(True)
        bt = QPushButton("Outros projetos…")
        bt.setObjectName("mini")
        bt.setToolTip("Lista de projetos de uma pasta, pasta de saída e de trabalho")
        bt.clicked.connect(lambda: app.ir(2))
        ln.addWidget(b)
        ln.addWidget(self.lbl_proj, 1)
        ln.addWidget(bt)
        l1.addLayout(ln)
        l1.addSpacing(8)
        self.faixa = QScrollArea()
        self.faixa.setFixedHeight(190)
        self.faixa.setWidgetResizable(True)
        self.faixa.setFrameShape(QFrame.NoFrame)
        self.faixa.setVisible(False)
        l1.addWidget(self.faixa)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setObjectName("ajuda")
        l1.addWidget(self.lbl_sel)
        lay.addWidget(f1)

        # ---- 2. objetivo
        f2, l2 = cartao("2   O que você quer fazer?")
        self.grupo = QButtonGroup(self)
        self.grupo.setExclusive(True)
        cards = QHBoxLayout()
        cards.setSpacing(12)
        self.botoes = {}
        for tf in TAREFAS:
            bt = QPushButton()
            bt.setObjectName("tarefa")
            bt.setCheckable(True)
            bt.setMinimumHeight(132)
            vb = QVBoxLayout(bt)
            vb.setContentsMargins(16, 12, 16, 12)
            tit = QLabel(f"{tf['icone']}   {tf['titulo']}")
            tit.setStyleSheet("font-size:14px;font-weight:650;background:transparent")
            des = QLabel(tf["sub"])
            des.setWordWrap(True)
            des.setObjectName("suave")
            des.setStyleSheet("background:transparent")
            for lb_ in (tit, des):
                lb_.setAttribute(Qt.WA_TransparentForMouseEvents)
            vb.addWidget(tit)
            vb.addWidget(des)
            vb.addStretch()
            bt.setCursor(Qt.PointingHandCursor)
            bt.clicked.connect(lambda _=False, tf=tf: self.escolher_tarefa(tf))
            self.grupo.addButton(bt)
            self.botoes[tf["id"]] = bt
            cards.addWidget(bt, 1)
        l2.addLayout(cards)
        lay.addWidget(f2)

        # ---- 3. ajustes
        self.f3, self.l3 = cartao("3   Ajustes")
        self.area = QVBoxLayout()
        self.l3.addLayout(self.area)
        self.previa = QLabel("")
        self.previa.setObjectName("ajuda")
        self.previa.setWordWrap(True)
        self.previa.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.l3.addWidget(self.previa)
        lb = QHBoxLayout()
        self.b_ini = QPushButton("▶   Iniciar")
        self.b_ini.setObjectName("primario")
        self.b_ini.setMinimumHeight(40)
        self.b_ini.setMinimumWidth(180)
        self.b_ini.clicked.connect(self.iniciar)
        b_fila = QPushButton("＋ Adicionar à fila")
        b_fila.clicked.connect(lambda: self.iniciar(so_fila=True))
        b_av = QPushButton("Abrir na tela avançada")
        b_av.setToolTip("Mesmo script com todos os parâmetros")
        b_av.clicked.connect(lambda: app.abrir_script(self.tarefa["script"]) if self.tarefa else None)
        lb.addWidget(self.b_ini)
        lb.addWidget(b_fila)
        lb.addWidget(b_av)
        lb.addStretch()
        self.l3.addSpacing(6)
        self.l3.addLayout(lb)
        self.f3.setVisible(False)
        lay.addWidget(self.f3)

        # ---- 4. andamento / resultado
        self.f4, l4 = cartao("4   Andamento e resultado")
        self.banner = QLabel("")
        self.banner.setWordWrap(True)
        self.banner.setStyleSheet("font-size:15px;font-weight:650")
        l4.addWidget(self.banner)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(150)
        self.log.setStyleSheet("font-family:'Cascadia Mono',Consolas,monospace;font-size:12px")
        l4.addWidget(self.log)
        self.resumo = QLabel("")
        self.resumo.setTextFormat(Qt.RichText)
        self.resumo.setWordWrap(True)
        l4.addWidget(self.resumo)
        self.imagem = QLabel("")
        self.imagem.setAlignment(Qt.AlignLeft)
        l4.addWidget(self.imagem)
        self.acoes = QHBoxLayout()
        self.b_pasta = QPushButton("Abrir pasta de saída")
        self.b_revo = QPushButton("Abrir o .revo no Revo Scan")
        self.b_revo.setObjectName("primario")
        self.b_ent = QPushButton("Ver em Entregáveis")
        self.b_copia = QPushButton("Copiar caminho")
        self.b_pasta.clicked.connect(self.abrir_pasta)
        self.b_revo.clicked.connect(self.abrir_revo)
        self.b_ent.clicked.connect(self.ver_entregaveis)
        self.b_copia.clicked.connect(lambda: self.app.copiar(self.saida_run))
        for b in (self.b_revo, self.b_pasta, self.b_ent, self.b_copia):
            self.acoes.addWidget(b)
        self.acoes.addStretch()
        l4.addLayout(self.acoes)
        self.f4.setVisible(False)
        lay.addWidget(self.f4)
        lay.addStretch()

        app.runner.linha.connect(self._linha)
        app.runner.fila_concluida.connect(self._fim)
        self.atualizar()

    # ------------------------------------------------------------------ projeto
    def escolher(self):
        ini = self.app.estado.ctx.get("projeto") or self.app.estado.ctx.get("raiz") or ""
        p = QFileDialog.getExistingDirectory(self, "Pasta do projeto (contém o .revo)", ini)
        if p:
            self.usar_pasta(p)

    def usar_pasta(self, p):
        self.app.p_projeto.raiz.ed.setText(os.path.normpath(p))
        self.app.p_projeto._t_raiz.stop()
        self.app.p_projeto.recarregar()
        self.atualizar()

    def atualizar(self):
        ctx = self.app.estado.ctx
        scans = self.app.scans_projeto()
        ok = bool(ctx.get("projeto")) and bool(scans)
        self.lbl_proj.setText(f"{os.path.basename(ctx['projeto'])}   ·   {len(scans)} scan(s)\n{ctx['projeto']}" if ok else
                              "Nenhum projeto selecionado. Escolha a pasta que contém o arquivo .revo.")
        self.faixa.setVisible(ok)
        self.checks = {}
        if ok:
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(12)
            marcados = set(ctx.get("scans", []))
            for s in scans:
                c = QFrame()
                c.setObjectName("cartao")
                v = QVBoxLayout(c)
                v.setContentsMargins(8, 8, 8, 8)
                img = QLabel()
                img.setFixedSize(170, 112)
                img.setAlignment(Qt.AlignCenter)
                pm = QPixmap(os.path.join(s["pasta"], "preview.png"))
                if not pm.isNull():
                    img.setPixmap(pm.scaled(170, 112, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                else:
                    img.setText("sem miniatura")
                v.addWidget(img)
                ck = QCheckBox(s["guid"])
                ck.setChecked(s["guid"] in marcados or not marcados)
                ck.setToolTip(f"{s['formato']}")
                ck.toggled.connect(self._sel)
                self.checks[s["guid"]] = ck
                v.addWidget(ck)
                h.addWidget(c)
            h.addStretch()
            self.faixa.setWidget(w)
            self._sel()
        for b in self.botoes.values():
            b.setEnabled(ok)
            b.setToolTip("" if ok else "Escolha um projeto primeiro")
        if not ok:
            self.f3.setVisible(False)
        elif self.tarefa:
            self._previa()

    def _sel(self):
        marcados = [g for g, c in self.checks.items() if c.isChecked()]
        self.app.p_projeto.marcar(marcados if len(marcados) < len(self.checks) else [])
        self.lbl_sel.setText(f"{len(marcados)} de {len(self.checks)} scan(s) serão processados" if self.checks else "")
        if self.tarefa:
            self._previa()

    # ------------------------------------------------------------------ ajustes
    def escolher_tarefa(self, tf):
        self.tarefa = tf
        self.f3.setVisible(True)
        while self.area.count():
            it = self.area.takeAt(0)
            if it.widget():
                it.widget().setParent(None)
        if tf["id"] not in self.paineis:
            self.paineis[tf["id"]] = self._painel(tf)
        w, _ = self.paineis[tf["id"]]
        self.area.addWidget(w)
        self._previa()

    def _painel(self, tf):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        item = _item(tf["script"])
        linhas = {}
        # pasta de saida: o campo mais importante
        cab = QLabel("Onde salvar o resultado")
        cab.setStyleSheet("font-weight:600")
        v.addWidget(cab)
        ajuda = QLabel("Uma pasta nova (os scripts nunca sobrescrevem). {SAIDA} e {NOME} vêm da página Projeto e pastas.")
        ajuda.setObjectName("ajuda")
        v.addWidget(ajuda)
        saida = CampoTexto(app=self.app, caminho=False)
        saida.ed.setText(tf["saida_padrao"])
        b = QPushButton("Escolher…")
        b.setObjectName("mini")
        b.clicked.connect(lambda: self._pasta_saida(saida))
        saida.layout().addWidget(b)
        saida.mudou.connect(self._previa)
        v.addWidget(saida)
        res = QLabel("")
        res.setObjectName("ajuda")
        res.setWordWrap(True)
        v.addWidget(res)
        v.addSpacing(10)
        for dest, rot, aj, inicial in tf["campos"]:
            spec = next(p for p in item["params"] if p["dest"] == dest)
            lp = LinhaParam(spec, self.app, rotulo_txt=rot, simples=True, ajuda_txt=aj)
            if inicial is not None:
                lp._set(inicial)
            elif spec["tipo"] == "flag":
                lp._set(False)
            lp.mudou.connect(self._previa)
            linhas[dest] = lp
            v.addWidget(lp)
        return w, {"saida": saida, "linhas": linhas, "resolvido": res}

    def _pasta_saida(self, campo):
        p = QFileDialog.getExistingDirectory(self, "Pasta onde criar o resultado", substituir(self.app.estado.ctx.get("saida", ""), self.app.estado.ctx))
        if p:
            campo.ed.setText(os.path.join(os.path.normpath(p), "{NOME}_limpo" if self.tarefa["entrega"] == "revo" else "{NOME}_resultado"))

    def comando(self):
        tf, ctx = self.tarefa, self.app.estado.ctx
        _, d = self.paineis[tf["id"]]
        if not ctx.get("projeto"):
            raise ValueError("Escolha o projeto")
        saida = substituir(d["saida"].ed.text().strip().strip('"'), ctx)
        if not saida:
            raise ValueError("Escolha a pasta de saída")
        item = _item(tf["script"])
        a = [self.app.estado.env["python"], "-u", item["caminho"], ctx["projeto"], *tf["fixos"], tf["saida_flag"], saida]
        for flag, val in tf["extras"]:
            a += [flag, substituir(val, ctx)]
        for lp in d["linhas"].values():
            a += lp.tokens(ctx)
        marcados = [g for g, c in self.checks.items() if c.isChecked()]
        if not marcados:
            raise ValueError("Marque pelo menos um scan")
        if len(marcados) < len(self.checks):
            a += ["--scans", *marcados]
        return a, saida

    def _previa(self):
        try:
            a, saida = self.comando()
            self.previa.setText("Comando: " + citar(a[2:]))
            self.b_ini.setEnabled(True)
            ja = os.path.exists(saida) and (not os.path.isdir(saida) or bool(os.listdir(saida)))
            self.paineis[self.tarefa["id"]][1]["resolvido"].setText(
                f"Será salvo em: {saida}" + ("   ⚠ já existe: ao iniciar eu sugiro um nome novo" if ja else ""))
        except ValueError as e:
            self.previa.setText("⚠ " + str(e))
            self.b_ini.setEnabled(False)

    # ------------------------------------------------------------------ execucao
    def iniciar(self, so_fila=False):
        try:
            a, saida = self.comando()
        except ValueError as e:
            QMessageBox.warning(self, "Falta algo", str(e))
            return
        if os.path.exists(saida) and (not os.path.isdir(saida) or os.listdir(saida)):
            novo, n = saida, 2
            while os.path.exists(novo):
                novo, n = f"{saida}_{n}", n + 1
            r = QMessageBox.question(self, "A pasta já existe", f"{saida}\njá existe e tem conteúdo (os scripts não sobrescrevem).\n\n"
                                     f"Usar uma pasta nova?\n{novo}")
            if r != QMessageBox.Yes:
                return
            d = self.paineis[self.tarefa["id"]][1]["saida"].ed
            d.setText(novo)
            a, saida = self.comando()
        job = {"nome": self.tarefa["titulo"], "argv": a}
        if so_fila:
            self.app.adicionar_fila([job])
            return
        self.linhas_run, self.rodando_aqui, self.saida_run = [], True, saida
        self.log.clear()
        self.resumo.clear()
        self.imagem.clear()
        self.banner.setText("⏳  Executando… você pode acompanhar abaixo")
        self.banner.setStyleSheet(f"font-size:15px;font-weight:650;color:{self.app.pal['acento']}")
        for b in (self.b_revo, self.b_pasta, self.b_ent, self.b_copia):
            b.setVisible(False)
        self.f4.setVisible(True)
        QTimer.singleShot(50, lambda: self.rolagem.ensureWidgetVisible(self.f4, 0, 40))
        self.b_ini.setEnabled(False)
        if not self.app.executar([job], ficar=True):
            self.rodando_aqui = False

    def _linha(self, texto, tipo):
        if not self.rodando_aqui:
            return
        self.linhas_run.append((texto, tipo))
        self.log.appendPlainText(texto)

    def _fim(self, ok):
        if not self.rodando_aqui:
            return
        self.rodando_aqui = False
        self.b_ini.setEnabled(True)
        pal = self.app.pal
        if ok:
            self.banner.setText("✔  Concluído")
            self.banner.setStyleSheet(f"font-size:15px;font-weight:650;color:{pal['ok']}")
        else:
            erros = [t for t, k in self.linhas_run if k == "erro"][-4:]
            self.banner.setText("✖  A etapa falhou ou foi interrompida" + ("\n" + "\n".join(erros) if erros else ""))
            self.banner.setStyleSheet(f"font-size:14px;font-weight:650;color:{pal['erro']}")
        # tabela por scan lida do proprio log
        linhas, atual = [], None
        for t, _ in self.linhas_run:
            m = re.match(r"== (\S+)\s*$", t)
            if m:
                atual = m.group(1)
            m = re.search(r"pontos dos quadros: ([\d,]+) -> ([\d,]+) \((\d+)% mantidos", t)
            if m and atual:
                linhas.append((atual, m.group(1), m.group(2), m.group(3)))
            elif atual and "PULADO" in t:
                linhas.append((atual, "—", "—", "pulado (calibração ruim): copiado sem alteração"))
        if linhas:
            rows = "".join(f"<tr><td>{a}</td><td align=right>&nbsp;&nbsp;{b}</td><td align=right>&nbsp;&nbsp;{c}</td><td>&nbsp;&nbsp;{d if not d.isdigit() else d + '% mantidos'}</td></tr>"
                           for a, b, c, d in linhas)
            self.resumo.setText(f"<table><tr><th align=left>scan</th><th>&nbsp;&nbsp;pontos antes</th><th>&nbsp;&nbsp;depois</th><th></th></tr>{rows}</table>")
        # imagem antes/depois
        extras = [v for f, v in self.tarefa["extras"] if f == "--diagnostico"]
        pastas = [substituir(extras[0], self.app.estado.ctx)] if extras else []
        pastas.append(self.saida_run)
        for p in pastas:
            imgs = sorted(glob.glob(os.path.join(p, "*_antes_depois.png")))
            if imgs:
                pm = QPixmap(imgs[0])
                if not pm.isNull():
                    self.imagem.setPixmap(pm.scaledToWidth(min(760, pm.width()), Qt.SmoothTransformation))
                    self.imagem.setToolTip(imgs[0])
                break
        existe = os.path.isdir(self.saida_run)
        self.revo_saida = next(iter(glob.glob(os.path.join(self.saida_run, "*.revo"))), None)
        self.b_pasta.setVisible(existe)
        self.b_copia.setVisible(existe)
        self.b_ent.setVisible(existe)
        self.b_revo.setVisible(bool(self.revo_saida) and ok)
        QTimer.singleShot(50, lambda: self.rolagem.ensureWidgetVisible(self.f4, 0, 40))

    # ------------------------------------------------------------------ pos-execucao
    def abrir_pasta(self):
        if os.path.isdir(self.saida_run):
            os.startfile(self.saida_run)

    def ver_entregaveis(self):
        self.app.p_entrega.origem.ed.setText(self.saida_run)
        self.app.ir(6)

    def abrir_revo(self):
        import psutil
        if any(p.info["name"] == "RevoScan6.exe" for p in psutil.process_iter(["name"])):
            QMessageBox.information(self, "Revo Scan aberto", "O Revo Scan já está aberto. Feche-o (salvando o que precisar) e clique de novo: "
                                    "a automação não mexe em uma sessão aberta por você.")
            return
        if QMessageBox.question(self, "Abrir no Revo Scan", "Vou importar o projeto na biblioteca do Revo e abrir o primeiro scan.\n"
                                "Isso usa o mouse: não mexa no PC até terminar. Continuar?") != QMessageBox.Yes:
            return
        job = {"nome": "Abrir no Revo Scan", "argv": [self.app.estado.env["python"], "-u", os.path.join(catalog.PIPE, "abrir_no_revo.py"), self.revo_saida]}
        self.rodando_aqui = False
        self.app.executar([job])
