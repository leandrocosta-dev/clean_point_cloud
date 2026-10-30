"""GUI do pipeline de nuvem de pontos (Revo Scan).   python gui/app.py"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QByteArray, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

import catalog
import paginas as PG
import theme
from core import Estado
from assistente import PaginaAssistente
from inicio import PaginaInicio
from runner import Runner, _dur

# indices das paginas
INICIO, ASSISTENTE, PROJETO, PIPELINE, FERRAMENTAS, FILA, ENTREGAVEIS, JSON, AMBIENTE = range(9)


class Janela(QMainWindow):
    def __init__(self):
        super().__init__()
        self.estado = Estado()
        self.pal = theme.ESCURO if self.estado.d.get("tema", "escuro") == "escuro" else theme.CLARO
        self.itens = catalog.carregar_cache()
        self.runner = Runner(self.estado)
        self.setWindowTitle("Clean Point Cloud · Pipeline")
        self.resize(1480, 920)
        self.setAcceptDrops(True)
        self._timer_salvar = QTimer(self)
        self._timer_salvar.setSingleShot(True)
        self._timer_salvar.timeout.connect(self.estado.salvar)

        raiz = QWidget()
        raiz.setObjectName("raiz")
        self.setCentralWidget(raiz)
        h = QHBoxLayout(raiz)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        lateral = QFrame()
        lateral.setObjectName("lateral")
        lateral.setFixedWidth(230)
        lv = QVBoxLayout(lateral)
        lv.setContentsMargins(14, 18, 14, 14)
        lv.setSpacing(2)
        marca = QLabel("Clean Point Cloud")
        marca.setObjectName("marca")
        sub = QLabel("Revo Scan · pipeline de scans")
        sub.setObjectName("marcasub")
        lv.addWidget(marca)
        lv.addWidget(sub)
        lv.addSpacing(10)

        self.pilha = QStackedWidget()
        self.navs = []
        self.p_projeto = PG.PaginaProjeto(self)
        self.p_inicio = PaginaInicio(self)
        self.p_assist = PaginaAssistente(self)
        self.p_pipeline = PG.pagina_pipeline(self)
        self.p_scripts = PG.PaginaScripts(self)
        self.p_fila = PG.PaginaFila(self)
        self.p_entrega = PG.PaginaEntregaveis(self)
        self.p_json = PG.PaginaJson(self)
        self.p_amb = PG.PaginaAmbiente(self)
        # (grupo, rotulo, widget) na ordem dos indices acima; o grupo vira um titulo pequeno na barra lateral
        paginas = [("COMEÇAR", "Início", self.p_inicio), ("", "Assistente (Claude)", self.p_assist),
                   ("TRABALHAR", "Projeto e pastas", self.p_projeto), ("", "Pipeline completo", self.p_pipeline),
                   ("", "Ferramentas (todas)", self.p_scripts), ("", "Fila e execução", self.p_fila), ("", "Entregáveis", self.p_entrega),
                   ("CONFIGURAR", "Configurações JSON", self.p_json), ("", "Ambiente", self.p_amb)]
        for i, (grupo, nome, w) in enumerate(paginas):
            if grupo:
                g = QLabel(grupo)
                g.setObjectName("grupo")
                lv.addWidget(g)
            b = QPushButton(nome)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, i=i: self.ir(i))
            lv.addWidget(b)
            self.navs.append(b)
            self.pilha.addWidget(w)
        lv.addStretch()
        self.lbl_fila = QLabel("")
        self.lbl_fila.setObjectName("marcasub")
        lv.addWidget(self.lbl_fila)
        h.addWidget(lateral)
        h.addWidget(self.pilha, 1)

        sb = self.statusBar()
        self.st_msg = QLabel("Pronto")
        self.st_tempo = QLabel("")
        self.st_tempo.setObjectName("suave")
        self.barra = QProgressBar()
        self.barra.setFixedWidth(160)
        self.barra.setRange(0, 0)
        self.barra.setVisible(False)
        self.st_stop = QPushButton("Parar")
        self.st_stop.setObjectName("perigo")
        self.st_stop.setVisible(False)
        self.st_stop.clicked.connect(self.runner.parar)
        sb.addWidget(self.st_msg, 1)
        sb.addPermanentWidget(self.st_tempo)
        sb.addPermanentWidget(self.barra)
        sb.addPermanentWidget(self.st_stop)

        self.runner.linha.connect(self.p_fila.console.escrever)
        self.runner.linha.connect(self._status_linha)
        self.runner.job_iniciado.connect(self._job_iniciou)
        self.runner.job_concluido.connect(self._job_concluiu)
        self.runner.fila_concluida.connect(self._fim)
        self.runner.estado.connect(self._estado_run)
        self.t_run = QTimer(self)
        self.t_run.timeout.connect(self._tick)
        self.t_ini = 0.0
        self._na_fila = False
        self.estado_run = False

        QShortcut(QKeySequence("Ctrl+Return"), self, activated=self._atalho_run)
        geo = self.estado.d.get("geometria")
        if geo:
            self.restoreGeometry(QByteArray.fromBase64(geo.encode()))
        self.ir(INICIO)
        self.p_inicio.atualizar()
        self.atualiza_contagem_fila(len(self.estado.d["fila"]))

    # ------------------------------------------------------------------ navegacao
    def ir(self, i):
        self.pilha.setCurrentIndex(i)
        for k, b in enumerate(self.navs):
            b.setChecked(k == i)
        if i == ENTREGAVEIS:
            self.p_entrega.listar()

    def abrir_script(self, id_):
        self.ir(FERRAMENTAS)
        self.p_scripts.abrir(id_)

    def _atalho_run(self):
        if self.pilha.currentIndex() == INICIO:
            if self.p_inicio.f3.isVisible() and self.p_inicio.b_ini.isEnabled():
                self.p_inicio.iniciar()
            return
        for p in self.pilha.currentWidget().findChildren(PG.PaginaScript):
            if p.isVisible():
                p.executar()
                return

    # arrastar a pasta do projeto para a janela
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls() and any(os.path.isdir(u.toLocalFile()) for u in e.mimeData().urls()):
            e.acceptProposedAction()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if os.path.isdir(p):
                self.ir(INICIO)
                self.p_inicio.usar_pasta(p)
                self.aviso(f"Projeto: {os.path.basename(p)}")
                return

    # ------------------------------------------------------------------ servicos usados pelas paginas
    def scans_projeto(self):
        return self.p_projeto.scans

    def contexto_mudou(self):
        if hasattr(self, "p_inicio"):
            self.p_inicio.atualizar()
        proj = self.estado.ctx.get("projeto")
        self.setWindowTitle("Clean Point Cloud · " + (os.path.basename(proj) if proj else "Pipeline"))

    def salvar_depois(self):
        self._timer_salvar.start(700)

    def aviso(self, txt):
        self.st_msg.setText(txt)
        QTimer.singleShot(5000, lambda: self.st_msg.setText("Executando…" if self.estado_run else "Pronto"))

    def copiar(self, texto):
        QApplication.clipboard().setText(texto)
        self.aviso("Copiado")

    def atualiza_contagem_fila(self, n):
        if hasattr(self, "lbl_fila"):
            self.lbl_fila.setText(f"{n} etapa(s) na fila" if n else "")

    def adicionar_fila(self, jobs):
        self.p_fila.adicionar(jobs)
        self.aviso(f"{len(jobs)} etapa(s) adicionada(s) à fila")

    def executar(self, jobs, na_pagina_fila=False, ficar=False):
        """ficar=True: nao troca de pagina (o chamador mostra o andamento). Devolve False se ja ha execucao."""
        self._na_fila = na_pagina_fila
        if not na_pagina_fila and not ficar:
            self.ir(FILA)
        if self.runner.executar(jobs, parar_no_erro=bool(self.estado.env.get("parar_no_erro", True))):
            self.p_fila.console.escrever(f"══ {time.strftime('%H:%M:%S')} · {len(jobs)} etapa(s)", "info")
            return True
        QMessageBox.information(self, "Em andamento", "Já existe uma execução em andamento. Espere terminar ou use Parar.")
        return False

    def mudar_tema(self, nome):
        self.estado.d["tema"] = nome
        self.pal = theme.ESCURO if nome == "escuro" else theme.CLARO
        self.p_fila.console.pal = self.pal
        QApplication.instance().setStyleSheet(theme.qss(self.pal))
        self.estado.salvar()

    # ------------------------------------------------------------------ execucao
    def _estado_run(self, on):
        self.estado_run = on
        self.barra.setVisible(on)
        self.st_stop.setVisible(on)
        self.p_fila.b_run.setEnabled(not on)
        self.p_fila.b_stop.setEnabled(on)
        if on:
            self.t_ini = time.time()
            self.t_run.start(1000)
        else:
            self.t_run.stop()
            self.st_tempo.setText("")

    def _tick(self):
        self.st_tempo.setText(f"decorrido {_dur(time.time() - self.t_ini)}")

    def _status_linha(self, texto, tipo):
        """Mostra a ultima linha util do log na barra de status enquanto roda."""
        t = texto.strip()
        if self.estado_run and t and not t.startswith(("$", "▶")):
            self.st_msg.setText(t if len(t) < 150 else t[:147] + "…")

    def _job_iniciou(self, i, n, nome):
        self.st_msg.setText(f"Executando [{i + 1}/{n}] {nome}")
        if self._na_fila:
            self.p_fila.marcar(i, "rodando")

    def _job_concluiu(self, i, codigo):
        if self._na_fila:
            self.p_fila.marcar(i, "ok" if codigo == 0 else "erro")

    def _fim(self, ok):
        self.st_msg.setText("Concluído com sucesso" if ok else "Terminou com erro ou foi interrompido")
        self.p_fila.carregar_hist()
        QApplication.instance().alert(self)

    def closeEvent(self, e):
        if self.runner.rodando:
            r = QMessageBox.question(self, "Execução em andamento", "Há uma execução em andamento. Parar e sair?")
            if r != QMessageBox.Yes:
                e.ignore()
                return
            self.runner.parar()
        self.estado.d["geometria"] = bytes(self.saveGeometry().toBase64()).decode()
        self.estado.salvar()
        e.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    j = Janela()
    app.setStyleSheet(theme.qss(j.pal))
    j.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
