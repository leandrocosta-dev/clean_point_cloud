"""Pagina "Assistente": conversa com o Claude Code do proprio usuario (assinatura), sem chave de API.

O Claude Code roda como subprocesso em modo headless (claude -p --output-format stream-json). Ele usa o login ja feito no CLI,
le o repositorio (CLAUDE.md, guia, scripts) e pode propor etapas, que so entram na fila depois que o usuario clicar.
"""
import html
import json
import os
import re
import shutil
import time

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import (QCheckBox, QFrame, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QTextBrowser,
                               QVBoxLayout, QWidget)

from catalog import PIPE, RAIZ
from core import substituir
from paginas import titulo_pagina
from widgets import Combo

MODELOS = [("Padrão da conta", ""), ("Opus", "opus"), ("Sonnet", "sonnet"), ("Haiku (rápido)", "haiku")]
MODOS = [("Analista (roda scripts e analisa a nuvem, como no CLI)", "analista"), ("Consultar (somente leitura)", "leitura"),
         ("Avançado (também edita o código do repositório)", "avancado")]
FERRAMENTAS = {"leitura": ["Read", "Grep", "Glob"], "analista": ["Read", "Grep", "Glob", "Write", "Bash"],
               "avancado": ["Read", "Grep", "Glob", "Edit", "Write", "Bash"]}
# freio contra acidentes, nao e uma sandbox: o que mais importa continua sendo a regra do prompt
BLOQUEADOS = ["Bash(rm:*)", "Bash(del:*)", "Bash(rmdir:*)", "Bash(rd:*)", "Bash(Remove-Item:*)", "Bash(move:*)", "Bash(mv:*)",
              "Bash(git push:*)", "Bash(git reset:*)", "Bash(curl:*)", "Bash(wget:*)"]

ANALISTA = """
MODO ANALISTA: voce pode rodar comandos (Bash) e ler imagens (Read em .png). Trabalhe como no CLI: entenda o pedido, olhe os dados de verdade, decida com evidencia e so entao processe. Para analisar uma nuvem use os scripts de pipeline/ e o python do .venv (.venv\\Scripts\\python.exe), leia os resultados (numeros, *_antes_depois.png, raw_preview.ply) e diga o que viu antes de concluir.
- Rascunhos e scripts de analise ficam em {TRABALHO}\\_analise (crie a pasta). Nunca escreva dentro do projeto original.
- Nunca apague, mova ou sobrescreva nada. Saidas vao para pastas novas; se o destino existir e nao estiver vazio, escolha outro nome.
- Rode analises curtas direto. Para o processamento longo (limpeza, uniao, Revo; mais de uns 8 minutos) NAO rode aqui: proponha o cartao "acao", assim a fila da GUI executa com prioridade baixa e nova tentativa em caso de falha. Depois que o usuario rodar, leia o log/saida e confira.
- Maquina instavel sob carga: use OMP_NUM_THREADS=4 e prioridade baixa em qualquer script pesado que rodar diretamente.
- Ao terminar, resuma o que mediu, o que decidiu e o que falta validar, sem afirmar o que nao verificou.
"""

SISTEMA = """\
Voce e o assistente embutido na GUI do pipeline de nuvem de pontos do Revo Scan 6 (repositorio atual). Responda sempre em portugues, \
direto e sem enfeite. Antes de afirmar algo sobre o projeto, leia o que for preciso: CLAUDE.md, diario/GUIA_PIPELINE.md, README.md e os \
scripts em pipeline/. Cite o arquivo quando relevante.

Regras do projeto, sempre validas:
- Os projetos originais (ex.: D:\\Projects) nunca sao alterados; toda saida vai para uma pasta nova.
- Os scripts nunca sobrescrevem um destino que ja existe e nao esta vazio.
- A limpeza e feita nos frames (profundidade zerada), nunca no .ply; poses nao mudam.
- Nao ha envio de dados para servicos externos.

Cada mensagem do usuario comeca com um bloco [ESTADO DA GUI] (projeto, pastas, scans, fila). Use-o, mas nao o repita.

Para PROPOR uma etapa, escreva um bloco de codigo com a linguagem "acao" contendo UM objeto JSON:
```acao
{"nome": "Limpar por placa", "script": "limpar_por_placa.py", "args": ["{PROJETO}", "--saida", "{SAIDA}\\\\{NOME}_LIMPO"]}
```
"script" e um arquivo de pipeline/. Em "args" pode usar {PROJETO} {SAIDA} {TRABALHO} {NOME}. Voce so propoe: o usuario revisa e clica \
em "Adicionar à fila". Proponha no maximo as etapas necessarias e explique em uma frase por que cada uma.
"""

RAPIDOS = [
    ("Explicar o último erro", "Leia o fim do log abaixo, diga o que deu errado e o que fazer. Se couber uma etapa, proponha.\n\n{LOG}"),
    ("Qual tarefa usar?", "Olhando o projeto selecionado e o guia, qual tarefa devo usar e com quais parâmetros? Explique em poucas linhas."),
    ("Revisar a fila", "Revise a fila atual: a ordem faz sentido, falta algum argumento, algum destino já existe?"),
]

_RE_ACAO = re.compile(r"```acao\s*\n(.*?)```", re.S)


def achar_claude():
    return shutil.which("claude") or next((p for p in (os.path.expanduser(r"~\.local\bin\claude.exe"),
                                                       os.path.expandvars(r"%APPDATA%\npm\claude.cmd")) if os.path.exists(p)), None)


def extrair_acoes(texto):
    """Separa o texto das propostas. Devolve (texto_sem_blocos, [dict validado ou {'erro': ...}])."""
    acoes = []
    for m in _RE_ACAO.finditer(texto):
        try:
            d = json.loads(m.group(1))
            script, args = d["script"], d.get("args", [])
            if not isinstance(script, str) or os.path.basename(script) != script or not os.path.isfile(os.path.join(PIPE, script)):
                raise ValueError(f"script inexistente em pipeline/: {script}")
            if not isinstance(args, list):
                raise ValueError("'args' precisa ser uma lista")
            acoes.append({"nome": str(d.get("nome") or script), "script": script, "args": [str(a) for a in args]})
        except Exception as e:                      # JSON invalido ou script que nao existe: mostra o motivo
            acoes.append({"erro": str(e)})
    return _RE_ACAO.sub("", texto).strip(), acoes


class PaginaAssistente(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.proc = None
        self.buf = b""
        self.errbuf = b""
        self.resposta = ""                           # texto da resposta em andamento
        self.acoes = []
        saved = app.estado.d.get("assistente", {})
        self.sid = saved.get("sid")
        self.msgs = saved.get("msgs", [])            # [(papel, texto)]
        self.t_render = QTimer(self)
        self.t_render.setSingleShot(True)
        self.t_render.timeout.connect(self.renderizar)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 22, 28, 12)
        lay.setSpacing(10)
        lay.addWidget(titulo_pagina("Assistente", "Conversa com o Claude Code da sua assinatura. Ele lê o guia e os scripts deste projeto, "
                                    "explica erros e sugere etapas. Nada roda sem o seu clique."))

        # barra de opcoes
        op = QHBoxLayout()
        op.setSpacing(8)
        self.lbl_cli = QLabel("")
        self.lbl_cli.setObjectName("suave")
        self.c_modelo = Combo()
        for t, v in MODELOS:
            self.c_modelo.addItem(t, v)
        self.c_modo = Combo()
        for t, v in MODOS:
            self.c_modo.addItem(t, v)
        self.c_modelo.setCurrentIndex(max(0, self.c_modelo.findData(saved.get("modelo", ""))))
        self.c_modo.setCurrentIndex(max(0, self.c_modo.findData(saved.get("modo", "analista"))))
        self._modo_ant = self.c_modo.currentData()
        self.c_modo.activated.connect(self._modo_mudou)
        self.c_modelo.activated.connect(lambda _=0: self._salvar())
        op.addWidget(QLabel("Modelo"))
        op.addWidget(self.c_modelo)
        op.addWidget(QLabel("Modo"))
        op.addWidget(self.c_modo, 1)
        b_novo = QPushButton("Nova conversa")
        b_novo.clicked.connect(self.nova)
        op.addWidget(b_novo)
        lay.addLayout(op)
        lay.addWidget(self.lbl_cli)

        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(True)
        self.chat.setFrameShape(QFrame.NoFrame)
        lay.addWidget(self.chat, 1)

        # propostas do assistente
        self.f_acoes = QFrame()
        self.f_acoes.setObjectName("cartao")
        self.l_acoes = QVBoxLayout(self.f_acoes)
        self.l_acoes.setContentsMargins(14, 10, 14, 10)
        self.f_acoes.setVisible(False)
        lay.addWidget(self.f_acoes)

        rap = QHBoxLayout()
        for rot, txt in RAPIDOS:
            b = QPushButton(rot)
            b.clicked.connect(lambda _=False, t=txt: self._rapido(t))
            rap.addWidget(b)
        rap.addStretch()
        self.lbl_estado = QLabel("")
        self.lbl_estado.setObjectName("suave")
        rap.addWidget(self.lbl_estado)
        lay.addLayout(rap)

        ent = QHBoxLayout()
        self.entrada = QPlainTextEdit()
        self.entrada.setPlaceholderText("Pergunte algo ou descreva o que quer fazer. Ctrl+Enter envia.")
        self.entrada.setFixedHeight(76)
        ent.addWidget(self.entrada, 1)
        col = QVBoxLayout()
        self.b_env = QPushButton("Enviar")
        self.b_env.setObjectName("primario")
        self.b_env.clicked.connect(self.enviar)
        self.b_parar = QPushButton("Parar")
        self.b_parar.clicked.connect(self.parar)
        self.b_parar.setEnabled(False)
        col.addWidget(self.b_env)
        col.addWidget(self.b_parar)
        ent.addLayout(col)
        lay.addLayout(ent)
        from PySide6.QtGui import QKeySequence, QShortcut
        QShortcut(QKeySequence("Ctrl+Return"), self.entrada, activated=self.enviar)

        self.exe = achar_claude()
        self._checar_cli()
        self.renderizar()

    # ------------------------------------------------------------------ estado
    def _checar_cli(self):
        if not self.exe:
            self.lbl_cli.setText("Claude Code não encontrado. Instale o CLI e faça login (comando `claude`) para usar o assistente.")
            self.b_env.setEnabled(False)
            return
        p = QProcess(self)
        p.finished.connect(lambda *_: self.lbl_cli.setText(
            f"Usando {bytes(p.readAllStandardOutput()).decode(errors='replace').strip()} com o login da sua assinatura (sem chave de API)."))
        p.start(self.exe, ["--version"])

    def _salvar(self):
        self.app.estado.d["assistente"] = {"sid": self.sid, "msgs": self.msgs[-60:], "modelo": self.c_modelo.currentData(),
                                           "modo": self.c_modo.currentData()}
        self.app.salvar_depois()

    def _modo_mudou(self, _):
        modo = self.c_modo.currentData()
        if modo != "leitura":
            msg = ("Neste modo o assistente executa comandos no seu computador sem pedir confirmação a cada passo, para analisar a nuvem "
                   "e gravar resultados em pastas novas. Apagar, mover e acessar a rede estão bloqueados, mas isso é um freio e não uma "
                   "proteção total; o projeto original continua protegido só pela regra dada a ele.")
            if modo == "avancado":
                msg += "\n\nAlém disso, ele pode editar o código deste repositório."
            r = QMessageBox.warning(self, "Modo " + ("analista" if modo == "analista" else "avançado"), msg + "\n\nAtivar?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if r != QMessageBox.Yes:
                self.c_modo.setCurrentIndex(self.c_modo.findData(self._modo_ant))
                return
        self._modo_ant = self.c_modo.currentData()
        self._salvar()

    def nova(self):
        if self.proc:
            self.parar()
        self.sid, self.msgs, self.resposta = None, [], ""
        self._limpar_acoes()
        self._salvar()
        self.renderizar()

    # ------------------------------------------------------------------ envio
    def _estado_gui(self):
        e = self.app.estado
        ctx = e.ctx
        return json.dumps({
            "projeto": ctx.get("projeto"), "saida": ctx.get("saida"), "trabalho": ctx.get("trabalho"), "scans_selecionados": ctx.get("scans", []),
            "fila": [j.get("nome") for j in e.d.get("fila", [])], "executando": bool(self.app.estado_run),
            "pagina_atual": self.app.navs[self.app.pilha.currentIndex()].text(),
        }, ensure_ascii=False)

    def _rapido(self, modelo):
        log = "\n".join(self.app.p_fila.console.toPlainText().splitlines()[-80:]) or "(o log está vazio)"
        self.entrada.setPlainText(modelo.replace("{LOG}", log))
        self.enviar()

    def enviar(self):
        texto = self.entrada.toPlainText().strip()
        if not texto or self.proc or not self.exe:
            return
        self.entrada.clear()
        self.msgs.append(("user", texto))
        self.resposta = ""
        self._limpar_acoes()
        self.renderizar()
        modo = self.c_modo.currentData()
        args = ["-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--append-system-prompt", SISTEMA + (ANALISTA.replace("{TRABALHO}", self.app.estado.ctx.get("trabalho") or "{TRABALHO}") if modo != "leitura" else ""),
                "--allowedTools", *FERRAMENTAS[modo]]
        if modo != "leitura":
            args += ["--disallowedTools", *BLOQUEADOS]
            for k in ("projeto", "saida", "trabalho"):          # deixa o Claude ler/gravar nessas pastas
                d = self.app.estado.ctx.get(k)
                if d and os.path.isdir(d):
                    args += ["--add-dir", d]
        if modo != "leitura":
            args += ["--permission-mode", "acceptEdits"]
        if self.c_modelo.currentData():
            args += ["--model", self.c_modelo.currentData()]
        if self.sid:
            args += ["--resume", self.sid]
        p = self.proc = QProcess(self)
        p.setWorkingDirectory(RAIZ)
        env = QProcessEnvironment.systemEnvironment()
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):       # garante o uso da assinatura, nao de uma chave
            env.remove(k)
        env.insert("BASH_DEFAULT_TIMEOUT_MS", "600000")                # analises de nuvem passam dos 2 min padrao
        env.insert("BASH_MAX_TIMEOUT_MS", "600000")
        env.insert("OMP_NUM_THREADS", "4")
        p.setProcessEnvironment(env)
        p.readyReadStandardOutput.connect(self._le_saida)
        p.readyReadStandardError.connect(lambda: setattr(self, "errbuf", self.errbuf + bytes(p.readAllStandardError())))
        p.finished.connect(self._terminou)
        p.errorOccurred.connect(lambda e: e == QProcess.FailedToStart and self._falha("Não consegui iniciar o Claude Code."))
        self.buf, self.errbuf, self.t0, self.ok_resultado = b"", b"", time.time(), False
        p.start(self.exe, args)
        p.write(f"[ESTADO DA GUI]\n{self._estado_gui()}\n[/ESTADO]\n\n{texto}".encode("utf-8"))
        p.closeWriteChannel()
        self.b_env.setEnabled(False)
        self.b_parar.setEnabled(True)
        self.lbl_estado.setText("Pensando…")

    def parar(self):
        if self.proc:
            self.proc.kill()

    # ------------------------------------------------------------------ saida do CLI
    def _le_saida(self):
        self.buf += bytes(self.proc.readAllStandardOutput())
        *linhas, self.buf = self.buf.split(b"\n")
        for ln in linhas:
            try:
                self._evento(json.loads(ln.decode("utf-8", errors="replace")))
            except ValueError:
                pass

    def _evento(self, d):
        tipo = d.get("type")
        if d.get("session_id") and tipo == "system":
            self.sid = d["session_id"]
        elif tipo == "stream_event":
            ev = d.get("event", {})
            if ev.get("type") == "content_block_delta" and ev["delta"].get("type") == "text_delta":
                self.resposta += ev["delta"]["text"]
                self.lbl_estado.setText("Escrevendo…")
                self.t_render.start(150)
            elif ev.get("type") == "content_block_start" and ev["content_block"].get("type") == "tool_use":
                nome = {"Read": "Lendo arquivo", "Grep": "Buscando no código", "Glob": "Listando arquivos", "Bash": "Executando comando", "Write": "Gravando arquivo"}.get(ev["content_block"]["name"], "Usando " + ev["content_block"]["name"])
                self.lbl_estado.setText(nome + "…")
            elif ev.get("type") == "message_start" and self.resposta:
                self.resposta += "\n\n"              # nova mensagem do mesmo turno (depois de usar uma ferramenta)
        elif tipo == "assistant":
            for b in d.get("message", {}).get("content", []):
                if b.get("type") == "tool_use":
                    i = b.get("input", {})
                    alvo = i.get("command") or i.get("file_path") or i.get("pattern") or ""
                    self.resposta += f"\n\n**▶ {b['name']}**\n```\n{str(alvo)[:900]}\n```\n"
        elif tipo == "user":
            cont = d.get("message", {}).get("content")
            for b in cont if isinstance(cont, list) else []:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    c = "".join(x.get("text", "") for x in c if isinstance(x, dict)) if isinstance(c, list) else str(c or "")
                    ln = c.strip().splitlines()
                    c = "\n".join(ln[:14] + (["… (+%d linhas)" % (len(ln) - 14)] if len(ln) > 14 else []))
                    self.resposta += f"```\n{c[:1500]}\n```\n\n"
        elif tipo == "rate_limit_event":
            info = d.get("rate_limit_info", {})
            if info.get("status") not in (None, "allowed", "allowed_warning"):
                self.resposta += "\n\n_Limite de uso da assinatura atingido. Tente de novo mais tarde._"
        elif tipo == "result":
            self.ok_resultado = True
            if d.get("is_error"):
                msg = str(d.get("result") or "erro desconhecido")
                dica = "\n\nFaça login no terminal com o comando `claude` e tente de novo." if "login" in msg.lower() else ""
                self.resposta = (self.resposta + "\n\n" if self.resposta else "") + f"**Erro:** {msg}{dica}"
            elif not self.resposta and d.get("result"):
                self.resposta = str(d["result"])

    def _falha(self, msg):
        self.resposta = f"**Erro:** {msg}"
        self.ok_resultado = True

    def _terminou(self, codigo, _status):
        resto = self.buf.strip()
        if resto:
            try:
                self._evento(json.loads(resto.decode("utf-8", errors="replace")))
            except ValueError:
                pass
        if not self.ok_resultado and not self.resposta:
            err = self.errbuf.decode("utf-8", errors="replace").strip()
            self.resposta = "_Interrompido._" if codigo != 0 and not err else f"**Erro:** {err or 'o Claude Code terminou sem resposta.'}"
        self.proc = None
        texto, acoes = extrair_acoes(self.resposta)
        self.msgs.append(("assistant", texto or "(sem texto)"))
        self.resposta = ""
        self.acoes = acoes
        self._mostrar_acoes()
        self.b_env.setEnabled(bool(self.exe))
        self.b_parar.setEnabled(False)
        self.lbl_estado.setText(f"Respondido em {time.time() - self.t0:.0f}s")
        self._salvar()
        self.renderizar()

    # ------------------------------------------------------------------ propostas
    def _limpar_acoes(self):
        self.acoes = []
        while self.l_acoes.count():
            w = self.l_acoes.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.f_acoes.setVisible(False)

    def _mostrar_acoes(self):
        self._limpar_acoes_widgets()
        if not self.acoes:
            return
        cab = QLabel("Etapas sugeridas pelo assistente")
        cab.setObjectName("h2")
        self.l_acoes.addWidget(cab)
        for a in self.acoes:
            linha = QWidget()
            h = QHBoxLayout(linha)
            h.setContentsMargins(0, 0, 0, 0)
            if "erro" in a:
                t = QLabel("Proposta inválida ignorada: " + a["erro"])
                t.setObjectName("suave")
                h.addWidget(t, 1)
            else:
                t = QLabel(f"<b>{html.escape(a['nome'])}</b><br><span style='color:{self.app.pal['suave']}'>{html.escape(a['script'])} "
                           f"{html.escape(' '.join(a['args']))}</span>")
                t.setWordWrap(True)
                t.setTextInteractionFlags(Qt.TextSelectableByMouse)
                h.addWidget(t, 1)
                b = QPushButton("Adicionar à fila")
                b.clicked.connect(lambda _=False, a=a, b=b: self._adicionar(a, b))
                h.addWidget(b)
            self.l_acoes.addWidget(linha)
        self.f_acoes.setVisible(True)

    def _limpar_acoes_widgets(self):
        while self.l_acoes.count():
            w = self.l_acoes.takeAt(0).widget()
            if w:
                w.deleteLater()

    def _adicionar(self, a, botao):
        e = self.app.estado
        argv = [e.env["python"], "-u", os.path.join(PIPE, a["script"])] + [substituir(x, e.ctx) for x in a["args"]]
        self.app.adicionar_fila([{"nome": a["nome"], "argv": argv}])
        botao.setText("Na fila ✓")
        botao.setEnabled(False)

    # ------------------------------------------------------------------ desenho do chat
    @staticmethod
    def _md_html(txt):
        d = QTextDocument()
        d.setMarkdown(txt)
        h = d.toHtml()
        m = re.search(r"<body[^>]*>(.*)</body>", h, re.S)
        return m.group(1) if m else html.escape(txt)

    def renderizar(self):
        p = self.app.pal
        partes = []
        itens = list(self.msgs)
        if self.proc is not None or self.resposta:
            itens.append(("assistant", extrair_acoes(self.resposta)[0] if self.resposta else "…"))
        if not itens:
            partes.append(f"<p style='color:{p['suave']}'>Sem mensagens ainda. Pergunte, por exemplo: <i>qual tarefa uso para um scan com placa de "
                          f"marcadores?</i> ou clique em <i>Explicar o último erro</i> depois de uma execução que falhou.</p>")
        for papel, txt in itens:
            if papel == "user":
                partes.append(f"<table width='100%' style='margin:8px 0'><tr><td style='background:{p['sel']};padding:8px 12px;border-radius:8px'>"
                              f"<b style='color:{p['acento']}'>Você</b><br>{html.escape(txt).replace(chr(10), '<br>')}</td></tr></table>")
            else:
                partes.append(f"<div style='margin:8px 0'><b style='color:{p['ok']}'>Assistente</b>{self._md_html(txt)}</div>")
        bar = self.chat.verticalScrollBar()
        fim = bar.value() >= bar.maximum() - 20
        self.chat.setHtml(f"<body style='color:{p['texto']}'>" + "".join(partes) + "</body>")
        if fim or self.proc is not None:
            bar.setValue(bar.maximum())
