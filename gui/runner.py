"""Executor sequencial de comandos: fila, prioridade baixa, repeticao em crash do Open3D, historico."""
import json
import os
import subprocess
import time

from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal

from core import ARQ_HISTORICO, citar

CODIGOS_CRASH = {139, 3221225477, -1073741819, -11, 3221226505}   # segfault do Open3D (guia, secao 8)


def carregar_historico():
    try:
        with open(ARQ_HISTORICO, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def _gravar_historico(reg):
    h = carregar_historico()
    h.append(reg)
    try:
        with open(ARQ_HISTORICO, "w", encoding="utf-8") as f:
            json.dump(h[-500:], f, indent=1, ensure_ascii=False)
    except OSError:
        pass


class Runner(QObject):
    linha = Signal(str, str)                 # texto, tipo (out|info|erro|ok)
    job_iniciado = Signal(int, int, str)     # indice, total, nome
    job_concluido = Signal(int, int)         # indice, codigo
    fila_concluida = Signal(bool)
    estado = Signal(bool)                    # rodando?

    def __init__(self, estado):
        super().__init__()
        self.estado_app = estado
        self.proc = None
        self.jobs, self.i, self.rodando = [], -1, False
        self.t0 = 0.0
        self.tentativa = 0
        self.parar_pedido = False
        self.falhou = False
        self._buf = ""

    # ------------------------------------------------------------- API
    def executar(self, jobs, parar_no_erro=True):
        if self.rodando:
            self.linha.emit("Já existe uma execução em andamento.", "erro")
            return False
        self.jobs, self.i, self.parar_no_erro = list(jobs), -1, parar_no_erro
        self.parar_pedido = self.falhou = False
        self.rodando = True
        self.estado.emit(True)
        self._proximo()
        return True

    def parar(self):
        self.parar_pedido = True
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.linha.emit("Parando (encerrando a árvore de processos)…", "erro")
            pid = int(self.proc.processId())
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)

    # ------------------------------------------------------------- interno
    def _proximo(self):
        self.i += 1
        if self.parar_pedido or self.i >= len(self.jobs):
            self._fim()
            return
        self.tentativa = 0
        self._iniciar()

    def _iniciar(self):
        job = self.jobs[self.i]
        env = self.estado_app.env
        self.job_iniciado.emit(self.i, len(self.jobs), job["nome"])
        cmd = job["argv"]
        suf = f"  (tentativa {self.tentativa + 1})" if self.tentativa else ""
        self.linha.emit(f"\n▶ [{self.i + 1}/{len(self.jobs)}] {job['nome']}{suf}", "info")
        self.linha.emit("$ " + citar(cmd), "info")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        pe = QProcessEnvironment.systemEnvironment()
        pe.insert("PYTHONUNBUFFERED", "1")
        pe.insert("PYTHONIOENCODING", "utf-8")
        if env.get("omp_threads"):
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                pe.insert(k, str(env["omp_threads"]))
        for par in (env.get("extra_env") or "").replace(";", "\n").splitlines():
            if "=" in par:
                k, v = par.split("=", 1)
                pe.insert(k.strip(), v.strip())
        for k, v in job.get("env", {}).items():
            pe.insert(k, v)
        self.proc.setProcessEnvironment(pe)
        self.proc.setWorkingDirectory(job.get("cwd") or env.get("cwd") or os.getcwd())
        self.proc.readyReadStandardOutput.connect(self._ler)
        self.proc.started.connect(self._iniciou)
        self.proc.finished.connect(self._terminou)
        self.proc.errorOccurred.connect(self._erro)
        self._buf = ""
        self.t0 = time.time()
        self.proc.start(cmd[0], cmd[1:])

    def _iniciou(self):
        if self.estado_app.env.get("baixa_prioridade"):
            try:
                import psutil
                psutil.Process(int(self.proc.processId())).nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            except Exception:
                pass

    def _ler(self):
        dados = bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        self._buf += dados.replace("\r\n", "\n").replace("\r", "\n")
        *linhas, self._buf = self._buf.split("\n")
        for l in linhas:
            self.linha.emit(l, "erro" if _parece_erro(l) else "out")

    def _erro(self, e):
        if e == QProcess.FailedToStart:
            self.linha.emit(f"Falha ao iniciar: {self.jobs[self.i]['argv'][0]}", "erro")

    def _terminou(self, codigo, status):
        if self._buf.strip():
            self.linha.emit(self._buf, "out")
        self._buf = ""
        dt = time.time() - self.t0
        job = self.jobs[self.i]
        crash = (codigo in CODIGOS_CRASH or status == QProcess.CrashExit) and not self.parar_pedido
        if crash and self.tentativa < int(self.estado_app.env.get("tentativas_crash", 0)):
            self.tentativa += 1
            self.linha.emit(f"Crash do processo (código {codigo}); repetindo em 3 s…", "erro")
            QTimer.singleShot(3000, self._iniciar)
            return
        ok = codigo == 0 and not self.parar_pedido
        self.linha.emit(f"{'✔ concluído' if ok else '✖ terminou com código ' + str(codigo)} em {_dur(dt)}", "ok" if ok else "erro")
        _gravar_historico({"quando": time.strftime("%Y-%m-%d %H:%M:%S"), "nome": job["nome"], "argv": job["argv"],
                           "codigo": codigo, "segundos": round(dt, 1), "tentativas": self.tentativa + 1})
        self.job_concluido.emit(self.i, codigo)
        if not ok:
            self.falhou = True
            if self.parar_no_erro or self.parar_pedido:
                self._fim()
                return
        self._proximo()

    def _fim(self):
        self.rodando = False
        self.estado.emit(False)
        self.fila_concluida.emit(not self.falhou and not self.parar_pedido)


def _parece_erro(l):
    s = l.lower()
    return s.startswith("traceback") or "error" in s[:40] or "erro" in s[:20] or s.startswith("  file \"")


def _dur(s):
    s = int(s)
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else (f"{m}m {s:02d}s" if m else f"{s}s")
