"""Estado persistente, variaveis de contexto, leitura de projetos .revo e cabecalho de .ply."""
import json
import os
import re
import shlex
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
ESTADO_DIR = os.path.join(AQUI, "estado")
os.makedirs(ESTADO_DIR, exist_ok=True)
ARQ_ESTADO = os.path.join(ESTADO_DIR, "estado.json")
ARQ_HISTORICO = os.path.join(ESTADO_DIR, "historico.json")
PRESETS_DIR = os.path.join(ESTADO_DIR, "presets")

PADRAO = {
    "ctx": {"raiz": "", "projeto": "", "saida": os.path.join(os.path.expanduser("~"), "Desktop", "Entregas"),
            "trabalho": os.path.join(os.path.expanduser("~"), "Desktop", "_trabalho"), "scans": []},
    "env": {
        "python": os.path.join(RAIZ, ".venv", "Scripts", "python.exe") if os.path.exists(os.path.join(RAIZ, ".venv", "Scripts", "python.exe")) else sys.executable,
        "baixa_prioridade": True, "omp_threads": 4, "tentativas_crash": 2, "parar_no_erro": True,
        "meshinspector": r"C:\Program Files\MeshInspector\MeshInspector\MeshInspector.exe",
        "revo": r"C:\Program Files\Revo Scan 6\RevoScan6.exe", "extra_env": "", "cwd": RAIZ,
    },
    "tema": "escuro",
    "fila": [],
    "form": {},
}


class Estado:
    def __init__(self):
        self.d = json.loads(json.dumps(PADRAO))
        try:
            with open(ARQ_ESTADO, encoding="utf-8") as f:
                salvo = json.load(f)
            for k, v in salvo.items():
                if isinstance(v, dict) and isinstance(self.d.get(k), dict):
                    self.d[k].update(v)
                else:
                    self.d[k] = v
        except (OSError, ValueError):
            pass

    def salvar(self):
        try:
            tmp = ARQ_ESTADO + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.d, f, indent=1, ensure_ascii=False)
            os.replace(tmp, ARQ_ESTADO)
        except OSError:
            pass

    @property
    def ctx(self):
        return self.d["ctx"]

    @property
    def env(self):
        return self.d["env"]


def variaveis(ctx):
    proj = ctx.get("projeto") or ""
    nome = os.path.basename(proj.rstrip("\\/")) if proj else ""
    return {"PROJETO": proj, "SAIDA": ctx.get("saida", ""), "TRABALHO": ctx.get("trabalho", ""), "NOME": nome,
            "SCANS": " ".join(ctx.get("scans", []))}


def substituir(texto, ctx):
    if not isinstance(texto, str):
        return texto
    v = variaveis(ctx)
    return re.sub(r"\{(PROJETO|SAIDA|TRABALHO|NOME|SCANS)\}", lambda m: v[m.group(1)], texto)


def dividir(texto):
    """Divide uma linha em argumentos respeitando aspas (caminhos com espaco)."""
    try:
        return shlex.split(texto.replace("\\", "\\\\"))     # barras de caminho do Windows nao sao escape
    except ValueError:
        return texto.split()


def citar(argv):
    """Linha de comando legivel/copiavel para o PowerShell."""
    def q(a):
        return f'"{a}"' if re.search(r"[\s&()]", a) or a == "" else a
    return " ".join(q(a) for a in argv)


# ---------------------------------------------------------------- projetos .revo
def _tem_revo(p):
    try:
        return any(f.endswith(".revo") for f in os.listdir(p))
    except OSError:
        return False


def listar_projetos(raiz, fundo=2):
    """A propria pasta (se tem .revo) ou as subpastas ate `fundo` niveis que tem."""
    raiz = os.path.normpath(raiz.strip().strip('"')) if raiz.strip() else ""
    if not raiz or not os.path.isdir(raiz):
        return []
    if _tem_revo(raiz):
        return [raiz]
    out = []
    try:
        for n in sorted(os.listdir(raiz)):
            p = os.path.join(raiz, n)
            if not os.path.isdir(p):
                continue
            if _tem_revo(p):
                out.append(p)
            elif fundo > 1:
                out += listar_projetos(p, fundo - 1)
    except OSError:
        pass
    return out


def ler_projeto(pasta):
    """-> (nome_do_revo, [scans]) ; scan = dict(guid, nome, formato, pasta)"""
    try:
        revo = next(f for f in os.listdir(pasta) if f.endswith(".revo"))
        with open(os.path.join(pasta, revo), encoding="utf-8") as f:
            nos = json.load(f).get("nodes", [])
    except (OSError, StopIteration, ValueError):
        return None, []
    scans = []
    for n in nos:
        if n.get("type") == 5:               # pasta de marcadores, nao e scan
            continue
        d = os.path.join(pasta, "data", n["guid"])
        cache = os.path.join(d, "cache")
        if os.path.exists(os.path.join(cache, "frames.dataset")):
            fmt = "frames.dataset"
        elif os.path.isdir(cache) and any(x.endswith(".dph") for x in os.listdir(cache)):
            fmt = ".dph (antigo)"
        elif os.path.isdir(d):
            fmt = "sem quadros"
        else:
            fmt = "ausente"
        scans.append({"guid": n["guid"], "nome": n.get("name", n["guid"]), "formato": fmt, "pasta": d,
                      "fuse": os.path.exists(os.path.join(d, "fuse.ply")), "malha": os.path.exists(os.path.join(d, "fuse_mesh.ply"))})
    return revo, scans


def tamanho_pasta(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return t


def humano(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024


def cabecalho_ply(caminho):
    """Contagem de vertices/faces lida do cabecalho (sem carregar o arquivo)."""
    info = {}
    try:
        with open(caminho, "rb") as f:
            for _ in range(60):
                linha = f.readline().decode("ascii", "replace").strip()
                if linha.startswith("element vertex"):
                    info["vertices"] = int(linha.split()[-1])
                elif linha.startswith("element face"):
                    info["faces"] = int(linha.split()[-1])
                elif linha == "end_header" or not linha:
                    break
    except OSError:
        pass
    return info
