"""Banco de regressao: roda casos congelados e compara as metricas com a base registrada.

    python pipeline/regressao.py --listar
    python pipeline/regressao.py [--caso p8_metrica_entrega p9_triagem] [--incluir-caros]
    python pipeline/regressao.py --caso p8_triagem --registrar-base      (grava bench/base/<caso>.json)

Casos em bench/casos.json. Cada caso: `entradas` (caminhos; sha256 do arquivo ou dos frames.dataset do projeto,
conferido contra a base: entrada mudou = codigo 3), `comandos` (argv com {PY} {RAIZ} {PIPE} {SAIDA} {HOME} e os
nomes das entradas), `metricas` (nuvem/malha via pipeline/metricas.py, ou campos de um JSON) e `tolerancias`
({"abs": x} | {"rel": x} | {"igual": true} | {"min": x} | {"max": x}; padrao: igual para texto/inteiro/bool,
rel 0,02 para float). `limites` vale sem base (ex.: pontos == 834686). Cada rodada grava numa pasta NOVA em
pipeline/output/regressao/ (nunca sobrescreve). Casos `caro: true` so rodam com --incluir-caros.
Codigos: 0 ok; 1 regrediu; 2 comando falhou; 3 entrada mudou; 4 caso sem base (rode --registrar-base).
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

from comum import sha256

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
from recursos import baixa_prioridade, orcamento_threads  # noqa: E402

CASOS = os.path.join(RAIZ, "bench", "casos.json")
BASE = os.path.join(RAIZ, "bench", "base")
RODADAS = os.path.join(RAIZ, "pipeline", "output", "regressao")


def assinatura(entrada):
    """sha256 do arquivo, ou combinado dos data/*/cache/frames.dataset de uma pasta de projeto."""
    c = entrada["caminho"]
    if os.path.isfile(c):
        return sha256(c)
    if os.path.isdir(c):
        h = hashlib.sha256()
        datasets = (
            sorted(
                os.path.join(c, "data", g, "cache", "frames.dataset")
                for g in os.listdir(os.path.join(c, "data"))
                if os.path.isfile(os.path.join(c, "data", g, "cache", "frames.dataset"))
            )
            if os.path.isdir(os.path.join(c, "data"))
            else []
        )
        if not datasets:
            raise FileNotFoundError(f"pasta sem frames.dataset: {c}")
        for d in datasets:
            h.update(os.path.relpath(d, c).replace("\\", "/").encode())
            h.update(sha256(d).encode())
        return h.hexdigest()
    raise FileNotFoundError(f"entrada nao existe: {c}")


def substituir(x, ctx):
    if isinstance(x, str):
        for k, v in ctx.items():
            x = x.replace("{" + k + "}", v)
        return x
    if isinstance(x, list):
        return [substituir(i, ctx) for i in x]
    if isinstance(x, dict):
        return {k: substituir(v, ctx) for k, v in x.items()}
    return x


def achatar(d, prefixo=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(achatar(v, f"{prefixo}{k}."))
    elif isinstance(d, (int, float, str, bool)) or d is None:
        out[prefixo[:-1]] = d
    return out


def campo(d, caminho):
    for p in caminho.split("."):
        if isinstance(d, list):
            d = d[int(p)]
        elif isinstance(d, dict):
            d = d.get(p)
        else:
            return None
        if d is None:
            return None
    return d


def medir(spec):
    """Uma especificacao de metrica -> dict plano {nome: valor}."""
    t = spec["tipo"]
    pre = spec.get("prefixo", "")
    if t in ("nuvem", "malha"):
        from metricas import medir as medir_nuvem

        r = medir_nuvem(
            spec["arquivo"],
            spec.get("referencia"),
            spec.get("rugosidade", t == "malha"),
            spec.get("componentes", False),
            malha=t == "malha",
        )
        r.pop("arquivo", None)
        r.pop("referencia", None)
        return {pre + k: v for k, v in achatar(r).items()}
    if t == "json":
        with open(spec["arquivo"], encoding="utf-8") as f:
            d = json.load(f)
        out = {}
        for nome, cam in spec.get("campos", {}).items():
            out[pre + nome] = campo(d, cam)
        for nome, c in spec.get("contem", {}).items():
            v = campo(d, c["chave"])
            texto = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            out[pre + nome] = c["texto"] in texto
        return out
    raise ValueError(f"tipo de metrica desconhecido: {t}")


def comparar(valores, base, tolerancias, limites):
    """Lista de problemas (vazia = ok)."""
    prob = []
    for k, lim in (limites or {}).items():
        if k not in valores:
            prob.append(f"{k}: ausente (limite {lim})")
            continue
        v = valores[k]
        if v is None and ("min" in lim or "max" in lim):
            prob.append(f"{k}: sem valor (limite {lim})")
            continue
        if "igual" in lim and v != lim["igual"]:
            prob.append(f"{k}: {v!r} != {lim['igual']!r}")
        if "min" in lim and v < lim["min"]:
            prob.append(f"{k}: {v} < minimo {lim['min']}")
        if "max" in lim and v > lim["max"]:
            prob.append(f"{k}: {v} > maximo {lim['max']}")
    if base is None:
        return prob
    for k, b in base.items():
        if k not in valores:
            prob.append(f"{k}: sumiu (base {b!r})")
            continue
        v = valores[k]
        tol = (tolerancias or {}).get(k)
        if tol is None:
            tol = {"rel": 0.02} if isinstance(b, float) and not isinstance(b, bool) else {"igual": True}
        if tol.get("ignorar"):
            continue
        if tol.get("igual"):
            if v != b:
                prob.append(f"{k}: {v!r} != base {b!r}")
            continue
        if v is None or b is None:
            if v != b:
                prob.append(f"{k}: {v!r} != base {b!r}")
            continue
        d = abs(float(v) - float(b))
        if "abs" in tol and d > tol["abs"]:
            prob.append(f"{k}: {v} vs base {b} (|d| {d:.4g} > {tol['abs']})")
        if "rel" in tol and d > tol["rel"] * max(abs(float(b)), 1e-12):
            prob.append(f"{k}: {v} vs base {b} (rel {d / max(abs(float(b)), 1e-12):.3%} > {tol['rel']:.1%})")
    return prob


def rodar_caso(caso, registrar=False, raiz_rodadas=RODADAS, base_dir=BASE, log=print):
    nome = caso["nome"]
    arq_base = os.path.join(base_dir, nome + ".json")
    base = json.load(open(arq_base, encoding="utf-8")) if os.path.exists(arq_base) else None
    raiz = os.path.join(raiz_rodadas, time.strftime("%Y%m%d_%H%M%S") + "_" + nome)
    saida, n = raiz, 1
    while True:  # pasta NOVA sempre (duas rodadas no mesmo segundo ganham sufixo)
        try:
            os.makedirs(saida)
            break
        except FileExistsError:
            n += 1
            saida = f"{raiz}_{n}"
    ctx = dict(PY=sys.executable, RAIZ=RAIZ, PIPE=AQUI, SAIDA=saida, HOME=os.path.expanduser("~"))
    entradas = substituir(caso.get("entradas", {}), ctx)
    assin = {}
    for k, e in entradas.items():
        ctx[k] = e["caminho"]
        if e.get("verificar", True):
            assin[k] = assinatura(e)
    res = dict(caso=nome, pasta=saida, entradas=assin, inicio=time.strftime("%Y-%m-%d %H:%M:%S"))
    if base and not registrar:
        mud = {k: (v, base["entradas"].get(k)) for k, v in assin.items() if base["entradas"].get(k) not in (None, v)}
        if mud:
            res.update(
                status="entrada_mudou", problemas=[f"{k}: sha {v[0][:12]} != base {v[1][:12]}" for k, v in mud.items()]
            )
            return res, 3
    env = dict(os.environ)
    if caso.get("threads"):  # padrao: orcamento do recursos.py (modo maximo herdado pelos comandos)
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "REVO_THREADS"):
            env[v] = str(caso["threads"])
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    for i, cmd in enumerate(substituir(caso.get("comandos", []), ctx)):
        with open(os.path.join(saida, f"comando_{i}.log"), "w", encoding="utf-8") as lg:
            lg.write(" ".join(cmd) + "\n\n")
            lg.flush()
            rc = subprocess.run(cmd, cwd=RAIZ, env=env, stdout=lg, stderr=subprocess.STDOUT).returncode
        if rc not in caso.get("codigos_aceitos", [0]):
            res.update(status="comando_falhou", problemas=[f"comando {i} saiu com {rc} (ver comando_{i}.log)"])
            return res, 2
    res["segundos_comandos"] = round(time.time() - t0, 1)
    valores = {}
    for spec in substituir(caso.get("metricas", []), ctx):
        valores.update(medir(spec))
    res["valores"] = valores
    prob = comparar(
        valores, None if registrar else (base or {}).get("valores"), caso.get("tolerancias"), caso.get("limites")
    )
    res["problemas"] = prob
    if registrar:
        if prob:
            res["status"] = "limites_violados"
            return res, 1
        os.makedirs(base_dir, exist_ok=True)
        with open(arq_base, "w", encoding="utf-8") as f:
            json.dump(
                dict(caso=nome, registrado=res["inicio"], entradas=assin, valores=valores),
                f,
                indent=1,
                ensure_ascii=False,
            )
        res["status"] = "base_registrada"
        return res, 0
    if base is None:
        res["status"] = "sem_base"
        return res, 4 if not prob else 1
    res["status"] = "regrediu" if prob else "ok"
    return res, 1 if prob else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--casos", default=CASOS)
    ap.add_argument("--caso", nargs="+", help="nomes (padrao: todos os nao caros)")
    ap.add_argument("--incluir-caros", action="store_true")
    ap.add_argument("--registrar-base", action="store_true")
    ap.add_argument("--listar", action="store_true")
    a = ap.parse_args(argv)
    baixa_prioridade()
    orcamento_threads(1)
    casos = json.load(open(a.casos, encoding="utf-8"))["casos"]
    if a.listar:
        for c in casos:
            tem = os.path.exists(os.path.join(BASE, c["nome"] + ".json"))
            print(
                f"{c['nome']:28s} {'CARO ' if c.get('caro') else '     '}{'base ok' if tem else 'sem base'} | {c.get('descricao', '')}"
            )
        return 0
    sel = [
        c for c in casos if (a.caso and c["nome"] in a.caso) or (not a.caso and (a.incluir_caros or not c.get("caro")))
    ]
    if a.caso and len(sel) != len(set(a.caso)):
        ap.error(f"caso desconhecido: {sorted(set(a.caso) - {c['nome'] for c in casos})}")
    pior = 0
    relatorio = []
    for c in sel:
        print(f"== {c['nome']}", flush=True)
        try:
            r, cod = rodar_caso(c, a.registrar_base)
        except FileNotFoundError as e:
            r, cod = dict(caso=c["nome"], status="entrada_ausente", problemas=[str(e)]), 3
        relatorio.append(r)
        print(
            f"   {r['status']}"
            + (f" ({r.get('segundos_comandos')} s)" if r.get("segundos_comandos") is not None else ""),
            flush=True,
        )
        for p in r.get("problemas", []):
            print("   -", p, flush=True)
        if "pasta" in r:
            with open(os.path.join(r["pasta"], "resultado.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1, ensure_ascii=False, default=float)
        pior = max(pior, cod) if cod != 4 or pior == 0 else pior
    return pior


if __name__ == "__main__":
    sys.exit(main())
