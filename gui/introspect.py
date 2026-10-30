"""Le os parametros de um script por AST (sem importar nem executar): add_argument -> especificacao."""
import ast
import os


def _lit(node, consts):
    try:
        return ast.literal_eval(node)
    except Exception:
        pass
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.Attribute) and node.attr in consts:
        return consts[node.attr]
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "list" and node.args:
        v = _lit(node.args[0], consts)
        if isinstance(v, dict):
            return list(v)
        if isinstance(v, (list, tuple)):
            return list(v)
    return None


def _constantes(tree):
    consts = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                consts[n.targets[0].id] = ast.literal_eval(n.value)
            except Exception:
                pass
    return consts


def _tipo(node):
    if isinstance(node, ast.Name):
        return {"int": "int", "float": "float", "str": "str"}.get(node.id, "str")
    return "str"


def _spec_de_chamada(call, consts, nomes_loop=None):
    nomes = [_lit(a, consts) for a in call.args]
    if nomes_loop is not None and call.args and isinstance(call.args[0], ast.Name) and call.args[0].id == nomes_loop[0]:
        nomes = [nomes_loop[1]]
    nomes = [n for n in nomes if isinstance(n, str)]
    if not nomes:
        return None
    kw = {k.arg: k.value for k in call.keywords if k.arg}
    opcional = nomes[0].startswith("-")
    longo = next((n for n in nomes if n.startswith("--")), nomes[0])
    dest = _lit(kw["dest"], consts) if "dest" in kw else longo.lstrip("-").replace("-", "_")
    acao = _lit(kw["action"], consts) if "action" in kw else None
    nargs = _lit(kw["nargs"], consts) if "nargs" in kw else None
    metavar = _lit(kw["metavar"], consts) if "metavar" in kw else None
    if acao in ("store_true", "store_false"):
        tipo = "flag"
    elif acao in ("append",):
        tipo = "append"
    else:
        tipo = _tipo(kw.get("type"))
    spec = {
        "flag": longo if opcional else None,
        "aliases": nomes,
        "dest": dest,
        "tipo": tipo,
        "acao": acao,
        "posicional": not opcional,
        "obrigatorio": (nargs not in ("?", "*")) if not opcional else bool(_lit(kw["required"], consts) if "required" in kw else False),
        "padrao": _lit(kw["default"], consts) if "default" in kw else None,
        "padrao_dinamico": "default" in kw and _lit(kw["default"], consts) is None and not (isinstance(kw["default"], ast.Constant)),
        "ajuda": (_lit(kw["help"], consts) or "") if "help" in kw else "",
        "opcoes": _lit(kw["choices"], consts) if "choices" in kw else None,
        "nargs": nargs,
        "metavar": metavar,
    }
    if isinstance(spec["ajuda"], str):
        spec["ajuda"] = " ".join(spec["ajuda"].split())
    else:
        spec["ajuda"] = ""
    return spec


def ler_script(caminho):
    """Devolve {"arquivo", "descricao", "params": [...], "argparse": bool}."""
    with open(caminho, encoding="utf-8", errors="replace") as f:
        src = f.read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return {"arquivo": caminho, "descricao": "", "params": [], "argparse": False}
    consts = _constantes(tree)
    doc = ast.get_docstring(tree) or ""
    params, vistos = [], set()

    def visita(no, loop=None):
        for filho in ast.iter_child_nodes(no):
            if isinstance(filho, ast.For) and isinstance(filho.target, ast.Name):
                itens = _lit(filho.iter, consts)
                if isinstance(itens, (list, tuple)) and all(isinstance(i, str) for i in itens):
                    for item in itens:
                        visita_corpo(filho, (filho.target.id, item))
                    continue
            if isinstance(filho, ast.Call) and getattr(filho.func, "attr", "") == "add_argument":
                s = _spec_de_chamada(filho, consts, loop)
                if s and s["dest"] not in vistos:
                    vistos.add(s["dest"])
                    params.append(s)
            visita(filho, loop)

    def visita_corpo(no_for, loop):
        for stmt in no_for.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and getattr(stmt.value.func, "attr", "") == "add_argument":
                s = _spec_de_chamada(stmt.value, consts, loop)
                if s and s["dest"] not in vistos:
                    vistos.add(s["dest"])
                    params.append(s)
            else:
                visita(stmt, loop)

    visita(tree)
    # posicionais primeiro, na ordem em que foram declarados
    params = [p for p in params if p["posicional"]] + [p for p in params if not p["posicional"]]
    return {"arquivo": caminho, "descricao": doc, "params": params, "argparse": "argparse" in src}


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(ler_script(sys.argv[1]), indent=1, ensure_ascii=False))
