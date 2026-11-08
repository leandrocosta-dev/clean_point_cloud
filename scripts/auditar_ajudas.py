"""Audita --help de CLIs argparse sem abrir scans ou executar scripts manuais.

    python scripts/auditar_ajudas.py --saida pipeline/output/auditoria_ajudas_NOVA

Usa AST para escolher os scripts; nao importa os modulos de processamento.
Os processos de ajuda rodam de uma pasta externa, com tempo limitado.
"""

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
# Inspecionados por AST: o parse_args interrompe --help antes de leitura de scans,
# gravacao ou automacao. A lista explicita evita executar novos scripts legados por acidente.
TOP_LEVEL_REVISADOS = {
    "alinhar_interna_por_ponte.py",
    "base_por_cobertura.py",
    "completar_com_poisson.py",
    "confianca.py",
    "corrigir_interna_suave.py",
    "criar_projeto_biblioteca.py",
    "fechar_furos_pequenos.py",
    "lacunas.py",
    "limpar_interna_com_externa.py",
    "malha_poisson.py",
    "montar_pecas.py",
    "remover_remendos_discordantes.py",
    "revo_antigo/auditar_limpeza_dph.py",
    "revo_antigo/completar_base_revo.py",
    "revo_antigo/limpar_malha_por_nuvem.py",
    "revo_antigo/limpar_por_fuse.py",
    "revo_antigo/nuvem_unida.py",
    "revo_antigo/unir_malhas_deriva.py",
    "rugosidade_interna.py",
    "sem_atravessar.py",
}


def nome_chamada(node: ast.AST) -> str:
    return ast.unparse(node.func) if isinstance(node, ast.Call) else ""


def classificar(path: Path, incluir_top_level: bool = False) -> dict:
    texto = path.read_text(encoding="utf-8-sig")
    arvore = ast.parse(texto, filename=str(path))
    chamadas = [nome_chamada(n) for n in ast.walk(arvore) if isinstance(n, ast.Call)]
    argparse_real = any(n.endswith("ArgumentParser") for n in chamadas)
    parse = any(n.endswith(("parse_args", "parse_known_args")) for n in chamadas)
    usa_argv = any(
        isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "sys" and n.attr == "argv"
        for n in ast.walk(arvore)
    )
    guard = any(
        isinstance(n, ast.If) and "__name__" in ast.unparse(n.test) and "__main__" in ast.unparse(n.test)
        for n in arvore.body
    )
    # Modulos antigos com parser no nivel superior sao apenas inventariados: importar seus
    # auxiliares pode executar processamento antes do parser, portanto nao e seguro roda-los aqui.
    revisado = path.relative_to(RAIZ / "pipeline").as_posix() in TOP_LEVEL_REVISADOS
    executar = argparse_real and parse and (guard or (incluir_top_level and revisado))
    return dict(
        arquivo=str(path.relative_to(RAIZ)),
        argparse=argparse_real,
        parse_args=parse,
        main_protegido=guard,
        top_level_revisado=revisado,
        argv_manual=usa_argv and not argparse_real,
        executar=executar,
        motivo=(
            "argparse com entrada protegida"
            if guard and executar
            else "argparse top-level revisado: ajuda antes de processamento"
            if executar
            else "CLI sem argparse: nao executada"
            if usa_argv and not argparse_real
            else "parser no nivel superior: revisar antes de executar"
            if argparse_real and not guard
            else "modulo auxiliar ou entrada indireta"
        ),
    )


def executar_ajuda(item: dict, cwd: str, saida: Path, timeout: float) -> dict:
    script = RAIZ / item["arquivo"]
    inicio = time.perf_counter()
    antes = hashlib.sha256(script.read_bytes()).hexdigest()
    try:
        p = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=cwd,
            capture_output=True,
            timeout=timeout,
        )
        stdout, stderr = p.stdout.decode("utf-8", errors="replace"), p.stderr.decode("utf-8", errors="replace")
        codigo = p.returncode
        estado = "ok" if codigo == 0 and "usage:" in stdout.lower() else "erro"
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace")
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        codigo, estado = None, "timeout"
    log = saida / (item["arquivo"].replace("\\", "__").replace("/", "__") + ".log")
    log.write_text(stdout + "\nSTDERR:\n" + stderr, encoding="utf-8")
    return dict(
        item,
        estado=estado,
        codigo=codigo,
        segundos=round(time.perf_counter() - inicio, 3),
        sha256_antes=antes,
        fonte_inalterada=antes == hashlib.sha256(script.read_bytes()).hexdigest(),
        log=str(log),
        stderr=stderr[-4000:],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--incluir-top-level-revisados", action="store_true")
    args = ap.parse_args()
    saida = args.saida.resolve()
    saida.mkdir(parents=True, exist_ok=False)
    itens = []
    for arquivo in sorted((RAIZ / "pipeline").rglob("*.py")):
        if any(x in arquivo.parts for x in ("output", "__pycache__", "legado")):
            continue
        itens.append(classificar(arquivo, args.incluir_top_level_revisados))
    selecionados = [x for x in itens if x["executar"]]
    with tempfile.TemporaryDirectory(prefix="ajudas_cwd_externo_") as tmp:
        with ThreadPoolExecutor(max(1, min(args.jobs, 4))) as ex:
            resultados = list(ex.map(lambda item: executar_ajuda(item, tmp, saida, args.timeout), selecionados))
    r = dict(
        python=sys.executable,
        cwd_externo=True,
        inventario=itens,
        ajudas=resultados,
        resumo=dict(
            inventariados=len(itens),
            executados=len(resultados),
            aprovados=sum(x["estado"] == "ok" for x in resultados),
            erros=sum(x["estado"] != "ok" for x in resultados),
            fontes_alteradas=sum(not x["fonte_inalterada"] for x in resultados),
            cli_manuais=sum(x["argv_manual"] for x in itens),
        ),
    )
    (saida / "auditoria_ajudas.json").write_text(json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(r["resumo"], indent=2))
    for x in resultados:
        if x["estado"] != "ok":
            print(x["arquivo"], x["estado"], x["stderr"][-500:])
    return int(bool(r["resumo"]["erros"]) or bool(r["resumo"]["fontes_alteradas"]))


if __name__ == "__main__":
    raise SystemExit(main())
