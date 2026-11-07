"""Cria copia com IDs independentes para importar ao lado do projeto original.

    python pipeline/preparar_importacao.py PROJETO --saida PASTA_NOVA

O Revo pode ignorar scans cujos IDs ja estao na biblioteca. Atualiza pastas,
nodes e referencias JSON, incluindo merge_depends, sem alterar frames ou poses.
"""
import argparse
import json
from pathlib import Path
import shutil
import uuid


def substituir_ids(valor, mapa):
    if isinstance(valor, dict):
        return {k: substituir_ids(v, mapa) for k, v in valor.items()}
    if isinstance(valor, list):
        return [substituir_ids(v, mapa) for v in valor]
    if isinstance(valor, str):
        return mapa.get(valor, valor)
    return valor


def reidentificar(arquivo):
    """Somente para projetos de SAIDA: renomeia scans e devolve o mapa de IDs."""
    arquivo = Path(arquivo).resolve()
    saida = arquivo.parent
    projeto = json.loads(arquivo.read_text(encoding="utf-8"))
    ids = [n["guid"] for n in projeto["nodes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("projeto contem IDs duplicados")
    for ident in ids:
        if Path(ident).name != ident or ident in ('.', '..') or not (saida / "data" / ident).is_dir():
            raise ValueError(f"pasta de scan ausente ou ID invalido: {ident}")
    mapa = {ident: str(uuid.uuid4()) for ident in ids}
    nomes = {mapa[n['guid']]: n['name'] for n in projeto['nodes']}
    for antigo, novo in mapa.items():
        pasta = saida / "data" / antigo
        for arq in pasta.rglob("*.rvproj"):
            conteudo = json.loads(arq.read_text(encoding="utf-8"))
            arq.write_text(json.dumps(substituir_ids(conteudo, mapa), indent=4), encoding="utf-8")
        pasta.rename(saida / "data" / novo)
    projeto = substituir_ids(projeto, mapa)
    projeto["guid"] = str(uuid.uuid4())
    projeto["name"] = saida.name
    for node in projeto["nodes"]:
        node["name"] = nomes[node['guid']] + "_COPIA"
    projeto['size'] = str(sum(p.stat().st_size for p in (saida/'data').rglob('*') if p.is_file()))
    arquivo.write_text(json.dumps(projeto, indent=4), encoding="utf-8")
    (saida / "identificadores.json").write_text(json.dumps(mapa, indent=2), encoding="utf-8")
    return mapa


def preparar(origem, saida):
    origem, saida = Path(origem).resolve(), Path(saida).resolve()
    if origem == saida or origem in saida.parents:
        raise ValueError("saida deve estar fora do projeto de origem")
    arquivos = list(origem.glob("*.revo"))
    if len(arquivos) != 1:
        raise ValueError("origem deve conter exatamente um arquivo .revo")
    shutil.copytree(origem, saida)  # recusa destino existente
    destino = saida / arquivos[0].name
    reidentificar(destino)
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projeto")
    ap.add_argument("--saida", required=True, help="pasta nova, fora do projeto original")
    args = ap.parse_args()
    print(preparar(args.projeto, args.saida))


if __name__ == "__main__":
    main()
