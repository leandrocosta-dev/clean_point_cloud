"""Catalogo: categoria, titulo e agrupamento de cada script; especificacao manual dos que leem sys.argv."""
import glob
import os

from introspect import ler_script

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE = os.path.join(RAIZ, "pipeline")

OCULTOS = {"revo_ui.py", "dph.py", "mundo_scan.py", "revo_projeto.py", "__init__.py"}

# arquivo (relativo a pipeline/) -> (categoria, titulo)
CATEGORIAS = [
    ("Pipeline Revo", {
        "rodar_tudo.py": "Pipeline completo (limpa, une, funde, exporta)",
        "limpar_raw.py": "Limpar frames (frames.dataset)",
        "limpar_por_placa.py": "Limpar frames por altura sobre a placa",
        "uniao_scans.py": "Unir duas partes (frames de B em A)",
        "revo_pipeline.py": "Fusão, malha e exportação no Revo",
        "build_revo_project.py": "Montar projeto .revo",
        "clean_revo.py": "Limpar .revo (CloudCompare)",
        "fuse_mesh.py": "Malha a partir de fuse.ply",
        "aprender.py": "Aprendizado (extrair/treinar)",
    }),
    ("Alinhamento de muitos scans (receita C)", {
        "alinhar_varios.py": "1. Alinhamento inicial (âncoras)",
        "alinhar_segmentos.py": "2. Segmentos e grafo de poses",
        "alinhar_direto.py": "3. Otimização conjunta (alinhar_direto)",
        "limpar_por_consenso.py": "4. Limpeza por consenso",
        "remapear_segmentos.py": "4b. Remapear segmentos",
        "cobertura.py": "Ordem gulosa de cobertura",
        "selecionar_segmentos.py": "5. Selecionar segmentos mínimos",
        "montar_pecas.py": "5b. Montar scan único + .revo",
        "avaliar_segmentos.py": "Avaliar viés por segmento",
        "avaliar_por_scan.py": "Avaliar viés por scan",
        "exp_pesos.py": "Experimento de pesos de odometria",
    }),
    ("Fusão própria e malha", {
        "fundir_proprio.py": "6. Fusão própria",
        "refinar_nuvem.py": "6b. Refinar nuvem",
        "uniao_revo_propria.py": "6c. União Revo + própria",
        "orientar_normais.py": "6d. Orientar normais",
        "malha_poisson.py": "6e. Malha de Poisson",
        "malha_meshlib.py": "Reparar/medir malha (MeshLib)",
        "completar_com_poisson.py": "Completar malha com Poisson",
        "poisson_de_pedacos.py": "Poisson de pedaços",
        "confianca.py": "Mapa de confiança",
        "limpar_fragmentos.py": "Remover fragmentos isolados",
    }),
    ("Construção por cobertura", {
        "construcao_por_cobertura.py": "Construção por cobertura",
        "base_por_cobertura.py": "Base por cobertura",
        "avaliar_fusao.py": "Avaliar fusão",
        "faltando.py": "O que falta na fundida",
        "buracos.py": "Buracos por contorno",
    }),
    ("Face interna", {
        "alinhar_interna_por_ponte.py": "Alinhar interna por scan-ponte",
        "corrigir_interna_suave.py": "Corrigir interna (campo suave)",
        "corrigir_interna_local.py": "Corrigir interna (local, não usar)",
        "sem_atravessar.py": "Cortar atravessamentos",
        "limpar_interna_com_externa.py": "Limpar interna com a externa",
        "deslocar_interna.py": "Deslocar interna",
        "encaixar_espessura.py": "Encaixar espessura nominal",
        "rugosidade_interna.py": "Medir rugosidade",
    }),
    ("Furos e reparo", {
        "fechar_furos_pequenos.py": "Fechar furinhos (com dado)",
        "fechar_furos_com_dado.py": "Fechar furos com dado",
        "remover_remendos_discordantes.py": "Remover remendos discordantes",
        "lacunas.py": "Relatório de lacunas",
    }),
    ("Medição e inspeção", {
        "metrica_malha.py": "Métricas de malha",
        "vista_malha.py": "Vista por trás (frente/verso)",
        "meshinspector_ui.py": "Captura do MeshInspector",
        "auditar_limpeza.py": "Auditar limpeza",
        "avaliar_periferia.py": "Avaliar periferia",
    }),
    ("Formato antigo (.dph)", {
        "revo_antigo/limpar_dph.py": "Limpar .dph",
        "revo_antigo/limpar_3d.py": "Limpar 3D",
        "revo_antigo/limpar_mesa.py": "Limpar mesa",
        "revo_antigo/limpar_por_fuse.py": "Limpar por fuse",
        "revo_antigo/auditar_limpeza_dph.py": "Auditar limpeza .dph",
        "revo_antigo/nuvem_scan.py": "Nuvem de um scan",
        "revo_antigo/nuvem_unida.py": "Nuvem unida",
        "revo_antigo/unir_dois.py": "Unir dois scans",
        "revo_antigo/unir_malhas_deriva.py": "Unir malhas com deriva",
        "revo_antigo/alinhar_crescendo.py": "Alinhar crescendo",
        "revo_antigo/alinhar_global.py": "Alinhar global",
        "revo_antigo/refinar_poses.py": "Refinar poses",
        "revo_antigo/tsdf_scan.py": "Malha TSDF",
        "revo_antigo/completar_base_revo.py": "Completar base Revo",
        "revo_antigo/completar_varias.py": "Completar com várias",
        "revo_antigo/limpar_malha_por_nuvem.py": "Limpar malha por nuvem",
    }),
    ("Creality", {
        "creality/alinhar_lados.py": "Alinhar dois lados",
        "creality/alinhar_malhas.py": "Alinhar malhas",
        "creality/fundir_malhas.py": "Fundir malhas",
        "creality/recortar_base.py": "Recortar base",
        "creality/recortar_faixas.py": "Recortar faixas",
        "creality/limpa_nuvem.py": "Limpar nuvem (caminhos fixos)",
        "creality/malha_gato.py": "Malha do gatinho (caminhos fixos)",
    }),
]

# agrupamento visual para os scripts principais: titulo do grupo -> destinos
GRUPOS = {
    "rodar_tudo.py": [
        ("Entrada e saída", ["projeto", "so_limpar", "saida_limpo", "saida_ply", "saida_uniao", "scans"]),
        ("Limpeza dos frames", ["periferia_mm", "periferia_banda", "periferia_alargue", "jobs"]),
        ("União de duas partes", ["unir", "lados"]),
        ("Resolução e malha (Revo)", ["ponto", "malha", "otimizar", "grade"]),
        ("Entregáveis e opções do Revo", ["entregaveis", "furos", "marcadores", "manter_isolados"]),
    ],
    "revo_pipeline.py": [
        ("Entrada e saída", ["revo", "saida", "scans", "reusar", "so_exportar", "fechar"]),
        ("Resolução e malha", ["ponto", "malha", "otimizar", "grade"]),
        ("Entregáveis e opções do Revo", ["entregaveis", "furos", "marcadores", "manter_isolados"]),
    ],
    "limpar_por_placa.py": [
        ("Entrada e saída", ["projeto", "saida", "scans", "diagnostico"]),
        ("Onde está a peça", ["altura_min", "voxel", "componentes", "margem_vox"]),
        ("Quadros e amostragem", ["min_pix_quadro", "sem_pixels", "amostra_quadros", "amostra_passo"]),
    ],
    "limpar_raw.py": [
        ("Entrada e saída", ["projeto", "saida", "scans", "sem_ply_antigos", "jobs"]),
        ("Filtros", ["sem_pixels", "sem_ilhas", "so_peca", "ext_max_mm", "min_mantido"]),
        ("Remoção da mesa (periferia)", ["periferia_mm", "periferia_banda", "periferia_alargue"]),
    ],
}

# ajustes de rotulo para destinos conhecidos
ROTULOS = {
    "ponto": "Resolução da nuvem (Point Distance, mm)",
    "grade": "Resolução da malha (Grid Size, mm)",
    "malha": "Tipo de malha",
    "entregaveis": "Entregáveis exportados por scan",
    "periferia_mm": "Remover mesa além de (mm)",
    "periferia_banda": "Espessura da faixa removida (mm)",
    "periferia_alargue": "Alargue do corpo (mm)",
    "ext_max_mm": "Extensão máxima da peça (mm)",
    "saida_ply": "Pasta de saída (nuvem e malha)",
    "so_limpar": "Só limpar e entregar o .revo (sem Revo)",
    "saida_limpo": "Pasta do projeto limpo",
    "saida_uniao": "Pasta da união",
}


def P(dest, tipo="str", ajuda="", posicional=False, obrigatorio=False, padrao=None, nargs=None, opcoes=None):
    flag = None if posicional else "--" + dest.replace("_", "-")
    return {"flag": flag, "aliases": [flag or dest], "dest": dest, "tipo": tipo, "acao": "store_true" if tipo == "flag" else None,
            "posicional": posicional, "obrigatorio": obrigatorio or (posicional and nargs != "*"), "padrao": padrao,
            "padrao_dinamico": False, "ajuda": ajuda, "opcoes": opcoes, "nargs": nargs, "metavar": None}


# scripts que leem sys.argv: especificacao manual
MANUAIS = {
    "avaliar_fusao.py": [P("ply", ajuda="nuvem fundida (.ply)", posicional=True), P("pasta", ajuda="pasta de segmentos", posicional=True),
                         P("segjson", ajuda="segmentos_indice.json", posicional=True), P("selecao", ajuda="selecao.json", posicional=True)],
    "avaliar_por_scan.py": [P("pasta", posicional=True), P("arq", ajuda="json de poses", posicional=True),
                            P("saida", ajuda="json de saida (opcional)", posicional=True, nargs="?")],
    "avaliar_segmentos.py": [P("pasta", posicional=True), P("arquivos", ajuda="jsons de poses a comparar", posicional=True, nargs="+")],
    "buracos.py": [P("malha", posicional=True), P("pasta", posicional=True), P("segjson", posicional=True), P("selecao", posicional=True),
                   P("saida", ajuda="imagem/relatorio de saida")],
    "faltando.py": [P("ply", posicional=True), P("pasta", posicional=True), P("segjson", posicional=True), P("selecao", posicional=True),
                    P("saida", obrigatorio=True)],
    "exp_pesos.py": [P("work", ajuda="pasta de trabalho dos segmentos", posicional=True), P("poses", ajuda="json de poses iniciais", posicional=True),
                     P("pesos", ajuda="pesos de odometria a testar", posicional=True, nargs="+")],
    "metrica_malha.py": [P("malhas", ajuda="uma ou mais malhas (.ply)", posicional=True, nargs="+")],
    "creality/alinhar_lados.py": [P("a", ajuda="nuvem do lado A", posicional=True), P("b", ajuda="nuvem do lado B", posicional=True),
                                  P("saida", ajuda="pasta de saida", posicional=True)],
    "creality/alinhar_malhas.py": [P("a", posicional=True), P("b", posicional=True), P("saida", posicional=True)],
    "creality/fundir_malhas.py": [P("a", posicional=True), P("b", posicional=True), P("saida", posicional=True),
                                  P("corte", "float", "distancia de corte (mm)", padrao=1.5)],
    "creality/recortar_base.py": [P("entrada", posicional=True), P("saida", posicional=True)],
    "creality/recortar_faixas.py": [P("entrada", posicional=True), P("saida", posicional=True)],
    "creality/limpa_nuvem.py": [],
    "creality/malha_gato.py": [],
}


def _titulo_padrao(rel):
    return os.path.splitext(os.path.basename(rel))[0].replace("_", " ").capitalize()


_CACHE = None


def carregar_cache():
    global _CACHE
    if _CACHE is None:
        _CACHE = carregar()
    return _CACHE


def carregar():
    """-> lista de {id, categoria, titulo, rel, caminho, descricao, params, manual}"""
    achados = {os.path.relpath(f, PIPE).replace("\\", "/"): f for f in glob.glob(os.path.join(PIPE, "**", "*.py"), recursive=True)}
    itens, usados = [], set()
    for categoria, scripts in CATEGORIAS + [("Outros", {})]:
        rels = list(scripts) if categoria != "Outros" else sorted(r for r in achados if r not in usados)
        for rel in rels:
            if rel not in achados or os.path.basename(rel) in OCULTOS or rel in usados:
                continue
            usados.add(rel)
            info = ler_script(achados[rel])
            manual = rel in MANUAIS
            if manual:
                info["params"] = MANUAIS[rel]
            itens.append({
                "id": rel, "categoria": categoria, "titulo": scripts.get(rel, _titulo_padrao(rel)), "rel": rel, "caminho": achados[rel],
                "descricao": info["descricao"], "params": info["params"], "manual": manual,
                "grupos": GRUPOS.get(os.path.basename(rel)) if "/" not in rel else None,
            })
    return itens
