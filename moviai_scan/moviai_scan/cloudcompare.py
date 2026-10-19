"""Limpeza e correcao de deriva pelo CloudCompare em linha de comando.

limpar: N scans PLY (aproximadamente alinhados no referencial do scanner)
  -> ICP sequencial + fusao incremental das clouds
  -> EXTRACT_CC (separa componentes desconexas) -> mantem so UMA (a peca;
     descarta base/mesa/suporte e specks de ruido). Qual componente e a peca
     (a maior ou a menor em pontos) depende da geometria do setup de scan:
     keep_largest_component.select ("largest" ou "smallest"). Default
     "smallest": numa base plana de frente pro scanner a amostragem costuma ser
     mais densa que numa peca em pe vista em angulo (foreshortening), entao a
     peca real pode ter MENOS pontos que a base. Confira sempre a listagem de
     componentes impressa antes de confiar na escolha.
  -> SOR -> Noise filter (RADIUS) -> Crop opcional -> Downsample espacial.
  A escolha da componente acontece aqui, entre duas chamadas do CloudCompare,
  lendo o cabecalho PLY de cada componente exportada.

corrigir_deriva: ajusta cilindros via RANSAC. Com deriva de rastreamento o
  ajuste fragmenta em varios cilindros; o maior e a "passada boa" e os menores
  (passadas desalinhadas) sao removidos da nuvem original por distancia C2C ~0.
  So pontos realmente escaneados sao mantidos. Revise a listagem "Cilindros
  detectados" e, se preciso, ajuste suporte.

Sintaxe dos comandos verificada contra o codigo-fonte oficial
(qCC/ccCommandLineCommands.cpp e plugins/core/Standard/qRANSAC_SD).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import uuid

CONFIG_PADRAO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloudcompare_padrao.json")
CAMINHOS_CC = [r"C:\Program Files\CloudCompare\CloudCompare.exe",
               r"C:\Program Files (x86)\CloudCompare\CloudCompare.exe"]
COMUNS = ["-SILENT", "-NO_TIMESTAMP", "-C_EXPORT_FMT", "PLY"]


def achar_cc(explicito=None, dry_run=False):
    if explicito:
        if not os.path.exists(explicito):
            raise FileNotFoundError(f"CloudCompare.exe nao encontrado em: {explicito}")
        return os.path.abspath(explicito)
    for c in CAMINHOS_CC:
        if os.path.exists(c):
            return c
    if dry_run:
        return "CloudCompare.exe"
    raise FileNotFoundError("CloudCompare.exe nao encontrado nos caminhos padrao. Use --cc <caminho>.")


def carregar_config(caminho=None):
    """Config padrao com as chaves do arquivo do usuario sobrepostas secao a secao."""
    with open(CONFIG_PADRAO, encoding="utf-8") as f:
        cfg = json.load(f)
    if caminho:
        with open(caminho, encoding="utf-8") as f:
            for secao, valores in json.load(f).items():
                cfg[secao].update(valores)
    return cfg


def _flags_icp(icp):
    f = ["-ICP", "-REFERENCE_IS_FIRST", "-MIN_ERROR_DIFF", str(icp["min_error_diff"]),
         "-ITER", str(icp["iterations"]), "-OVERLAP", str(icp["overlap"]),
         "-RANDOM_SAMPLING_LIMIT", str(icp["random_sampling_limit"])]
    if icp["farthest_removal"]:
        f.append("-FARTHEST_REMOVAL")
    return f


def _flags_ruido(nz):
    if nz["mode"] not in ("knn", "radius"):
        raise ValueError("noise.mode deve ser 'knn' ou 'radius'")
    f = ["-NOISE", nz["mode"].upper(), str(nz["value"]), nz["error_type"].upper(), str(nz["error"])]
    if nz["remove_isolated"]:
        f.append("RIP")
    return f


def _flags_carregar_alinhar(scans, cfg):
    f = ["-O", scans[0]]
    for s in scans[1:]:
        f += ["-O", s]
        if cfg["icp"]["enabled"]:
            f += _flags_icp(cfg["icp"])
        f.append("-MERGE_CLOUDS")
    return f


def _flags_limpeza(cfg):
    f = []
    if cfg["sor"]["enabled"]:
        f += ["-SOR", str(cfg["sor"]["knn"]), str(cfg["sor"]["sigma"])]
    if cfg["noise"]["enabled"]:
        f += _flags_ruido(cfg["noise"])
    if cfg["crop"]["enabled"]:
        f += ["-CROP", ":".join(str(x) for x in cfg["crop"]["box"])]
        if cfg["crop"]["outside"]:
            f.append("-OUTSIDE")
    if cfg["downsample"]["enabled"]:
        f += ["-SS", cfg["downsample"]["method"].upper(), str(cfg["downsample"]["value"])]
    return f


def contar_vertices_ply(path):
    with open(path, "rb") as f:
        texto = f.read(8192).decode("ascii", errors="replace")
    m = re.search(r"element vertex (\d+)", texto)
    if not m:
        raise ValueError(f"Nao foi possivel ler 'element vertex' do cabecalho PLY: {path}")
    return int(m.group(1))


def executar_cc(cc, args, pasta_log, prefixo, log=print):
    """Roda o CloudCompare guardando stdout/stderr em <pasta_log>/<prefixo>.*.log."""
    log(f"Comando CloudCompare ({prefixo}):")
    log(subprocess.list2cmdline([cc] + args))
    out_log = os.path.join(pasta_log, f"{prefixo}.stdout.log")
    err_log = os.path.join(pasta_log, f"{prefixo}.stderr.log")
    with open(out_log, "wb") as fo, open(err_log, "wb") as fe:
        r = subprocess.run([cc] + args, stdout=fo, stderr=fe)
    for p in (out_log, err_log):
        txt = open(p, encoding="utf-8", errors="replace").read().strip()
        if txt:
            log(txt)
    if r.returncode != 0:
        raise RuntimeError(f"CloudCompare falhou (exit code {r.returncode}) na etapa '{prefixo}'. Veja {out_log}")


def _ranking(paths):
    """[(n_pontos, caminho)] do maior para o menor."""
    return sorted(((contar_vertices_ply(p), p) for p in paths), reverse=True)


def _pasta_temporaria(base, prefixo):
    p = os.path.join(base, f"{prefixo}_{uuid.uuid4().hex[:8]}")
    os.makedirs(p)
    return p


def _conferir_saida(saida, log):
    if not os.path.exists(saida):
        raise RuntimeError(f"Falha: saida nao gerada em {saida}")
    log(f"OK: nuvem limpa salva em {saida}")


def limpar(scans, saida, cc_exe=None, config=None, dry_run=False, manter_temp=False, componente=None, log=print):
    """componente: pula a extracao e limpa direto um *_COMPONENT_*.ply ja extraido
    (rodada anterior com manter_temp em que a escolha automatica errou)."""
    saida = os.path.abspath(saida)
    pasta_saida = os.path.dirname(saida)
    os.makedirs(pasta_saida, exist_ok=True)
    cc = achar_cc(cc_exe, dry_run)
    cfg = carregar_config(config)

    if componente:
        if not os.path.exists(componente):
            raise FileNotFoundError(f"Componente nao encontrado: {componente}")
        args = COMUNS + ["-AUTO_SAVE", "OFF", "-O", os.path.abspath(componente)] + _flags_limpeza(cfg) \
            + ["-SAVE_CLOUDS", "FILE", saida]
        if dry_run:
            log(subprocess.list2cmdline([cc] + args))
            return
        executar_cc(cc, args, pasta_saida, "resume_clean", log)
        return _conferir_saida(saida, log)

    if not scans:
        raise ValueError("Informe os scans (ou --componente para retomar de um componente ja extraido)")
    for s in scans:
        if not os.path.exists(s):
            raise FileNotFoundError(f"Scan nao encontrado: {s}")
    scans = [os.path.abspath(s) for s in scans]

    if not cfg["keep_largest_component"]["enabled"]:
        args = COMUNS + ["-AUTO_SAVE", "OFF"] + _flags_carregar_alinhar(scans, cfg) + _flags_limpeza(cfg) \
            + ["-SAVE_CLOUDS", "FILE", saida]
        if dry_run:
            log(subprocess.list2cmdline([cc] + args))
            return
        executar_cc(cc, args, pasta_saida, "clean", log)
        return _conferir_saida(saida, log)

    # Dois estagios: 1) extrai as componentes desconexas numa pasta de staging
    # isolada (nunca escreve na pasta original dos scans); 2) limpa so a escolhida.
    kc = cfg["keep_largest_component"]
    marcador = "<componente escolhida>"
    estagio2 = COMUNS + ["-AUTO_SAVE", "OFF", "-O", marcador] + _flags_limpeza(cfg) + ["-SAVE_CLOUDS", "FILE", saida]
    if dry_run:
        # AUTO_SAVE ON faz o EXTRACT_CC exportar cada componente sozinho
        e1 = COMUNS + ["-AUTO_SAVE", "ON"] + _flags_carregar_alinhar(scans, cfg) \
            + ["-EXTRACT_CC", str(kc["octree_level"]), str(kc["min_points"])]
        log("Estagio 1 (extrai componentes):")
        log(subprocess.list2cmdline([cc] + e1))
        log("Estagio 2 (limpeza da componente escolhida entre *_COMPONENT_*.ply):")
        log(subprocess.list2cmdline([cc] + estagio2))
        return

    stage = _pasta_temporaria(pasta_saida, "_stage")
    try:
        copias = []
        for s in scans:
            copias.append(os.path.join(stage, os.path.basename(s)))
            shutil.copy(s, copias[-1])
        e1 = COMUNS + ["-AUTO_SAVE", "ON"] + _flags_carregar_alinhar(copias, cfg) \
            + ["-EXTRACT_CC", str(kc["octree_level"]), str(kc["min_points"])]
        executar_cc(cc, e1, stage, "stage1_extract", log)

        base = os.path.splitext(os.path.basename(copias[0]))[0]
        comps = glob.glob(os.path.join(glob.escape(stage), glob.escape(base) + "*_COMPONENT_*.ply"))
        if not comps:
            raise RuntimeError("EXTRACT_CC nao gerou nenhuma componente (ajuste keep_largest_component.min_points?)")
        rank = _ranking(comps)
        log(f"Componentes encontradas ({len(rank)}), por tamanho:")
        for i, (n, p) in enumerate(rank):
            log(f"  [{i}] {n} pontos - {p}")
        n, escolhida = rank[-1] if kc["select"] == "smallest" else rank[0]
        log(f"Selecao: '{kc['select']}' -> {escolhida} ({n} pontos)")
        log("Se a escolha estiver errada: rode de novo com --manter-temp, confira a pasta de staging "
            "e use --componente <caminho> apontando para o componente certo.")
        executar_cc(cc, [escolhida if a == marcador else a for a in estagio2], stage, "stage2_clean", log)
        _conferir_saida(saida, log)
    finally:
        if manter_temp:
            log(f"Staging mantido (--manter-temp): {stage}")
        else:
            shutil.rmtree(stage, ignore_errors=True)


def corrigir_deriva(entrada, saida, cc_exe=None, suporte=500, epsilon=0.001, manter_temp=False, log=print):
    """suporte: minimo de pontos para o RANSAC aceitar um cilindro (baixo fragmenta tudo,
    alto pode nao achar nada). epsilon: distancia C2C abaixo da qual um ponto da nuvem
    original e considerado copia de um ponto dos fragmentos descartados."""
    if not os.path.exists(entrada):
        raise FileNotFoundError(f"Arquivo nao encontrado: {entrada}")
    saida = os.path.abspath(saida)
    pasta_saida = os.path.dirname(saida)
    os.makedirs(pasta_saida, exist_ok=True)
    cc = achar_cc(cc_exe)
    entrada = os.path.abspath(entrada)

    stage = _pasta_temporaria(pasta_saida, "_drift")
    try:
        copia = os.path.join(stage, "input.ply")
        shutil.copy(entrada, copia)

        # 1) RANSAC: com deriva fragmenta em varios cilindros; sem deriva costuma
        #    achar 1 dominante (ou nenhum, se a peca nao for cilindrica).
        executar_cc(cc, COMUNS + ["-AUTO_SAVE", "OFF", "-O", copia,
                                  "-RANSAC", "ENABLE_PRIMITIVE", "CYLINDER", "SUPPORT_POINTS", str(suporte),
                                  "OUTPUT_INDIVIDUAL_SUBCLOUDS", "OUT_CLOUD_DIR", stage],
                    stage, "1_ransac", log)
        formas = glob.glob(os.path.join(glob.escape(stage), "*_CYLINDER_*_cloud.ply"))
        if not formas:
            log("Nenhum cilindro detectado: nada para corrigir. Copiando entrada para a saida sem alteracao.")
            shutil.copy(entrada, saida)
            return
        rank = _ranking(formas)
        log(f"Cilindros detectados ({len(rank)}):")
        for i, (n, p) in enumerate(rank):
            log(f"  [{i}] {n} pontos - {p}")
        if len(rank) == 1:
            log("So um cilindro consistente encontrado: sem fragmentacao, nada para descartar. "
                "Copiando entrada para a saida sem alteracao.")
            shutil.copy(entrada, saida)
            return
        ruins = rank[1:]
        log(f"Referencia (maior/mais consistente): {rank[0][1]} ({rank[0][0]} pontos)")
        log(f"Descartando {len(ruins)} fragmento(s) menor(es), total {sum(n for n, _ in ruins)} pontos")

        # 2) Junta os fragmentos ruins (a passada desalinhada) num arquivo so.
        ruins_ply = os.path.join(stage, "bad_fragments.ply")
        args = COMUNS + ["-AUTO_SAVE", "OFF"]
        for _, p in ruins:
            args += ["-O", p]
        executar_cc(cc, args + ["-MERGE_CLOUDS", "-SAVE_CLOUDS", "FILE", ruins_ply], stage, "2_merge_bad", log)

        # 3) Distancia C2C da original contra os fragmentos ruins + filtro: descarta
        #    pontos que sao copia (distancia ~0) de um ponto dos fragmentos. AUTO_SAVE ON
        #    porque -SAVE_CLOUDS FILE com varios nomes tem bug de parsing no CC.
        executar_cc(cc, COMUNS + ["-AUTO_SAVE", "ON", "-O", copia, "-O", ruins_ply,
                                  "-C2C_DIST", "-FILTER_SF", str(epsilon), "MAX"],
                    stage, "3_filter", log)
        res = glob.glob(os.path.join(glob.escape(stage), "input_C2C_DIST_FILTERED_*.ply"))
        if not res:
            raise RuntimeError(f"Nao achei o arquivo filtrado esperado (input_C2C_DIST_FILTERED_*.ply) em {stage}")
        shutil.copy(res[0], saida)
        log(f"OK: deriva corrigida, {contar_vertices_ply(saida)} pontos salvos em {saida}")
    finally:
        if manter_temp:
            log(f"Staging mantido (--manter-temp): {stage}")
        else:
            shutil.rmtree(stage, ignore_errors=True)
