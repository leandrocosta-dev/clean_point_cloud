"""Ciclo versionado: congelar -> fundir -> comparar -> inspecionar -> corrigir -> refundir.

    python pipeline/ciclo_precisao.py iniciar PROJETO --saida EXPERIMENTO
    python pipeline/ciclo_precisao.py triagem EXPERIMENTO
    python pipeline/ciclo_precisao.py limpar EXPERIMENTO --configuracao parametros.json
    python pipeline/ciclo_precisao.py consenso EXPERIMENTO --poses poses.json
    python pipeline/ciclo_precisao.py fundir EXPERIMENTO
    python pipeline/ciclo_precisao.py revo EXPERIMENTO
    python pipeline/ciclo_precisao.py inspecionar EXPERIMENTO --scan GUID --saida vista.html
    python pipeline/ciclo_precisao.py corrigir EXPERIMENTO --regioes regioes.json

Cada correcao cria revisao nova. Fusoes/merges anteriores nunca aprovam a nova.
Calculo numerico (fusao, comparacao, mascaras de regiao, projecao dos pixels) na GPU via gpu.py; SQLite, hashes e JSON na CPU.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import gpu
import numpy as np
from cache_observacoes import pontos_gpu
from comparar_fusao_cuda import avaliar, sha256, validar_ply
from fusao_superficies import executar, salvar_json
from preparar_importacao import preparar, reidentificar
from revo_projeto import RevoScan, gravar_ply, ler_ply


def ler(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inventario(projeto):
    """Manifesto cobre TODOS os arquivos, incluindo calibracao e propriedades."""
    return {p.relative_to(projeto).as_posix(): sha256(p) for p in sorted(Path(projeto).rglob("*")) if p.is_file()}


def carregar(root):
    root = Path(root).resolve()
    st = ler(root / "ciclo.json")
    rev = root / st["ativa"]
    m = ler(rev / "manifesto.json")
    if inventario(rev / "projeto") != m["arquivos"]:
        raise ValueError("entrada congelada foi alterada; criar nova revisao, nao reutilizar artefatos")
    return root, st, rev, m


def scans(projeto):
    arq = list(Path(projeto).glob("*.revo"))
    if len(arq) != 1:
        raise ValueError("esperado um unico .revo")
    r = ler(arq[0])
    ids = [
        n["guid"]
        for n in r["nodes"]
        if (Path(projeto) / "data" / n["guid"] / "cache/frames.dataset").is_file()
        and (Path(projeto) / "data" / n["guid"] / "raw_preview.ply").is_file()
    ]
    if not ids:
        raise ValueError("projeto sem scans de frames")
    return arq[0], ids


def manifesto(projeto, parent=None):
    arq, ids = scans(projeto)
    m = dict(
        versao=1,
        id=str(uuid.uuid4()),
        criado=datetime.now(timezone.utc).isoformat(),
        anterior=parent,
        projeto=arq.name,
        scans=ids,
        arquivos=inventario(projeto),
        referencial="mundo de cada scan; mm; poses originais preservadas",
    )
    aliases = Path(projeto) / "identificadores.json"
    if aliases.is_file():
        m["identificadores_origem"] = {k: v for k, v in ler(aliases).items() if v in ids}
    return m


def iniciar(origem, saida):
    origem, saida = Path(origem).resolve(), Path(saida).resolve()
    if saida == origem or origem in saida.parents:
        raise ValueError("experimento deve ficar fora da origem")
    _, ids_origem = scans(origem)
    aliases_origem = {}
    arquivo_aliases = origem / "identificadores.json"
    if arquivo_aliases.is_file():
        aliases_origem = ler(arquivo_aliases)
        if not isinstance(aliases_origem, dict):
            raise ValueError("identificadores.json deve mapear IDs de linhagem para IDs de scan")
    saida.mkdir(parents=True, exist_ok=False)
    rev = saida / "r000"
    rev.mkdir()
    preparar(origem, rev / "projeto")
    # `preparar` grava aliases apenas da entrada imediata para os IDs novos.
    # Componha-os com a linhagem anterior antes de congelar o manifesto.
    ids_novos = ler(rev / "projeto/identificadores.json")
    linhagem = {k: ids_novos[v] for k, v in aliases_origem.items() if v in ids_novos}
    linhagem.update(ids_novos)
    if set(ids_novos) != set(ids_origem):
        raise ValueError("identificadores da copia nao correspondem aos scans da origem")
    salvar_json(rev / "projeto/identificadores.json", linhagem)
    # Fusoes anteriores nao sao referencias da nova rodada; remover so da COPIA.
    # Preserva a copia congelada integral: derivados ficam fora dela e nunca sao
    # utilizados por este fluxo sem um relatorio Revo correspondente.
    m = manifesto(rev / "projeto")
    salvar_json(rev / "manifesto.json", m)
    salvar_json(rev / "estado.json", dict(estado="revisao_pendente", nativo={}, revo={}, comparacao={}, revisoes_3d={}))
    salvar_json(saida / "ciclo.json", dict(ativa="r000", origem=str(origem)))
    return m


def estado(rev):
    return ler(rev / "estado.json")


class ResultadoParcial(RuntimeError):
    """Uma etapa terminou parcialmente; nenhuma revisao foi publicada."""


def triagem(root, saida=None):
    """Analisa a copia congelada; salva o perfil fora do projeto e nao muda a revisao."""
    from triagem import triar

    root, _, rev, m = carregar(root)
    destino = Path(saida).resolve() if saida else rev / f"triagem_{uuid.uuid4().hex[:8]}.json"
    projeto = rev / "projeto"
    origem = Path(ler(root / "ciclo.json")["origem"]).resolve()
    protegidos = [projeto, origem, *root.glob("r*/projeto")]
    if any(destino == pasta or pasta in destino.parents for pasta in protegidos):
        raise ValueError("perfil de triagem deve ficar fora do projeto congelado e da origem")
    if destino.exists():
        raise FileExistsError(destino)
    perfil = triar(projeto, m["scans"])
    # Analise nao pode escrever nos dados congelados nem trocar o checkpoint.
    _, _, atual, atual_m = carregar(root)
    if atual != rev or atual_m["id"] != m["id"]:
        raise ValueError("revisao ativa mudou durante a triagem")
    perfil["ciclo"] = dict(manifesto=m["id"], revisao=rev.name, projeto=str(projeto))
    destino.parent.mkdir(parents=True, exist_ok=True)
    salvar_json(destino, perfil)
    return dict(arquivo=str(destino), manifesto=m["id"], perfil=perfil)


def _preparar_revisao(root, rev):
    nome = f"r{int(rev.name[1:]) + 1:03d}"
    if (root / nome).exists():
        raise FileExistsError(root / nome)
    stage = root / f"{nome}_preparando_{uuid.uuid4().hex[:8]}"
    stage.mkdir()
    return nome, stage


def _calibracoes(rev, m):
    calib = dict(m.get("calibracao_fixa", {}))
    st = estado(rev)
    for g in m["scans"]:
        if g not in calib and g in st.get("nativo", {}):
            calib[g] = ler(rev / st["nativo"][g] / "fusao.json")["entrada"]["calibracao"]
    return calib


def _metadados_revisao(novo, anterior, rev, mapa, calibracoes, poses=None):
    # Compoe aliases de todas as revisoes: poses originais continuam identificaveis.
    origem = dict(anterior.get("identificadores_origem", {}))
    ids = rev / "projeto/identificadores.json"
    if ids.is_file():
        origem.update(ler(ids))
    origem.update({g: g for g in anterior["scans"]})
    novo["identificadores_origem"] = {k: mapa[v] for k, v in origem.items() if v in mapa}
    novo["calibracao_fixa"] = {mapa[g]: v for g, v in calibracoes.items() if g in mapa}
    alinhadas = anterior.get("poses_alinhamento", {}) if poses is None else poses
    if alinhadas:
        novo["poses_alinhamento"] = {mapa[g]: T for g, T in alinhadas.items() if g in mapa}
        novo["referencial_alinhamento"] = anterior.get(
            "referencial_alinhamento", "poses explicitas scan -> referencial comum; mm"
        )
    if anterior.get("qualidade_3d"):
        novo["qualidade_3d"] = anterior["qualidade_3d"]


def _publicar_revisao(root, top, rev, m, nome, stage, calibracoes, poses=None, qualidade=None):
    from auditar_projeto import auditar

    auditoria = auditar(rev / "projeto", stage / "projeto")
    # Falha da auditoria ou alteracao concorrente nunca substitui a ativa.
    _, _, ativa, atual = carregar(root)
    if ativa != rev or atual["id"] != m["id"]:
        raise ValueError("revisao ativa mudou durante o processamento")
    arq, ids = scans(stage / "projeto")
    if set(ids) != set(m["scans"]):
        raise ValueError("processamento mudou o conjunto de scans")
    mapa = reidentificar(arq)
    novo = manifesto(stage / "projeto", m["id"])
    _metadados_revisao(novo, m, rev, mapa, calibracoes, poses)
    salvar_json(stage / "auditoria.json", dict(resultados=auditoria, mapa_ids=mapa))
    novo_estado = dict(
        estado="revisao_pendente",
        nativo={},
        revo={},
        comparacao={},
        revisoes_3d={},
        invalidado=["fusao", "malha", "comparacao", "merge"],
        anterior=rev.name,
    )
    resumo_qualidade = qualidade or novo.get("qualidade_3d")
    if resumo_qualidade:
        novo["qualidade_3d"] = resumo_qualidade
        novo_estado["qualidade_3d"] = resumo_qualidade
    salvar_json(stage / "estado.json", novo_estado)
    salvar_json(stage / "manifesto.json", novo)
    stage.rename(root / nome)
    top["ativa"] = nome
    salvar_json(root / "ciclo.json", top)
    return novo, mapa


def _rodar_etapa(cmd, stage):
    # O ciclo de producao nunca herda o opt-in de depuracao na CPU.
    env = dict(os.environ)
    env.pop("REVO_PERMITIR_CPU", None)
    run = subprocess.run(cmd, check=False, env=env)
    if run.returncode:
        salvar_json(stage / "falha.json", dict(comando=cmd, codigo=run.returncode, ativa_preservada=True))
        if run.returncode == 2:
            raise ResultadoParcial("etapa parcial: revisao ativa preservada; conferir pasta de preparo")
        raise subprocess.CalledProcessError(run.returncode, cmd)


def fundir(root, motor=None, voxel=0.08):
    root, _, rev, m = carregar(root)
    st = estado(rev)
    st["estado"] = "revisao_pendente"
    # Marca invalidacao antes do trabalho; uma falha nunca deixa aprovacao antiga.
    st["revisoes_3d"] = {}
    salvar_json(rev / "estado.json", st)
    for g in m["scans"]:
        out = rev / f"nativo_{g}_{uuid.uuid4().hex[:8]}"
        executar(rev / "projeto/data" / g, out, motor, voxel, calibracao_fixa=m.get("calibracao_fixa", {}).get(g))
        st["nativo"][g] = out.name
        st["comparacao"].pop(g, None)
        salvar_json(rev / "estado.json", st)
    return st


def limpar(root, configuracao=None):
    """Triagem e limpeza temporal em revisao auditada; JSON explicito prevalece."""
    root, top, rev, m = carregar(root)
    explicita = ler(configuracao) if configuracao else {}
    permitidos = {
        "pixel_tol",
        "pixel_comp_min",
        "temporal_passo",
        "temporal_min",
        "ext_max_mm",
        "min_mantido",
        "so_peca",
        "periferia_mm",
        "periferia_banda",
        "periferia_alargue",
        "jobs",
    }
    if not isinstance(explicita, dict) or set(explicita) - permitidos:
        raise ValueError("opcoes de limpeza desconhecidas")
    analise = triagem(root)
    recomendada = analise["perfil"].get("recomendacao", {}).get("limpar_raw", {})
    cfg = {k: v for k, v in recomendada.items() if k in permitidos}
    cfg.update(explicita)
    nome, stage = _preparar_revisao(root, rev)
    salvar_json(stage / "triagem.json", analise)
    salvar_json(stage / "configuracao_limpeza.json", dict(recomendada=recomendada, explicita=explicita, efetiva=cfg))
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("limpar_raw.py")),
        str(rev / "projeto"),
        "--saida",
        str(stage / "projeto"),
        "--sem-ply-antigos",
        "--manter-ids",
        "--modo-limpeza",
        "precisao",
        "--scans",
        *m["scans"],
    ]
    for k, v in cfg.items():
        if isinstance(v, bool):
            if v:
                cmd += ["--" + k.replace("_", "-")]
        else:
            cmd += ["--" + k.replace("_", "-"), str(v)]
    _rodar_etapa(cmd, stage)
    rel = ler(stage / "projeto/relatorio_limpeza.json")
    pendencias = {g: rel[g] for g in m["scans"] if rel.get(g, {}).get("status") != "limpo"}
    if pendencias:
        salvar_json(stage / "falha.json", pendencias)
        raise ResultadoParcial("limpeza parcial/revisar: entrada ativa preservada; conferir falha.json no preparo")
    _publicar_revisao(root, top, rev, m, nome, stage, _calibracoes(rev, m))
    return dict(revisao=nome, limpeza=rel)


def _poses_ativas(caminho, rev, m):
    def sem_duplicados(pares):
        objeto = {}
        for chave, valor in pares:
            if chave in objeto:
                raise ValueError(f"chave repetida no JSON de poses: {chave}")
            objeto[chave] = valor
        return objeto

    doc = json.loads(Path(caminho).read_text(encoding="utf-8"), object_pairs_hook=sem_duplicados)
    if not isinstance(doc, dict) or (doc.get("manifesto") is not None and doc["manifesto"] != m["id"]):
        raise ValueError("poses invalidas ou manifesto de outra revisao")
    poses = doc.get("poses", doc.get("poses_usadas", doc.get("poses_iniciais", doc)))
    if not isinstance(poses, dict):
        raise ValueError("poses devem ser um objeto {scan: matriz 4x4}")
    aliases = dict(m.get("identificadores_origem", {}))
    ids = rev / "projeto/identificadores.json"
    if ids.is_file():
        aliases.update(ler(ids))
    aliases.update({g: g for g in m["scans"]})
    atuais = {}
    for origem, valor in poses.items():
        g = aliases.get(origem)
        if g not in m["scans"]:
            raise ValueError(f"pose de scan desconhecido: {origem}")
        if g in atuais:
            raise ValueError(f"poses duplicadas por aliases do mesmo scan: {origem}")
        T = np.asarray(valor, float)
        if (
            T.shape != (4, 4)
            or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or abs(np.linalg.det(T[:3, :3])) < 1e-12
        ):
            raise ValueError(f"pose deve ser 4x4 finita, homogenea e inversivel: {origem}")
        atuais[g] = T.tolist()
    if len(atuais) < 2:
        raise ValueError("consenso exige pelo menos dois scans com poses explicitas")
    return atuais


def consenso(root, poses):
    """Cria revisao pelo espaco livre entre scans alinhados, sem aprovar derivados."""
    root, top, rev, m = carregar(root)
    origem_sha256 = sha256(poses)
    alinhadas = _poses_ativas(poses, rev, m)
    if sha256(poses) != origem_sha256:
        raise ValueError("arquivo de poses mudou durante a leitura")
    calib = _calibracoes(rev, m)
    nome, stage = _preparar_revisao(root, rev)
    salvar_json(stage / "poses_entrada.json", alinhadas)
    poses_congeladas_sha256 = sha256(stage / "poses_entrada.json")
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("consenso_visibilidade.py")),
        str(rev / "projeto"),
        "--poses",
        str(stage / "poses_entrada.json"),
        "--saida",
        str(stage / "projeto"),
        "--scans",
        *alinhadas,
    ]
    fixas = {g: calib[g] for g in alinhadas if g in calib}
    if fixas:
        salvar_json(stage / "calibracoes_entrada.json", fixas)
        cmd += ["--calibracoes", str(stage / "calibracoes_entrada.json")]
    _rodar_etapa(cmd, stage)
    if sha256(stage / "poses_entrada.json") != poses_congeladas_sha256:
        raise ValueError("poses de processamento foram alteradas durante o consenso")
    rel = ler(stage / "projeto/consenso_visibilidade.json")
    if set(rel.get("scans", {})) != set(alinhadas):
        raise ValueError("relatorio de consenso nao cobre os scans solicitados")
    pendencias = {
        g: r
        for g, r in rel["scans"].items()
        if r.get("integridade") != "ok"
        or r.get("pixels_depois", 0) <= 0
        or r.get("pixels_depois", 0) > r.get("pixels_antes", 0)
    }
    if rel.get("estado") != "revisao_pendente" or pendencias:
        salvar_json(stage / "falha.json", dict(estado=rel.get("estado"), scans=pendencias, ativa_preservada=True))
        raise ResultadoParcial("consenso parcial/invalido: revisao ativa preservada")
    from consenso_visibilidade import validar_calibracao_fixa

    for g, dados in rel["scans"].items():
        registrada = validar_calibracao_fixa(dados.get("calibracao"), ())
        if not registrada:
            raise ValueError(f"consenso sem calibracao registrada: {g}")
        if g in calib:
            fixa = validar_calibracao_fixa(calib[g], ())
            if not set(fixa).issubset(registrada) or any(
                not np.array_equal(fixa[modo]["K"], registrada[modo]["K"]) for modo in fixa
            ):
                raise ValueError(f"consenso nao respeitou calibracao fixa: {g}")
        else:
            calib[g] = {str(modo): valor for modo, valor in registrada.items()}
    salvar_json(
        stage / "consenso.json",
        dict(relatorio=rel, poses_origem=str(Path(poses).resolve()), poses_sha256=origem_sha256),
    )
    novo, mapa = _publicar_revisao(root, top, rev, m, nome, stage, calib, alinhadas)
    return dict(revisao=nome, manifesto=novo["id"], mapa_ids=mapa, consenso=rel)


def qualidade(root, poses):
    """Aplica limpeza conservadora 3D somente em cópia de frames, com revisão pendente."""
    root, top, rev, m = carregar(root)
    alinhadas = _poses_ativas(poses, rev, m)
    if set(alinhadas) != set(m["scans"]):
        raise ValueError("qualidade 3D exige poses explícitas para todos os scans da revisão")
    calib = _calibracoes(rev, m)
    nome, stage = _preparar_revisao(root, rev)
    pose_doc = stage / "poses_qualidade.json"
    salvar_json(pose_doc, alinhadas)
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("limpar_qualidade_3d.py")),
        str(rev / "projeto"),
        "--poses",
        str(pose_doc),
        "--saida",
        str(stage / "projeto"),
    ]
    fixas = {g: calib[g] for g in alinhadas if g in calib}
    if fixas:
        salvar_json(stage / "calibracoes_qualidade.json", fixas)
        cmd += ["--calibracoes", str(stage / "calibracoes_qualidade.json")]
    _rodar_etapa(cmd, stage)
    rel_path = stage / "qualidade_3d.json"
    rel = ler(rel_path)
    pose_hash = hashlib.sha256(
        json.dumps(alinhadas, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=float).encode()
    ).hexdigest()
    if rel.get("poses_sha256") != pose_hash:
        raise ValueError("relatorio de qualidade nao corresponde as poses solicitadas")
    if set(rel.get("scans", {})) != set(m["scans"]):
        raise ValueError("relatorio de qualidade nao cobre todos os scans")
    for g in m["scans"]:
        esperado = sha256(rev / "projeto/data" / g / "cache/frames.dataset")
        if rel.get("hashes", {}).get(g, {}).get("dataset_sha256") != esperado:
            raise ValueError(f"medicao de qualidade obsoleta para {g}")
    pendencias = rel.get("pendencias", [])
    resumo = dict(
        estado=rel.get("estado", "revisao_pendente"),
        pendencias=pendencias,
        arquivo="qualidade_3d.json",
        sha256=sha256(rel_path),
        revisao_origem=nome,
    )
    novo, mapa = _publicar_revisao(root, top, rev, m, nome, stage, calib, alinhadas, resumo)
    return dict(revisao=nome, manifesto=novo["id"], mapa_ids=mapa, qualidade=rel)


def executar_revo(root):
    """Nova importacao em grupo proprio pelo pipeline existente; sem furos/malha."""
    root, _, rev, m = carregar(root)
    st = estado(rev)
    if set(st.get("nativo", {})) != set(m["scans"]):
        raise ValueError("fundir nativamente todos os scans antes da importacao Revo")
    from comparar_fusao_cuda import validar_ply

    seeds = {}
    for g in m["scans"]:
        pasta_nativa = rev / st["nativo"][g]
        nuvem = pasta_nativa / "nuvem.ply"
        fusao = ler(pasta_nativa / "fusao.json")
        if fusao.get("entrada", {}).get("dataset_sha256") != sha256(rev / "projeto/data" / g / "cache/frames.dataset"):
            raise ValueError(f"fusão nativa obsoleta para {g}")
        if sha256(nuvem) != fusao.get("nuvem_sha256"):
            raise ValueError(f"nuvem nativa alterada ou incompleta para {g}")
        seeds[g] = dict(arquivo=nuvem, sha256=sha256(nuvem), pontos=validar_ply(nuvem))
    st["estado"] = "revisao_pendente"
    st["revisoes_3d"] = {}
    salvar_json(rev / "estado.json", st)
    out = rev / f"revo_{uuid.uuid4().hex[:8]}"
    out.mkdir()
    # Uma tentativa anterior pode ter importado estes IDs antes de falhar na
    # exportacao. Cada ensaio recebe IDs novos, sem mudar nenhum frame.
    importado = preparar(rev / "projeto", out / "entrada")
    mapa = ler(out / "entrada/identificadores.json")
    for g in m["scans"]:
        original_frames = sha256(rev / "projeto/data" / g / "cache/frames.dataset")
        clone = out / "entrada/data" / mapa[g]
        if sha256(clone / "cache/frames.dataset") != original_frames:
            raise ValueError("clone de referencia divergiu dos frames congelados")
        seed = seeds[g]
        shutil.copyfile(seed["arquivo"], clone / "fuse.ply")
        if sha256(clone / "fuse.ply") != seed["sha256"] or validar_ply(clone / "fuse.ply") != seed["pontos"]:
            raise ValueError(f"seed nativo divergiu ao preparar Revo para {g}")
        prop = clone / "property.rvproj"
        data = ler(prop)
        data.update(fuse_file="fuse.ply", point_count=seed["pontos"], mesh_file="", face_count=0, vertex_count=0)
        data["can_fuse_optimize"] = False
        data.setdefault("scan_param", {})["scan_mode"] = 1
        salvar_json(prop, data)
    salvar_json(
        out / "vinculo.json",
        dict(
            manifesto=m["id"],
            mapa=mapa,
            frames_dataset_sha256={g: sha256(rev / "projeto/data" / g / "cache/frames.dataset") for g in m["scans"]},
            seeds={g: {k: v for k, v in x.items() if k != "arquivo"} for g, x in seeds.items()},
            arquivos=inventario(out / "entrada"),
        ),
    )
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("revo_pipeline.py")),
        str(importado),
        "--saida",
        str(out),
        "--ponto",
        "0.1",
        "--sem-exportar",
        "--scans",
        ",".join(mapa[g] for g in m["scans"]),
        "--entregaveis",
        "nuvem",
        "--grupo",
        f"PRECISAO_{rev.name}_{uuid.uuid4().hex[:8]}",
    ]
    run = subprocess.run(cmd, check=False)
    if run.returncode:
        raise RuntimeError(f"Revo falhou ({run.returncode}); referencia nao aprovada")
    for g in m["scans"]:
        if sha256(rev / "projeto/data" / g / "cache/frames.dataset") != sha256(
            out / "entrada/data" / mapa[g] / "cache/frames.dataset"
        ):
            raise ValueError(f"Revo alterou os frames congelados de {g}")
    # A nuvem nativa ja esta salva pelo Revo; copiar evita depender do dialogo
    # Export Model. Aceita somente Fused gerado na importacao exclusiva.
    return registrar_revo(root, out / "revo_pipeline.json")


def copiar_referencia(src, dest, timeout=30):
    """O log de fusao termina antes da gravacao assincrona do PLY no Revo."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    inicio = time.monotonic()
    while time.monotonic() - inicio < timeout:
        temp = dest.with_suffix(".gravando")
        try:
            validar_ply(src)
            primeiro = sha256(src)
            time.sleep(0.5)
            if sha256(src) != primeiro:
                continue
            shutil.copyfile(src, temp)
            validar_ply(temp)
            if sha256(temp) != primeiro:
                continue
            os.replace(temp, dest)
            return
        except (OSError, ValueError):
            time.sleep(0.5)
    raise ValueError("Revo nao terminou de gravar uma nuvem integra")


def registrar_revo(root, relatorio):
    root, _, rev, m = carregar(root)
    relatorio = Path(relatorio).resolve()
    rel = ler(relatorio)
    mapa = {g: g for g in m["scans"]}
    if Path(rel["revo"]).resolve() != (rev / "projeto" / m["projeto"]).resolve():
        link = ler(relatorio.parent / "vinculo.json")
        if link["manifesto"] != m["id"] or Path(rel["revo"]).resolve().parent != relatorio.parent / "entrada":
            raise ValueError("relatorio Revo nao pertence a entrada congelada ativa")
        if inventario(relatorio.parent / "entrada") != link["arquivos"]:
            raise ValueError("clone importado foi alterado")
        mapa = {v: k for k, v in link["mapa"].items()}
        for g in m["scans"]:
            clone = link["mapa"][g]
            if sha256(relatorio.parent / "entrada/data" / clone / "cache/frames.dataset") != sha256(
                rev / "projeto/data" / g / "cache/frames.dataset"
            ):
                raise ValueError("frames Revo e nativo diferentes")
    if rel.get("parametros", {}).get("so_exportar") or rel.get("parametros", {}).get("reusar"):
        raise ValueError("comparacao exige nova importacao/fusao, sem reuso de sessao antiga")
    if rel.get("parametros", {}).get("furos"):
        raise ValueError("comparacao sem fechamento de furos")
    st = estado(rev)
    for row in rel["scans"]:
        g = mapa.get(row["guid"])
        if g not in m["scans"] or row.get("erro") or row.get("divergencias") or not row.get("fusao"):
            raise ValueError(f"fusao Revo nao validada: {g}")
        if row["fusao"].get("otimizar_poses") != 0:
            raise ValueError("comparacao exige confirmacao no log de optimizePose=0")
        if rel.get("parametros", {}).get("sem_exportar"):
            fontes = list((Path(row["pasta_biblioteca"]) / "pointclouds").glob("Fused*.ply"))
            if len(fontes) != 1:
                raise ValueError("esperada uma unica fusao nova na biblioteca")
            copiar_referencia(fontes[0], relatorio.parent / row["guid"] / fontes[0].name)
        candidates = list((relatorio.parent / row["guid"]).glob("Fused*.ply"))
        if len(candidates) != 1:
            raise ValueError(f"esperada uma unica fusao exportada: {g}")
        validar_ply(candidates[0])
        st["revo"][g] = dict(
            arquivo=str(candidates[0].resolve()),
            sha256=sha256(candidates[0]),
            manifesto=m["id"],
            relatorio=str(relatorio),
            fusao=row["fusao"],
        )
    st["comparacao"] = {}
    st["revisoes_3d"] = {}
    st["estado"] = "revisao_pendente"
    salvar_json(rev / "estado.json", st)
    return st


def comparar(root):
    _, _, rev, m = carregar(root)
    st = estado(rev)
    for g in m["scans"]:
        if g not in st["nativo"] or g not in st["revo"]:
            raise ValueError(f"faltam fusoes da revisao atual: {g}")
        ref = st["revo"][g]
        if ref["manifesto"] != m["id"] or sha256(ref["arquivo"]) != ref["sha256"]:
            raise ValueError("referencia Revo alterada/desatualizada")
        nativo = rev / st["nativo"][g]
        f = ler(nativo / "fusao.json")
        if sha256(nativo / "nuvem.ply") != f["nuvem_sha256"]:
            raise ValueError("nuvem nativa alterada")
        p = np.load(nativo / "superficies.npz")["pontos"]
        metric = avaliar(p, ref["arquivo"])
        metric.update(manifesto=m["id"], nativo_sha256=f["nuvem_sha256"], revo_sha256=ref["sha256"])
        salvar_json(rev / f"comparacao_{g}.json", metric)
        st["comparacao"][g] = metric
    salvar_json(rev / "estado.json", st)
    return st["comparacao"]


def validar_regioes(doc, m):
    if doc.get("manifesto") != m["id"]:
        raise ValueError("mascara de outra revisao/referencial")
    if not doc.get("regioes"):
        raise ValueError("nenhuma regiao indicada")
    for r in doc["regioes"]:
        if r.get("scan") not in m["scans"] or r.get("classe") not in ("suporte", "residuo", "proteger"):
            raise ValueError("scan/classe de regiao invalido")
        if not r.get("motivo", "").strip():
            raise ValueError("cada regiao precisa de justificativa")
        tipo = r.get("tipo")
        if tipo == "caixa":
            lo, hi = np.asarray(r["min"], float), np.asarray(r["max"], float)
            if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or not (hi > lo).all():
                raise ValueError("caixa invalida")
        elif tipo == "esfera":
            if (
                np.asarray(r["centro"]).shape != (3,)
                or not np.isfinite(r["centro"]).all()
                or not np.isfinite(r["raio"])
                or r["raio"] <= 0
            ):
                raise ValueError("esfera invalida")
        else:
            raise ValueError("tipo de regiao desconhecido")


def mascara_regioes(p, regioes, g):
    """Mascara bool (numpy) dos pontos de `p` (numpy ou GPU) a remover pelas regioes do scan `g`; calculada na GPU."""
    m = gpu.xp()
    p = gpu.para_gpu(p, m.float64)
    apagar = m.zeros(len(p), bool)
    proteger = apagar.copy()
    for r in regioes:
        if r["scan"] != g:
            continue
        if r["tipo"] == "caixa":
            mask = m.all(
                (p >= gpu.para_gpu(np.asarray(r["min"], float))) & (p <= gpu.para_gpu(np.asarray(r["max"], float))),
                axis=1,
            )
        else:
            mask = m.linalg.norm(p - gpu.para_gpu(np.asarray(r["centro"], float)), axis=1) <= r["raio"]
        if r["classe"] == "proteger":
            proteger |= mask
        else:
            apagar |= mask
    if bool((apagar & proteger).any()):
        raise ValueError("regioes de remocao e protecao se sobrepoem em pontos medidos; refinar selecao")
    return gpu.para_cpu(apagar)


def corrigir_scan(src, dst, regioes, g, calibracao):
    """Reescreve somente os pixels rejeitados; FrameInfo e z retidos identicos."""
    with tempfile.TemporaryDirectory() as tmp:
        scan = RevoScan(str(src), tmp)
        try:
            scan.K = {int(k): tuple(v["K"]) for k, v in calibracao.items()}
            stats = dict(antes=0, depois=0, quadros=0, removidos=0, vazios=0)
            # Modifica apenas a COPIA temporaria; leitura integral originais intacta.
            for k, i in enumerate(scan.ids):
                if not scan.tem_prof[k]:
                    continue
                w, h, mask, z, tail = scan.profundidade(i)
                p, pix, _, _ = pontos_gpu(scan, k)
                remove = mascara_regioes(p, regioes, g)
                ok = ~remove
                stats["antes"] += len(z)
                stats["depois"] += int(ok.sum())
                stats["quadros"] += 1
                if not remove.any():
                    continue
                if not ok.any():
                    scan.c.execute("delete from FrameRGBD where id=?", (i,))
                    stats["vazios"] += 1
                else:
                    nova = np.zeros(w * h, bool)
                    nova[pix[ok]] = True
                    blob = (
                        struct.pack("<III", w, h, int(ok.sum()))
                        + np.packbits(nova, bitorder="little").tobytes()
                        + tail
                        + z[ok].astype("<f4").tobytes()
                    )
                    scan.c.execute("update FrameRGBD set data=? where id=?", (blob, i))
            if stats["depois"] == 0:
                raise ValueError("correcao apagaria todo o scan")
            scan.c.commit()
            scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
            if scan.c.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise ValueError("integridade SQLite falhou")
            scan.c.close()
            with open(scan.db, "r+b") as f:
                f.write(scan.sig)
            shutil.copyfile(scan.db, dst / "cache/frames.dataset")
            hdr, prev = ler_ply(src / "raw_preview.ply")
            keep = ~mascara_regioes(prev[:, :3], regioes, g)
            if not keep.any():
                raise ValueError("preview ficaria vazio")
            gravar_ply(dst / "raw_preview.ply", hdr, prev[keep])
            stats["removidos"] = stats["antes"] - stats["depois"]
            # Artefatos visuais copiados jamais podem mostrar fusao velha como atual.
            for nome in ("fuse.ply", "fuse_mesh.ply", "mark.ply"):
                arq = dst / nome
                if arq.is_file():
                    arq.unlink()
            prop = dst / "property.rvproj"
            data = ler(prop)
            data.update(fuse_file="", mesh_file="", point_count=0, face_count=0, vertex_count=0)
            salvar_json(prop, data)
            return stats
        finally:
            scan.c.close()


def corrigir(root, arquivo):
    root, top, rev, m = carregar(root)
    doc = ler(arquivo)
    validar_regioes(doc, m)
    st = estado(rev)
    if any(g not in st["nativo"] for g in m["scans"]):
        raise ValueError("fundir antes de selecionar correcoes no referencial validado")
    nome = f"r{int(rev.name[1:]) + 1:03d}"
    if (root / nome).exists():
        raise FileExistsError(root / nome)
    nova = root / f"{nome}_preparando_{uuid.uuid4().hex[:8]}"
    nova.mkdir(exist_ok=False)
    shutil.copytree(rev / "projeto", nova / "projeto")
    stats = {}
    calibracoes = {}
    for g in m["scans"]:
        calib = ler(rev / st["nativo"][g] / "fusao.json")["entrada"]["calibracao"]
        calibracoes[g] = calib
        stats[g] = corrigir_scan(rev / "projeto/data" / g, nova / "projeto/data" / g, doc["regioes"], g, calib)
    if not sum(v["removidos"] for v in stats.values()):
        raise ValueError("mascaras nao removeram observacoes; revisao ativa preservada")
    # Audita ANTES de remapear IDs para manter correspondencia direta dos frames.
    from auditar_projeto import auditar

    audit = auditar(rev / "projeto", nova / "projeto")
    arq, _ = scans(nova / "projeto")
    # Todos os IDs mudam: evita reuso de fusao/merge em cache na biblioteca Revo.
    mapa = reidentificar(arq)
    novo = manifesto(nova / "projeto", m["id"])
    _metadados_revisao(novo, m, rev, mapa, calibracoes)
    salvar_json(nova / "manifesto.json", novo)
    salvar_json(nova / "correcao.json", dict(selecao=doc, contagens=stats, auditoria=audit, mapa_ids=mapa))
    salvar_json(
        nova / "estado.json",
        dict(
            estado="revisao_pendente",
            nativo={},
            revo={},
            comparacao={},
            revisoes_3d={},
            invalidado=["fusao", "malha", "comparacao", "merge"],
            anterior=rev.name,
        ),
    )
    nova.rename(root / nome)
    top["ativa"] = nome
    salvar_json(root / "ciclo.json", top)
    return stats


def aprovar(root, parecer):
    _, _, rev, m = carregar(root)
    st = estado(rev)
    qualidade_atual = m.get("qualidade_3d") or st.get("qualidade_3d")
    if qualidade_atual:
        pasta_relatorio = Path(root) / qualidade_atual.get("revisao_origem", rev.name)
        arq_relatorio = pasta_relatorio / qualidade_atual.get("arquivo", "qualidade_3d.json")
        if not arq_relatorio.is_file() or sha256(arq_relatorio) != qualidade_atual.get("sha256"):
            raise ValueError("relatorio de qualidade 3D ausente ou alterado; aprovacao bloqueada")
        rel_qualidade = ler(arq_relatorio)
        if rel_qualidade.get("pendencias") != qualidade_atual.get("pendencias"):
            raise ValueError("pendencias de qualidade 3D foram alteradas; aprovacao bloqueada")
        if qualidade_atual.get("pendencias"):
            raise ValueError("pendencias de qualidade 3D precisam de revisao antes do merge")
    doc = ler(parecer)
    if doc.get("manifesto") != m["id"] or set(doc.get("scans", {})) != set(m["scans"]):
        raise ValueError("parecer incompleto ou de outra revisao")
    for g in m["scans"]:
        p = doc["scans"][g]
        comp = st["comparacao"].get(g)
        if (
            not comp
            or p.get("pendencias") != []
            or not all(
                p.get(k) is True
                for k in ("inspecao_3d_completa", "suportes_excluidos", "detalhes_preservados", "incertos_revisados")
            )
        ):
            raise ValueError(f"pendencias ou inspecao incompleta: {g}")
        nativo = rev / st["nativo"][g]
        if (
            comp["manifesto"] != m["id"]
            or comp["nativo_sha256"] != sha256(nativo / "nuvem.ply")
            or comp["revo_sha256"] != sha256(st["revo"][g]["arquivo"])
        ):
            raise ValueError("artefatos mudaram apos comparacao")
        if not st["revisoes_3d"].get(g) or not any(r["tipo"] == "consulta_integral" for r in st["revisoes_3d"][g]):
            raise ValueError("falta consulta integral 3D registrada para esse scan")
        if not str(p.get("evidencia", "")).strip() or len(str(p.get("evidencia", ""))) < 30:
            raise ValueError("parecer precisa descrever evidencia da revisao 3D")
        if comp["revo_para_nativo"]["acima_03mm_pct"] > 1:
            raise ValueError("mais de 1% da referencia sem cobertura nativa a 0,3 mm: revisar perda de detalhe")
    st["estado"] = "liberado_para_merge"
    st["parecer"] = doc
    salvar_json(rev / "estado.json", st)
    return st


def exportar_merge(root, saida):
    """Exporta projeto e nuvens aprovados; nao executa alinhamento nao validado."""
    root, _, rev, m = carregar(root)
    st = estado(rev)
    if st.get("estado") != "liberado_para_merge":
        raise ValueError("merge bloqueado: concluir comparacao, inspecao e parecer da revisao atual")
    # Revalida hashes/criterios antes de consumir o estado de aprovacao.
    parecer = rev / "parecer_exportacao.json"
    salvar_json(parecer, st["parecer"])
    aprovar(root, parecer)
    out = Path(saida).resolve()
    if out == root or root in out.parents:
        raise ValueError("entrega deve ficar fora do experimento")
    out.mkdir(parents=True, exist_ok=False)
    projeto = ler(rev / "projeto" / m["projeto"])
    projeto["nodes"] = [n for n in projeto["nodes"] if n["guid"] in m["scans"]]
    salvar_json(out / m["projeto"], projeto)
    for g in m["scans"]:
        shutil.copytree(
            rev / "projeto/data" / g,
            out / "data" / g,
            ignore=shutil.ignore_patterns("fuse_mesh.ply", "mark.ply", "global_register_pose.pose"),
        )
        cloud = rev / st["nativo"][g] / "nuvem.ply"
        shutil.copyfile(cloud, out / "data" / g / "fuse.ply")
        meta = ler(out / "data" / g / "property.rvproj")
        meta.update(
            fuse_file="fuse.ply",
            point_count=ler(rev / st["nativo"][g] / "fusao.json")["pontos"],
            mesh_file="",
            face_count=0,
            vertex_count=0,
            can_fuse_optimize=False,
        )
        meta.setdefault("scan_param", {})["scan_mode"] = 1
        salvar_json(out / "data" / g / "property.rvproj", meta)
    arq, _ = scans(out)
    reidentificar(arq)
    salvar_json(
        out / "LIBERACAO_MERGE.json", dict(manifesto=m["id"], parecer=st["parecer"], motor="nativo", origem=str(rev))
    )
    return dict(entrega=str(out), estado="liberado_para_merge")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="acao", required=True)
    p = sp.add_parser("iniciar")
    p.add_argument("projeto")
    p.add_argument("--saida", required=True)
    p = sp.add_parser("fundir")
    p.add_argument("ciclo")
    p.add_argument("--motor", choices=["cuda"], default="cuda", help=argparse.SUPPRESS)
    p.add_argument("--voxel", type=float, default=0.08)
    p = sp.add_parser("limpar")
    p.add_argument("ciclo")
    p.add_argument("--configuracao")
    p = sp.add_parser("triagem")
    p.add_argument("ciclo")
    p.add_argument("--saida", help="JSON novo fora do projeto; padrao: arquivo na revisao ativa")
    p = sp.add_parser("consenso")
    p.add_argument("ciclo")
    p.add_argument("--poses", required=True, help="JSON de poses 4x4, por ID original ou da revisao ativa")
    p = sp.add_parser("qualidade")
    p.add_argument("ciclo")
    p.add_argument("--poses", required=True, help="JSON de poses 4x4, por ID original ou da revisao ativa")
    p = sp.add_parser("revo")
    p.add_argument("ciclo")
    p = sp.add_parser("registrar-revo")
    p.add_argument("ciclo")
    p.add_argument("--relatorio", required=True)
    p = sp.add_parser("comparar")
    p.add_argument("ciclo")
    p = sp.add_parser("corrigir")
    p.add_argument("ciclo")
    p.add_argument("--regioes", required=True)
    p = sp.add_parser("aprovar")
    p.add_argument("ciclo")
    p.add_argument("--parecer", required=True)
    p = sp.add_parser("exportar-merge")
    p.add_argument("ciclo")
    p.add_argument("--saida", required=True)
    p = sp.add_parser("inspecionar")
    p.add_argument("ciclo")
    p.add_argument("--scan", required=True)
    p.add_argument("--saida", required=True)
    p = sp.add_parser("consultar")
    p.add_argument("ciclo")
    p.add_argument("--scan", required=True)
    p.add_argument("--caixa", nargs=6, type=float, required=True)
    p.add_argument("--saida", required=True)
    p = sp.add_parser("status")
    p.add_argument("ciclo")
    a = ap.parse_args()
    lock = None
    if a.acao != "iniciar" and os.name == "nt":
        import msvcrt

        lock = open(Path(a.ciclo) / ".execucao.lock", "a+b")
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock.close()
            ap.error("outra etapa esta executando neste ciclo; aguarde o checkpoint")
    if a.acao == "iniciar":
        r = iniciar(a.projeto, a.saida)
    elif a.acao == "fundir":
        r = fundir(a.ciclo, None, a.voxel)
    elif a.acao in ("limpar", "consenso", "qualidade"):
        try:
            if a.acao == "limpar":
                r = limpar(a.ciclo, a.configuracao)
            elif a.acao == "consenso":
                r = consenso(a.ciclo, a.poses)
            else:
                r = qualidade(a.ciclo, a.poses)
        except ResultadoParcial as exc:
            print(json.dumps(dict(estado="parcial", ativa_preservada=True, motivo=str(exc)), ensure_ascii=False))
            if lock is not None:
                lock.close()
            raise SystemExit(2) from exc
    elif a.acao == "triagem":
        r = triagem(a.ciclo, a.saida)
    elif a.acao == "revo":
        r = executar_revo(a.ciclo)
    elif a.acao == "registrar-revo":
        r = registrar_revo(a.ciclo, a.relatorio)
    elif a.acao == "comparar":
        r = comparar(a.ciclo)
    elif a.acao == "corrigir":
        r = corrigir(a.ciclo, a.regioes)
    elif a.acao == "aprovar":
        r = aprovar(a.ciclo, a.parecer)
    elif a.acao == "exportar-merge":
        r = exportar_merge(a.ciclo, a.saida)
    elif a.acao == "status":
        _, _, rev, _ = carregar(a.ciclo)
        r = estado(rev)
    else:
        from inspecao_fusao import consultar, gerar_html

        r = (
            gerar_html(a.ciclo, a.scan, a.saida)
            if a.acao == "inspecionar"
            else consultar(a.ciclo, a.scan, a.caixa, a.saida)
        )
    print(json.dumps(r, indent=2, ensure_ascii=False))
    if lock is not None:
        lock.close()


if __name__ == "__main__":
    main()
