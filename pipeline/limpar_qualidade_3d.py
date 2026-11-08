"""Seleciona observacoes redundantes de menor qualidade e grava copia revisavel dos frames.

    python pipeline/limpar_qualidade_3d.py PROJ --poses poses.json --saida COPIA_NOVA

As poses sao scan->referencial comum, em mm. So pixels comprovadamente cobertos por uma observacao
boa de outra vista podem sair. Origem, poses, cabecalhos dos quadros e pixels retidos sao preservados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
from pathlib import Path

import gpu
import numpy as np
from comum import sha256
from revo_projeto import RevoScan, gravar_ply


def _hash_objeto(obj) -> str:
    bruto = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=float).encode()
    return hashlib.sha256(bruto).hexdigest()


def _transformar(pontos, T):
    xp = gpu.xp()
    return pontos @ gpu.para_gpu(T[:3, :3].T, xp.float64) + gpu.para_gpu(T[:3, 3], xp.float64)


def _ler_poses(path, ids):
    doc = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_sem_duplicados)
    poses = doc.get("poses", doc.get("poses_usadas", doc.get("poses_iniciais", doc))) if isinstance(doc, dict) else None
    if not isinstance(poses, dict) or set(poses) != set(ids):
        raise ValueError("poses devem cobrir exatamente os scans selecionados")
    out = {}
    for guid, valor in poses.items():
        T = np.asarray(valor, dtype=np.float64)
        if (
            T.shape != (4, 4)
            or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=2e-3, rtol=0)
            or np.linalg.det(T[:3, :3]) <= 0
        ):
            raise ValueError(f"pose invalida ou nao rigida: {guid}")
        out[guid] = T
    return out


def _sem_duplicados(pares):
    d = {}
    for k, v in pares:
        if k in d:
            raise ValueError(f"chave repetida no JSON de poses: {k}")
        d[k] = v
    return d


def _referencias_por_representante(preparado, selecao, guid):
    """Superficie e observacao real de cada referencia, apenas para fontes que permanecem."""
    xp = gpu.xp()
    n = len(preparado["medicoes"][guid]["pontos"])
    pos = xp.full((n, 3), xp.nan, dtype=xp.float64)
    raw_pos = xp.full((n, 3), xp.nan, dtype=xp.float64)
    normal = xp.full((n, 3), xp.nan, dtype=xp.float64)
    noise = xp.full(n, xp.inf, dtype=xp.float64)
    origem = preparado["origem"]
    candidatas = gpu.para_cpu(xp.flatnonzero(selecao["remover"][guid])).tolist()
    grupos = {}
    for i in candidatas:
        ref_guid = selecao["referencia_scan"][guid][i]
        if ref_guid is None:
            continue
        j = int(gpu.para_cpu(selecao["referencia_indice"][guid][i]))
        grupos.setdefault(ref_guid, []).append((i, j))
    for ref_guid, pares in grupos.items():
        rep_ids = np.asarray([p[0] for p in pares], dtype=np.int64)
        ref_ids = np.asarray([p[1] for p in pares], dtype=np.int64)
        if (ref_ids < 0).any() or (ref_ids >= len(preparado["medicoes"][ref_guid]["pontos"])).any():
            raise ValueError("indice de referencia fora da medicao; proposta obsoleta")
        ainda_removida = gpu.para_cpu(selecao["remover"][ref_guid][ref_ids]).astype(bool)
        rep_ids, ref_ids = rep_ids[~ainda_removida], ref_ids[~ainda_removida]
        if not len(rep_ids):
            continue
        obs_ids = origem[ref_guid]["observacao_global"][ref_ids]
        raw = preparado["observacoes"][ref_guid]
        if (obs_ids < 0).any() or (obs_ids >= len(raw)).any():
            raise ValueError("origem da referencia fora das observacoes; proposta obsoleta")
        Tref = preparado["poses"][ref_guid]
        p = gpu.para_gpu(np.asarray(raw["p"][obs_ids]), xp.float64)
        raw_pos[rep_ids] = _transformar(p, Tref)
        pos[rep_ids] = preparado["medicoes"][ref_guid]["pontos"][ref_ids]
        normal[rep_ids] = preparado["medicoes"][ref_guid]["normais"][ref_ids]
        noise[rep_ids] = preparado["medicoes"][ref_guid]["ruido_mm"][ref_ids]
    return pos, raw_pos, normal, noise


def _validar_hashes(projeto, poses, preparado):
    """Revalida identidade do dado e alinhamento antes de aplicar qualquer mascara."""
    for guid, T in poses.items():
        base = Path(projeto) / "data" / guid
        h = preparado["hashes"].get(guid, {})
        if sha256(base / "cache/frames.dataset") != h.get("dataset_sha256"):
            raise ValueError(f"frames.dataset mudou depois da medicao: {guid}")
        if _hash_objeto(T.tolist()) != h.get("poses_sha256"):
            raise ValueError(f"pose mudou depois da medicao: {guid}")
        if sha256(base / "raw_preview.ply") != h.get("preview_sha256"):
            raise ValueError(f"raw_preview.ply mudou depois da medicao: {guid}")
        if _hash_objeto(preparado["calibracoes"][guid]) != h.get("calibracao_sha256"):
            raise ValueError(f"calibracao mudou depois da medicao: {guid}")
        cache = Path(h.get("cache_pasta", ""))
        if not cache.is_dir() or sha256(cache / "observacoes.json") != h.get("cache_sha256"):
            raise ValueError(f"cache de observacoes mudou depois da medicao: {guid}")
        if not h.get("arquivo_observacoes_sha256") or sha256(cache / "observacoes.npy") != h.get(
            "arquivo_observacoes_sha256"
        ):
            raise ValueError(f"observacoes.npy mudou depois da medicao: {guid}")
        chave = json.loads((cache / "chave.json").read_text(encoding="utf-8"))
        if chave.get("dataset_sha256") != h["dataset_sha256"]:
            raise ValueError(f"cache de qualidade pertence a outro dataset: {guid}")


def _validar_origem(preparado):
    """Vincula cada representante a um registro real do cache e seus índices quadro/pixel."""
    for guid, obs in preparado["observacoes"].items():
        origem = preparado["origem"][guid]
        n = len(preparado["medicoes"][guid]["pontos"])
        ids = np.asarray(origem["observacao_global"], dtype=np.int64)
        if ids.shape != (n,) or (ids < 0).any() or (ids >= len(obs)).any():
            raise ValueError(f"origem de representante invalida para {guid}")
        if not np.array_equal(np.asarray(origem["quadro"]), np.asarray(obs["quadro"][ids])):
            raise ValueError(f"índices de quadro divergiram da origem para {guid}")
        if not np.array_equal(np.asarray(origem["pixel"]), np.asarray(obs["pixel"][ids])):
            raise ValueError(f"índices de pixel divergiram da origem para {guid}")
        if np.asarray(obs["quadro"]).ndim != 1 or np.asarray(obs["pixel"]).ndim != 1:
            raise ValueError(f"índices de observação invalidos para {guid}")


def _propor_observacoes(preparado, selecao, dist_max, cos_normais, voxel):
    """Expande surfels propostos a pixels apenas apos busca 3D na referencia real escolhida."""
    xp = gpu.xp()
    remover = {}
    motivos = {}
    radius_rep = float(np.sqrt(3) * voxel + 1e-6)
    for guid, obs in preparado["observacoes"].items():
        med = preparado["medicoes"][guid]
        rep = med["pontos"]
        proposta = selecao["remover"][guid]
        ref_p, ref_raw_p, ref_n, ref_noise = _referencias_por_representante(preparado, selecao, guid)
        orig = preparado["origem"][guid]["observacao_global"]
        rep_v = gpu.para_gpu(np.asarray(obs["v"][orig]), xp.float64) @ gpu.para_gpu(
            preparado["poses"][guid][:3, :3].T, xp.float64
        )
        rep_v /= xp.maximum(xp.linalg.norm(rep_v, axis=1, keepdims=True), 1e-12)
        rep_n = med["normais"]
        rep_indice = gpu.Indice(rep, radius_rep)
        removidos, sem_cobertura = [], 0
        T = preparado["poses"][guid]
        for a in range(0, len(obs), 500_000):
            b = min(a + 500_000, len(obs))
            raw_p = _transformar(gpu.para_gpu(np.asarray(obs["p"][a:b]), xp.float64), T)
            raw_v = gpu.para_gpu(np.asarray(obs["v"][a:b]), xp.float64) @ gpu.para_gpu(T[:3, :3].T, xp.float64)
            raw_v /= xp.maximum(xp.linalg.norm(raw_v, axis=1, keepdims=True), 1e-12)
            d, j = rep_indice.mais_proximo(raw_p)
            ok = j >= 0
            jj = xp.maximum(j, 0)
            ok &= proposta[jj] & (d <= radius_rep)
            ok &= xp.einsum("ij,ij->i", raw_v, rep_v[jj]) >= 0.35
            delta = raw_p - ref_p[jj]
            cos = xp.einsum("ij,ij->i", rep_n[jj], ref_n[jj])
            ok &= xp.isfinite(ref_p[jj]).all(axis=1) & xp.isfinite(ref_n[jj]).all(axis=1)
            signed = xp.einsum("ij,ij->i", delta, ref_n[jj])
            tangential = xp.linalg.norm(delta - signed[:, None] * ref_n[jj], axis=1)
            # Um surfel so cobre sua area tangencial; o erro normal precisa exceder
            # a incerteza da referencia antes de remover pontos brutos candidatos.
            limite_normal = xp.maximum(0.05, 1.5 * ref_noise[jj] + 0.015)
            ref_source_ok = xp.linalg.norm(ref_raw_p[jj] - ref_p[jj], axis=1) <= radius_rep
            detail = med.get("detalhe_protegido")
            if detail is not None:
                ok &= ~detail[jj]
            ok &= ref_source_ok & (tangential <= radius_rep) & (xp.abs(signed) > limite_normal) & (cos >= cos_normais)
            good = gpu.para_cpu(xp.flatnonzero(ok)).astype(np.int64) + a
            if len(good):
                removidos.append(good)
            sem_cobertura += int((proposta[jj] & ~ok).sum())
        remover[guid] = np.unique(np.concatenate(removidos)) if removidos else np.empty(0, np.int64)
        motivos[guid] = dict(
            observacoes_propostas=int(proposta.sum()),
            observacoes_revalidas=len(remover[guid]),
            sem_cobertura_real=sem_cobertura,
        )
    return remover, motivos


def _escrever_frames(src, dst, remover, obs):
    """Aplica a lista de observacoes por quadro/pixel em uma copia do banco SQLite."""
    with tempfile.TemporaryDirectory(prefix="qualidade_3d_") as tmp:
        scan = RevoScan(str(src), tmp)
        try:
            remover = np.asarray(remover, dtype=np.int64)
            if (remover < 0).any() or (remover >= len(obs)).any() or len(np.unique(remover)) != len(remover):
                raise ValueError("índice global de observação invalido ou repetido")
            por_quadro = {}
            for oi in remover:
                frame, pixel = int(obs["quadro"][oi]), int(obs["pixel"][oi])
                if frame < 0 or frame >= len(scan.ids) or pixel < 0:
                    raise ValueError("índice de quadro/pixel fora do dataset de origem")
                por_quadro.setdefault(frame, set()).add(pixel)
            antes = depois = quadros = quadros_zerados = 0
            mascara_frames = []
            aplicadas = []
            quadros_com_proposta = set(por_quadro)
            for k, frame_id in enumerate(scan.ids):
                if not scan.tem_prof[k]:
                    if k in quadros_com_proposta:
                        raise ValueError("proposta aponta para quadro sem profundidade no dataset original")
                    continue
                quadros_com_proposta.discard(k)
                dados = scan.profundidade(frame_id)
                w, h, mask, z, tail = dados
                pix = np.flatnonzero(mask)
                remover_pix = por_quadro.get(k, set())
                if any(p >= w * h for p in remover_pix):
                    raise ValueError("pixel proposto fora da dimensao do quadro")
                if remover_pix:
                    if not len(pix):
                        raise ValueError("pixel proposto para quadro sem profundidade")
                    propostos = np.asarray(sorted(remover_pix), dtype=np.int64)
                    loc = np.searchsorted(pix, propostos)
                    presente = (loc < len(pix)) & (pix[np.minimum(loc, max(0, len(pix) - 1))] == propostos)
                    if not presente.all():
                        raise ValueError("pixel proposto nao existe na mascara de profundidade original")
                    obs_quadro = remover[np.asarray(obs["quadro"][remover]) == k]
                    obs_pixel = np.asarray(obs["pixel"][obs_quadro], dtype=np.int64)
                    idx_pixel = np.searchsorted(propostos, obs_pixel)
                    escolhidos_pixel = (idx_pixel < len(propostos)) & (
                        propostos[np.minimum(idx_pixel, max(0, len(propostos) - 1))] == obs_pixel
                    )
                    validar_ids = obs_quadro[escolhidos_pixel]
                    z_ref = z[loc]
                    pontos_locais = np.asarray(obs["p"][validar_ids], dtype=np.float64)
                    T_frame = scan.T[k]
                    camera = (pontos_locais - T_frame[:3, 3]) @ T_frame[:3, :3]
                    z_cache = -camera[:, 2]
                    mapa_z = dict(zip(propostos.tolist(), z_ref.tolist()))
                    z_orig = np.asarray([mapa_z[int(x)] for x in obs_pixel[escolhidos_pixel]])
                    if not np.allclose(z_cache, z_orig, atol=2e-3, rtol=2e-5):
                        raise ValueError("profundidade da observacao diverge do pacote quadro/pixel original")
                if not remover_pix:
                    antes += len(z)
                    depois += len(z)
                    quadros += 1
                    continue
                drop = np.isin(pix, np.fromiter(remover_pix, dtype=np.int64))
                if not drop.any():
                    antes += len(z)
                    depois += len(z)
                    quadros += 1
                    continue
                keep = ~drop
                antes += len(z)
                quadros += 1
                if not keep.any():
                    # Evidencia redundante para todo pixel autoriza apagar somente RGBD;
                    # a linha FrameInfo e sua pose permanecem intactas.
                    bits = np.zeros(w * h, dtype=bool)
                    bits[pix] = True
                    mascara_frames.append(dict(quadro=k, forma=[h, w], bits=np.packbits(bits, bitorder="little")))
                    do_quadro = remover[np.asarray(obs["quadro"][remover]) == k]
                    aplicadas.extend(do_quadro[np.isin(np.asarray(obs["pixel"][do_quadro]), pix)].tolist())
                    scan.c.execute("delete from FrameRGBD where id=?", (frame_id,))
                    quadros_zerados += 1
                    continue
                if drop.any():
                    bits = np.zeros(w * h, dtype=bool)
                    bits[pix[drop]] = True
                    mascara_frames.append(
                        dict(
                            quadro=k,
                            forma=[h, w],
                            bits=np.packbits(bits, bitorder="little"),
                        )
                    )
                    do_quadro = remover[np.asarray(obs["quadro"][remover]) == k]
                    aplicadas.extend(do_quadro[np.isin(np.asarray(obs["pixel"][do_quadro]), pix[drop])].tolist())
                nova = np.zeros(w * h, dtype=bool)
                nova[pix[keep]] = True
                blob = (
                    struct.pack("<III", w, h, int(keep.sum()))
                    + np.packbits(nova, bitorder="little").tobytes()
                    + tail
                    + z[keep].astype("<f4").tobytes()
                )
                scan.c.execute("update FrameRGBD set data=? where id=?", (blob, frame_id))
                depois += int(keep.sum())
            if quadros_com_proposta:
                raise ValueError(f"proposta aponta para quadro inexistente: {sorted(quadros_com_proposta)}")
            if not depois:
                raise ValueError("mascara deixaria o scan sem observacoes")
            scan.c.commit()
            scan.c.execute("pragma wal_checkpoint(TRUNCATE)")
            if scan.c.execute("pragma integrity_check").fetchone()[0] != "ok":
                raise ValueError("integridade SQLite falhou")
            scan.c.close()
            with open(scan.db, "r+b") as f:
                f.write(scan.sig)
            shutil.copyfile(scan.db, dst / "cache/frames.dataset")
            return (
                dict(
                    pixels_antes=antes,
                    pixels_depois=depois,
                    quadros=quadros,
                    quadros_zerados=quadros_zerados,
                ),
                dict(
                    quadros=np.asarray([x["quadro"] for x in mascara_frames], dtype=np.int32),
                    formas=np.asarray([x["forma"] for x in mascara_frames], dtype=np.uint32).reshape(-1, 2),
                    offsets=np.cumsum([0, *[len(x["bits"]) for x in mascara_frames]], dtype=np.int64),
                    bits=np.concatenate([x["bits"] for x in mascara_frames])
                    if mascara_frames
                    else np.empty(0, np.uint8),
                    observacoes=np.asarray(aplicadas, dtype=np.int64),
                ),
            )
        finally:
            scan.c.close()


def _artefatos(pasta, preparado, selecao, remover, poses, rel):
    """NPZ/PLY integrais e HTML de navegacao com amostra explicitamente marcada."""
    xp = gpu.xp()
    if not pasta.is_dir() or any(pasta.iterdir()):
        raise FileExistsError(f"diretorio de artefatos nao reservado ou nao vazio: {pasta}")
    bons, ruins, semref = [], [], []
    for g, med in preparado["medicoes"].items():
        p = gpu.para_cpu(med["pontos"])
        rm = gpu.para_cpu(selecao["remover"][g]).astype(bool)
        nr = np.asarray(selecao["referencia_scan"][g], dtype=object)
        covered = np.array([x is not None for x in nr])
        bons.append(p[~rm])
        ruins.append(p[rm])
        semref.append(
            p[(gpu.para_cpu(med["rastreio_ruim"]).astype(bool) | (gpu.para_cpu(med["ruido_mm"]) > 0.05)) & ~covered]
        )
    removidas_raw = []
    for g, indices in remover.items():
        if len(indices):
            obs = preparado["observacoes"][g]
            points = gpu.para_gpu(np.asarray(obs["p"][indices]), xp.float64)
            removidas_raw.append(gpu.para_cpu(_transformar(points, poses[g])))
    arrays = {
        "preferidas_mm": np.concatenate(bons) if bons else np.empty((0, 3)),
        "redundantes_candidatas_mm": np.concatenate(ruins) if ruins else np.empty((0, 3)),
        "sem_referencia_mm": np.concatenate(semref) if semref else np.empty((0, 3)),
        "pixels_removidos_mm": np.concatenate(removidas_raw) if removidas_raw else np.empty((0, 3)),
    }
    np.savez_compressed(pasta / "observacoes_3d_integral.npz", **arrays)
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
    for nome, pts in arrays.items():
        gravar_ply(str(pasta / f"{nome}.ply"), header, pts)
    resumo = {
        "estado": rel["estado"],
        "pendencias": len(rel["pendencias"]),
        "scans": {
            g: {
                "pixeles_removidos": r["pixels_removidos"],
                "quadros_zerados": r["quadros_zerados"],
            }
            for g, r in rel["scans"].items()
        },
    }
    _escrever_html_navegacao(pasta / "navegacao_amostrada.html", arrays, resumo)


def _escrever_html_navegacao(path, arrays, resumo, max_pontos=20_000):
    """HTML WebGL offline: órbita, zoom, legenda por camada e corte por plano."""
    from plotly.offline import get_plotlyjs

    cores = {
        "preferidas_mm": ("Superfícies preferidas / preservadas", "#65b9ff"),
        "redundantes_candidatas_mm": ("Candidatas redundantes", "#ff6060"),
        "sem_referencia_mm": ("Suspeitas sem referência segura", "#ffb347"),
        "pixels_removidos_mm": ("Observações removidas", "#ba83ff"),
    }
    camadas = []
    for nome, (rotulo, cor) in cores.items():
        pontos = np.asarray(arrays[nome], dtype=np.float32)
        passo = max(1, int(np.ceil(len(pontos) / max_pontos)))
        amostra = pontos[::passo]
        camadas.append(
            {
                "nome": rotulo,
                "cor": cor,
                "total": int(len(pontos)),
                "amostra": amostra.tolist(),
            }
        )
    script = r"""const camadas=__CAMADAS__;
const dados=camadas.map((c,i)=>({type:'scatter3d',mode:'markers',
 name:c.nome+' · amostra '+c.amostra.length.toLocaleString()+' / '+c.total.toLocaleString(),
 x:c.amostra.map(p=>p[0]),y:c.amostra.map(p=>p[1]),z:c.amostra.map(p=>p[2]),
 marker:{size:2,color:c.cor,opacity:.72},visible:i<2?true:'legendonly',
 hovertemplate:'X %{x:.3f} mm<br>Y %{y:.3f} mm<br>Z %{z:.3f} mm<extra>'+c.nome+'</extra>'}));
const layout={paper_bgcolor:'#111923',plot_bgcolor:'#111923',font:{color:'#e8edf1'},
 margin:{l:0,r:0,b:0,t:12},legend:{x:0,y:1},uirevision:'qualidade',
 scene:{aspectmode:'data',xaxis:{title:'X (mm)'},yaxis:{title:'Y (mm)'},zaxis:{title:'Z (mm)'}}};
Plotly.newPlot('plot',dados,layout,{responsive:true,displaylogo:false,scrollZoom:true});
function aplicarCorte(){
 const gd=document.getElementById('plot');
 const n=['nx','ny','nz'].map(k=>Number(document.getElementById(k).value));
 const d=Number(document.getElementById('d').value),e=Number(document.getElementById('esp').value);
 const nn=Math.hypot(...n);
 if(!Number.isFinite(nn)||nn<1e-12||!Number.isFinite(d)||!Number.isFinite(e)||e<0){alert('Plano de corte inválido');return}
 const xyz=camadas.map(c=>{const ps=c.amostra.filter(p=>e===0||Math.abs((p[0]*n[0]+p[1]*n[1]+p[2]*n[2]-d)/nn)<=e/2);
 return {x:ps.map(p=>p[0]),y:ps.map(p=>p[1]),z:ps.map(p=>p[2])}});
 Plotly.restyle(gd,{x:xyz.map(v=>v.x),y:xyz.map(v=>v.y),z:xyz.map(v=>v.z)},xyz.map((_,i)=>i));
}
function resetar(){document.getElementById('esp').value=0;aplicarCorte()}
document.getElementById('aplicar').onclick=aplicarCorte;
document.getElementById('resetar').onclick=resetar;
"""
    script = script.replace("__CAMADAS__", json.dumps(camadas, separators=(",", ":")))
    html = f"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qualidade 3D · navegação amostrada</title>
<style>body{{margin:0;background:#111923;color:#e8edf1;font:14px system-ui}}header{{padding:14px 18px}}
#plot{{height:78vh;min-height:480px}}input{{width:75px;padding:6px;margin:3px}}button{{padding:7px;margin:3px}}
.aviso{{color:#ffc875}}pre{{white-space:pre-wrap}}</style>
<header><h2>Qualidade 3D — navegação</h2>
<p class="aviso">Vista WebGL amostrada: até {max_pontos:,} pontos por camada. Use legenda para isolar camadas; orbite com arrasto e amplie com roda/gesto.
Os arquivos NPZ e PLY ao lado contêm os conjuntos integrais para consulta e revisão.</p>
<div>Corte por plano (mm): n = <input id="nx" value="0"><input id="ny" value="0"><input id="nz" value="1">
d = <input id="d" value="0"> espessura = <input id="esp" value="0"> <button id="aplicar">Aplicar corte</button><button id="resetar">Peça inteira</button></div>
<pre>{json.dumps(resumo, ensure_ascii=False, indent=2)}</pre></header><div id="plot"></div>
<script>{get_plotlyjs()}</script><script>{script}</script></html>"""
    Path(path).write_text(html, encoding="utf-8")


def _salvar_mascaras(pasta, guids, mascaras, hashes):
    """Grava bitsets compactos por quadro e os hashes que autorizam reaplicacao/revisao."""
    saida = {}
    for i, guid in enumerate(guids):
        payload = mascaras[guid]
        nome = f"mascara_{i:03d}.npz"
        caminho = pasta / nome
        np.savez_compressed(caminho, **payload)
        h = hashes[guid]
        saida[guid] = {
            "arquivo": f"qualidade_3d/{nome}",
            "sha256": sha256(caminho),
            "dataset_sha256": h["dataset_sha256"],
            "preview_sha256": h["preview_sha256"],
            "calibracao_sha256": h["calibracao_sha256"],
            "poses_sha256": h["poses_sha256"],
            "pixels_removidos": int(len(payload["observacoes"])),
            "quadros": int(len(payload["quadros"])),
        }
    return saida


def _salvar_regioes(pasta, guids, preparado, selecao):
    """Proveniência numérica integral por representante; GUIDs ficam em tabela JSON pequena."""
    tabela = sorted({ref for guid in guids for ref in selecao["referencia_scan"][guid] if ref is not None})
    codigo = {guid: i for i, guid in enumerate(tabela)}
    resultado = {"referencia_guid_codes": {str(i): guid for guid, i in codigo.items()}, "scans": {}}
    for i, guid in enumerate(guids):
        med = preparado["medicoes"][guid]
        origem = preparado["origem"][guid]
        refs = selecao["referencia_scan"][guid]
        arrays = {
            "pontos_mm": gpu.para_cpu(med["pontos"]),
            "normais": gpu.para_cpu(med["normais"]),
            "ruido_mm": gpu.para_cpu(med["ruido_mm"]),
            "rms_espacial_mm": gpu.para_cpu(med.get("rms_espacial_mm", np.full(len(refs), np.inf))),
            "incidencia": gpu.para_cpu(med["incidencia"]),
            "quadros": gpu.para_cpu(med["quadros"]),
            "rastreio_ruim": gpu.para_cpu(med["rastreio_ruim"]),
            "consistencia_temporal": gpu.para_cpu(med.get("consistencia_temporal", np.zeros(len(refs), bool))),
            "evidencia_temporal": gpu.para_cpu(med.get("evidencia_temporal", np.zeros(len(refs), bool))),
            "detalhe_protegido": gpu.para_cpu(med.get("detalhe_protegido", np.zeros(len(refs), bool))),
            "remover": gpu.para_cpu(selecao["remover"][guid]),
            "qualidade_score": gpu.para_cpu(
                med["ruido_mm"]
                + 0.01 / gpu.xp().maximum(med["quadros"], 1)
                + 0.01 * (1.0 - gpu.xp().clip(med["incidencia"], -1, 1))
            ),
            "referencia_guid_code": np.asarray([codigo.get(ref, -1) for ref in refs], dtype=np.int32),
            "referencia_indice": gpu.para_cpu(selecao["referencia_indice"][guid]).astype(np.int64),
            "observacao_global": np.asarray(origem["observacao_global"], dtype=np.int64),
            "quadro_origem": np.asarray(origem["quadro"], dtype=np.int32),
            "pixel_origem": np.asarray(origem["pixel"], dtype=np.int32),
        }
        nome = f"regioes_{i:03d}.npz"
        caminho = pasta / nome
        np.savez_compressed(caminho, **arrays)
        resultado["scans"][guid] = {
            "arquivo": f"qualidade_3d/{nome}",
            "sha256": sha256(caminho),
            "representantes": int(len(refs)),
        }
    return resultado


def aplicar(
    projeto,
    poses_path,
    saida,
    calibracoes_fixas=None,
    cache=None,
    voxel=0.2,
    raio=0.4,
    dist_max=0.3,
    cos_normais=0.95,
):
    """Gera copia filtrada; qualquer erro deixa a revisao ativa intacta e destino sem publicar."""
    from ciclo_precisao import scans
    from medicoes_qualidade_3d import preparar_medicoes
    from selecao_qualidade import selecionar_regioes

    projeto, saida = Path(projeto).resolve(), Path(saida).resolve()
    if projeto == saida or projeto in saida.parents:
        raise ValueError("saida deve ser nova e ficar fora do projeto fonte")
    _, ids = scans(projeto)
    if len(ids) < 2:
        raise ValueError("qualidade 3D exige ao menos dois scans")
    poses = _ler_poses(poses_path, ids)
    poses_sha = _hash_objeto({g: poses[g].tolist() for g in ids})
    artefatos = saida.parent / "qualidade_3d"
    rel_path = saida.parent / "qualidade_3d.json"
    criou_saida = criou_artefatos = criou_relatorio = False
    try:
        saida.mkdir()
        criou_saida = True
        artefatos.mkdir()
        criou_artefatos = True
        with rel_path.open("x", encoding="utf-8"):
            pass
        criou_relatorio = True
        preparado = preparar_medicoes(
            projeto,
            {g: poses[g] for g in ids},
            cache=cache,
            voxel=voxel,
            raio=raio,
            calibracoes_fixas=calibracoes_fixas,
        )
        _validar_hashes(projeto, poses, preparado)
        _validar_origem(preparado)
        selecao = selecionar_regioes(preparado["medicoes"], dist_max=dist_max, cos_normais=cos_normais)
        remover, motivos = _propor_observacoes(preparado, selecao, dist_max, cos_normais, voxel)
        _validar_hashes(projeto, poses, preparado)
        total_remover = sum(len(v) for v in remover.values())
        pendencias = selecao["relatorio"].get("pendencias", [])
        if total_remover == 0 and not pendencias:
            raise ValueError("nenhuma observacao redundante e comprovadamente pior")
        shutil.copytree(projeto, saida, dirs_exist_ok=True)
        _validar_hashes(projeto, poses, preparado)
        stats = {}
        mascaras = {}
        remover_real = {}
        for guid in ids:
            obs = preparado["observacoes"][guid]
            stats[guid], mascaras[guid] = _escrever_frames(
                projeto / "data" / guid, saida / "data" / guid, remover[guid], obs
            )
            remover_real[guid] = mascaras[guid]["observacoes"]
            stats[guid].update(motivos[guid])
            stats[guid]["pixels_removidos"] = stats[guid]["pixels_antes"] - stats[guid]["pixels_depois"]
        aplicadas = sum(v["pixels_removidos"] for v in stats.values())
        if aplicadas == 0 and not pendencias:
            raise ValueError("nenhuma observacao foi alterada ou ficou pendente")
        rel = {
            "estado": "revisao_pendente" if pendencias else "revisao_aguardando_inspecao",
            "poses_arquivo": str(Path(poses_path).resolve()),
            "poses_sha256": poses_sha,
            "hashes": preparado["hashes"],
            "calibracoes": preparado["calibracoes"],
            "medicao": preparado.get("_meta", {}),
            "scans": stats,
            "pendencias": pendencias,
            "selecao": selecao["relatorio"],
            "limites": "So pixels raw revalidados contra observacao real referenciada saem. Deslocamento sem correspondencia fica como pendencia. Sem reparo, suavizacao ou fechamento de furos.",
        }
        from auditar_projeto import auditar

        rel["auditoria"] = auditar(projeto, saida)
        _artefatos(artefatos, preparado, selecao, remover_real, poses, rel)
        rel["regioes"] = _salvar_regioes(artefatos, ids, preparado, selecao)
        rel["mascaras"] = _salvar_mascaras(artefatos, ids, mascaras, preparado["hashes"])
        rel_path.write_text(json.dumps(rel, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
        _validar_hashes(projeto, poses, preparado)
        return dict(projeto=str(saida), qualidade=str(rel_path), relatorio=rel)
    except BaseException:
        if criou_saida:
            shutil.rmtree(saida, ignore_errors=True)
        if criou_artefatos:
            shutil.rmtree(artefatos, ignore_errors=True)
        if criou_relatorio:
            rel_path.unlink(missing_ok=True)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("projeto")
    ap.add_argument("--poses", required=True, help="JSON 4x4 {guid: scan->referencial comum}")
    ap.add_argument("--saida", required=True, help="nova copia; nunca a origem")
    ap.add_argument("--cache")
    ap.add_argument("--calibracoes", help="JSON fixo {guid:{modo:{K,erro_preview_mm}}}, se disponivel")
    ap.add_argument("--voxel", type=float, default=0.2)
    ap.add_argument("--raio", type=float, default=0.4)
    ap.add_argument("--dist-max", type=float, default=0.3)
    ap.add_argument("--cos-normais", type=float, default=0.95)
    a = ap.parse_args()
    try:
        calibracoes = json.loads(Path(a.calibracoes).read_text(encoding="utf-8")) if a.calibracoes else None
        r = aplicar(
            a.projeto,
            a.poses,
            a.saida,
            calibracoes_fixas=calibracoes,
            cache=a.cache,
            voxel=a.voxel,
            raio=a.raio,
            dist_max=a.dist_max,
            cos_normais=a.cos_normais,
        )
    except ValueError as exc:
        ap.error(str(exc))
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
