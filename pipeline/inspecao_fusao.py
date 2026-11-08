"""Inspecao 3D local e consultas integrais com origem quadro/pixel.

    python pipeline/ciclo_precisao.py inspecionar CICLO --scan GUID --saida vista.html
    python pipeline/ciclo_precisao.py consultar CICLO --scan GUID --caixa X0 Y0 Z0 X1 Y1 Z1 --saida consulta

HTML reduz pontos apenas para navegacao; consultas usam todas as observacoes.
As mascaras de caixa sao calculadas na GPU (gpu.py); a geracao do HTML e os JSON ficam na CPU (sem calculo numerico pesado).
"""

import json
from pathlib import Path

import gpu
import numpy as np
import open3d as o3d
from fusao_superficies import salvar_json
from plotly.offline import get_plotlyjs


def dados(ciclo, scan):
    from ciclo_precisao import carregar, estado

    _, _, rev, m = carregar(ciclo)
    if scan not in m["scans"]:
        raise ValueError("scan desconhecido")
    st = estado(rev)
    if scan not in st["nativo"]:
        raise ValueError("falta fusao nativa")
    base = rev / st["nativo"][scan]
    return rev, m, st, base


def consultar(ciclo, scan, caixa, saida):
    rev, m, st, base = dados(ciclo, scan)
    lo, hi = np.asarray(caixa[:3], float), np.asarray(caixa[3:], float)
    if not np.isfinite([lo, hi]).all() or not (hi > lo).all():
        raise ValueError("caixa invalida")
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=False)
    obs = np.load(base / "origem/observacoes.npy", mmap_mode="r")
    selecionados = []
    xp = gpu.xp()
    glo, ghi = gpu.para_gpu(lo, xp.float64), gpu.para_gpu(hi, xp.float64)
    for a in range(0, len(obs), 4_000_000):
        p = gpu.para_gpu(np.ascontiguousarray(obs["p"][a : a + 4_000_000]), xp.float64)
        mask = xp.all((p >= glo) & (p <= ghi), axis=1)
        ids = gpu.para_cpu(xp.flatnonzero(mask)) + a
        if len(ids):
            selecionados.append(ids)
    idx = np.concatenate(selecionados) if selecionados else np.empty(0, np.int64)
    # Arquivo integral para ferramentas; JSON da LLM permanece pequeno.
    np.save(saida / "indices_observacoes.npy", idx)
    np.save(saida / "observacoes.npy", obs[idx])
    frameinfo = json.loads((base / "origem/observacoes.json").read_text())
    lookup = {x["indice"]: x["id"] for x in frameinfo["quadros"]}
    frames, counts = np.unique(obs["quadro"][idx], return_counts=True)
    with np.load(base / "superficies.npz") as z:
        pz = gpu.para_gpu(z["pontos"], xp.float64)
        keep = gpu.para_cpu(xp.all((pz >= gpu.para_gpu(lo)) & (pz <= gpu.para_gpu(hi)), axis=1))
        np.savez(saida / "superficies.npz", **{k: z[k][keep] for k in z.files})
    am = idx[:: max(1, len(idx) // 100)][:100]
    rel = dict(
        manifesto=m["id"],
        scan=scan,
        caixa_mm=[lo.tolist(), hi.tolist()],
        observacoes=len(idx),
        superficies=int(keep.sum()),
        quadros=[dict(indice=int(k), id=lookup[int(k)], observacoes=int(n)) for k, n in zip(frames, counts)],
        amostra=[
            dict(xyz_mm=obs["p"][i].tolist(), quadro_id=lookup[int(obs["quadro"][i])], pixel=int(obs["pixel"][i]))
            for i in am
        ],
        aviso="Consulta integral; amostra JSON limitada a 100, arquivos NPY contem todos os selecionados.",
    )
    salvar_json(saida / "consulta.json", rel)
    # Diario de inspecao fora dos dados congelados.
    st["revisoes_3d"].setdefault(scan, []).append(
        dict(tipo="consulta_integral", caixa=caixa, saida=str(saida.resolve()))
    )
    salvar_json(rev / "estado.json", st)
    return rel


def gerar_html(ciclo, scan, saida, max_pontos=100000):
    rev, m, st, base = dados(ciclo, scan)
    saida = Path(saida)
    if saida.exists():
        raise FileExistsError(saida)
    z = np.load(base / "superficies.npz")
    layers = []

    def layer(nome, p, cor):
        if not len(p):
            return
        am = p[:: max(1, int(np.ceil(len(p) / max_pontos)))]
        layers.append(dict(nome=nome, xyz=np.round(am, 5).tolist(), cor=cor, total=len(p)))

    layer("Nativo confirmado", z["pontos"][z["confirmado"]], "#6cbaff")
    layer("Nativo inconclusivo", z["pontos"][~z["confirmado"]], "#ff8e40")
    layer("Antes do refino", z["pontos_medidos"], "#9ca1a7")
    if scan in st["revo"]:
        layer(
            "Revo da mesma revisao", np.asarray(o3d.io.read_point_cloud(st["revo"][scan]["arquivo"]).points), "#80e191"
        )
    lo = z["pontos"].min(axis=0).tolist()
    hi = z["pontos"].max(axis=0).tolist()
    config = dict(manifesto=m["id"], scan=scan, min=lo, max=hi, layers=layers)
    template = r"""<!doctype html><html lang="pt-BR"><meta charset="utf-8"><title>Inspeção 3D da fusão</title>
<style>body{margin:0;background:#111923;color:#e8edf1;font:14px system-ui}header{padding:16px}input{width:72px}button,select,input{padding:7px;margin:3px}#plot{height:72vh}.aviso{color:#ffba75}label{margin-right:12px}</style>
<header><b>Inspeção 3D — peça / suporte / resíduos</b><div id="ident"></div>
<p class="aviso">Navegação com amostra. Para decisão e rastreabilidade use “consultar”: lê todos os pontos, quadros e pixels da região. Uma vista não comprova inspeção completa.</p>
<div>Caixa de corte (mm): <span id="bounds"></span><button onclick="atualizar()">Aplicar corte 3D</button><button onclick="resetar()">Peça inteira</button></div>
<div>Plano de corte arbitrário: n = <input id="nx" value="0"><input id="ny" value="0"><input id="nz" value="1"> d = <input id="d" value="0"> espessura = <input id="esp" value="0"> mm (0 desliga)</div>
<div><select id="classe"><option value="residuo">Resíduo</option><option value="suporte">Suporte</option><option value="proteger">Proteger peça</option></select>
<input id="motivo" style="width:360px" placeholder="Justificativa da classificação"><button onclick="adicionar()">Adicionar caixa à seleção</button><button onclick="baixar()">Salvar regiões JSON</button><span id="contagem"></span></div>
<p>Arraste para girar; roda para zoom; legenda para isolar camadas. Selecionar uma região apenas salva uma proposta: nenhum frame é alterado aqui.</p></header><div id="plot"></div><pre id="comando"></pre>
<script>__PLOTLY__</script><script>
const cfg=__CONFIG__;let regioes=[];
document.getElementById('ident').textContent='Scan '+cfg.scan+' · Revisão '+cfg.manifesto;
for(let j=0;j<3;j++){for(const tipo of ['min','max']){const i=document.createElement('input');i.id=tipo+j;i.type='number';i.step='.01';i.value=cfg[tipo][j];document.getElementById('bounds').appendChild(i)}}
function caixa(){return ['min','max'].map(tipo=>[0,1,2].map(j=>Number(document.getElementById(tipo+j).value)))}
function atualizar(){let [lo,hi]=caixa();if(lo.some((v,j)=>!Number.isFinite(v)||v>=hi[j])){alert('Caixa inválida');return}
 const n=['nx','ny','nz'].map(x=>Number(document.getElementById(x).value)),d=Number(document.getElementById('d').value),esp=Number(document.getElementById('esp').value),norm=Math.hypot(...n);
 const traces=cfg.layers.map((l,k)=>{const ps=l.xyz.filter(p=>p.every((v,j)=>v>=lo[j]&&v<=hi[j])&&(esp<=0||(norm>0&&Math.abs(p.reduce((s,v,j)=>s+v*n[j],d))/norm<=esp/2)));
 return {type:'scatter3d',mode:'markers',name:l.nome+' ('+l.total.toLocaleString()+' pontos integrais)',x:ps.map(p=>p[0]),y:ps.map(p=>p[1]),z:ps.map(p=>p[2]),marker:{size:1.5,color:l.cor,opacity:.8},visible:k<2?true:'legendonly'}});
 Plotly.react('plot',traces,{uirevision:'keep',paper_bgcolor:'#111923',font:{color:'#e8edf1'},margin:{l:0,r:0,b:0,t:0},scene:{aspectmode:'data',xaxis:{title:'X mm'},yaxis:{title:'Y mm'},zaxis:{title:'Z mm'}},legend:{x:0,y:1}});
 document.getElementById('comando').textContent='Consulta integral: ciclo_precisao.py consultar CICLO --scan '+cfg.scan+' --caixa '+[...lo,...hi].join(' ')+' --saida NOVA_PASTA';}
function resetar(){for(let j=0;j<3;j++)for(const t of ['min','max'])document.getElementById(t+j).value=cfg[t][j];document.getElementById('esp').value=0;atualizar()}
function adicionar(){let [lo,hi]=caixa(),motivo=document.getElementById('motivo').value.trim();if(!motivo||lo.some((v,j)=>!Number.isFinite(v)||v>=hi[j])){alert('Informe caixa válida e justificativa');return}
 if(Number(document.getElementById('esp').value)>0){alert('Desative o plano antes de selecionar: a máscara exportada é a caixa inteira.');return}
 regioes.push({scan:cfg.scan,tipo:'caixa',min:lo,max:hi,classe:document.getElementById('classe').value,motivo});document.getElementById('contagem').textContent=regioes.length+' regiões';}
function baixar(){const b=new Blob([JSON.stringify({manifesto:cfg.manifesto,regioes},null,2)],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='regioes_'+cfg.scan+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)}
atualizar();</script></html>"""
    documento = template.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__CONFIG__", json.dumps(config).replace("</", "<\\/")
    )
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(documento, encoding="utf-8")
    st["revisoes_3d"].setdefault(scan, []).append(
        dict(
            tipo="visualizador_gerado",
            arquivo=str(saida.resolve()),
            amostra_max_por_camada=max_pontos,
            nao_comprova_revisao=True,
        )
    )
    salvar_json(rev / "estado.json", st)
    return dict(
        arquivo=str(saida.resolve()),
        camadas=[dict(nome=l["nome"], total=l["total"], amostra=len(l["xyz"])) for l in layers],
    )
