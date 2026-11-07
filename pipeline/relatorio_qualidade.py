"""Resumo legivel e configuracao reprodutivel das etapas de limpeza."""
import json
from pathlib import Path


def gravar_relatorio(saida, resultados, configuracao):
    saida = Path(saida)
    (saida/'configuracao_limpeza.json').write_text(json.dumps(configuracao, indent=2), encoding='utf-8')
    linhas = ['# Qualidade da limpeza', '',
        'As contagens representam observacoes de pixels nos quadros, nao pontos unicos.', '',
        '| Scan | Estado | Antes | Depois | Remocao |', '|---|---|---:|---:|---:|']
    for ident, r in resultados.items():
        antes, depois = r.get('antes', 0), r.get('depois', 0)
        pct = f'{100*(antes-depois)/antes:.2f}%' if antes else '—'
        linhas.append(f"| {ident} | {r['status']} | {antes:,} | {depois:,} | {pct} |")
    linhas += ['', '## Comparacao dos filtros e preservacao', '',
        'Fixo e adaptativo sao propostas locais na mesma selecao geometrica. O resultado final',
        'tambem considera ilhas 3D e evidencia temporal; nao representa classificacao verdadeira de ruido.', '',
        '| Scan | Fixo local: mantidos | Adaptativo local: mantidos | Temporal final | Bordas mantidas | Celulas mantidas | Pares suspeitos |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for ident, r in resultados.items():
        q = r.get('precisao')
        if not q:
            continue
        b = q['borda_antes']
        borda = f"{q['borda_depois']:,}/{b:,}"
        cobertura = q['cobertura_amostrada']
        linhas.append(f"| {ident} | {q['fixo_depois']:,} | {q['adaptativo_depois']:,} | {r['depois']:,} | {borda} | {100*cobertura['fracao']:.2f}% | {q['pares_suspeitos']} |")
    linhas += ['', '## Como interpretar', '',
        '- Cobertura: celulas de 0,5 mm ocupadas por uma amostra de cada oitavo pixel da peca selecionada; nao e cobertura fisica absoluta.',
        '- Bordas: contornos de cobertura e saltos de profundidade; o numero nao garante preservacao de todos os detalhes.',
        '- Pares suspeitos: vies de profundidade por reprojecao, sem ICP. Sinaliza revisao; nao prova erro de pose e nao move pontos.',
        '- A validacao temporal protege descartes dos filtros de pixels e ilhas. A segmentacao geometrica inicial continua removendo objetos fora da peca.',
        '- Regioes inconclusivas sao preservadas: ruido residual pode permanecer.',
        '- Detalhes por quadro, dispersao e tolerancias: data/<ID>/qualidade_precisao.json.',
        '- Cortes comparativos: data/<ID>/cortes_qualidade.png. Pixels para revisao: revisao_pixels.npz (indices no mapa original por quadro).',
        '- identificadores.json relaciona IDs originais aos IDs da entrega, evitando colisao na biblioteca do Revo.',
        '- Merge e scans sem quadros ficam sem limpeza; refazer fusao para refletir os frames limpos.',
        '- Exatidao dimensional exige referencia fisica. Nao foi aplicado fechamento de furos.', '']
    (saida/'QUALIDADE.md').write_text('\n'.join(linhas), encoding='utf-8')
