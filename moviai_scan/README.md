# MoviAI Scan

Limpeza e realinhamento automático de nuvens de pontos do Revopoint Miraco Plus.
Pipeline determinística: a LLM saiu do caminho crítico e virou inspetora opcional.

## Instalação

```
pip install -r requirements.txt
```

Para a inspeção por LLM (opcional): `pip install anthropic` e defina `ANTHROPIC_API_KEY`.

## Uso

```
# Limpar uma nuvem (exporte do Revo Scan em PLY, unidade mm)
python run.py limpar peca.ply --saida resultado/

# Com inspeção final por LLM (1 chamada, no máximo 1 rodada de ajuste)
python run.py limpar peca.ply --saida resultado/ --llm

# Forçar ou desligar a detecção de base
python run.py limpar peca.ply --base off

# Mesclar varreduras separadas da mesma peça
python run.py mesclar parte1.ply parte2.ply parte3.ply --saida resultado/ --limpar-depois
```

Código de saída: 0 = aprovado, 2 = revisar. Dá para usar em lote.

## O que sai

| Arquivo | Conteúdo |
|---|---|
| `*_limpo.ply` | Nuvem final, com cores e normais originais preservadas |
| `*_removidos.ply` | Tudo que saiu, colorido por motivo (azul base, vermelho ruído, laranja ilha, roxo duplicado, amarelo trecho ambíguo) |
| `*_vistas_antes.png` / `*_vistas.png` | Três vistas antes e depois (resultado em cima, removido embaixo) |
| `*_trechoN_sugestao.ply` | Trecho deslocado que não foi mesclado, na melhor posição encontrada, para você conferir manualmente |
| `*_relatorio.json` | Parâmetros, métricas, critérios, candidatos de encaixe, alertas |
| `historico_execucoes.jsonl` | Uma linha por execução. É a base para calibrar critérios e, no futuro, treinar um modelo |

## Como funciona

1. **Diagnóstico**: espaçamento médio entre pontos (mediana do vizinho mais próximo). Todos os parâmetros são múltiplos dele, então valem para qualquer resolução. Nuvens acima de 400 mil pontos são processadas numa cópia reduzida e o resultado é levado de volta para a nuvem original.
2. **Base**: RANSAC de plano. No modo `auto` só é aceito como base um plano que deixa a peça de um lado só e é maior que a pegada da peça (evita confundir face plana da peça com a mesa).
3. **Ruído fino**: Statistical Outlier Removal.
4. **Ilhas**: DBSCAN. O maior cluster é a peça.
5. **Trechos deslocados** (perda de rastreamento): clusters grandes que sobram são candidatos. Isso acontece ANTES de descartar ilhas, senão um pedaço real da peça seria jogado fora.
6. **Métricas e aprovação**: se a tentativa com parâmetros padrão reprovar, o Optuna busca parâmetros (semente fixa, reprodutível, para cedo quando encontra resultado aprovado).

### Critérios de aprovação (`Gates` em `config.py`)

| Métrica | Limite padrão | O que mede |
|---|---|---|
| `noise_removed_frac` | 15% | Fração da peça removida como ruído |
| `isolated_frac` | 0,2% | Pontos soltos que sobraram |
| `base_residual_frac` | 0,3% | Restos de base (pontos perto do plano com normal paralela a ele) |
| `removed_dense_frac` | 3% | Superfície real removida por engano (ponto removido que estava em região densa) |

Alertas que mandam para revisão sem reprovar as métricas: trecho ambíguo, trecho rejeitado, suspeita de superfície dupla, base não detectada com um plano dominando a nuvem.

**Calibre os limites com o seu histórico.** Os valores atuais foram ajustados em dados sintéticos.

## Realinhamento de trechos deslocados

O registro global (FPFH + RANSAC) seguido de ICP é o padrão da indústria, mas tem uma armadilha: escolher o encaixe com mais pontos encostados falha em peças com simetria. O trecho cabe espelhado sobre outra região da peça e cobre mais área que o encaixe correto, que só tem sobreposição parcial.

O sistema usa três ideias para resolver:

- **Candidatos diversos**: além do RANSAC na nuvem inteira, registra pedaços do trecho separadamente e a região perto das bordas abertas.
- **Costura de bordas**: quando o rastreamento cai, a nuvem principal termina numa borda de corte e o trecho começa em outra. No encaixe correto as duas bordas ficam por dentro da superfície da outra parte e deixam de ser borda. Bordas encostadas na base (aro de baixo) ficam fora da conta, porque nunca fecham.
- **Estabilidade**: análise de degenerescência do ICP na região de sobreposição. Se a sobreposição for lisa (parede reta, extrusão, cilindro liso), o encaixe pode deslizar sem piorar. Encaixe instável nunca é aceito sozinho.

Se o encaixe for ambíguo, o trecho **não é mesclado**. Ele sai em `*_trechoN_sugestao.ply` e o status vira `revisar`. Um encaixe errado contamina a peça e é pior que uma lacuna.

### Limite físico

Se a região de sobreposição entre a parte boa e o trecho deslocado não tiver nenhum detalhe geométrico, nenhum algoritmo resolve só com a nuvem: existem infinitos encaixes igualmente bons. A correção nesse caso é na captura:

- Usar marcadores na peça ou na mesa nas regiões lisas.
- Dividir a captura em varreduras curtas, cada uma com sobreposição que inclua algum detalhe (furo, ressalto, aresta), e usar `python run.py mesclar`.

## Testes

```
python testes/gerar_sintetico.py testes/detalhe.ply detalhe   # sobreposição com detalhe: deve realinhar
python testes/gerar_sintetico.py testes/liso.ply liso         # sobreposição lisa: deve marcar revisar
python run.py limpar testes/detalhe.ply --saida saida_detalhe
python testes/avaliar.py saida_detalhe/detalhe_limpo.ply testes/detalhe_verdade.ply
```

Resultados de referência (dados sintéticos, ruído 0,03 mm, espaçamento 0,26 mm):

| Cenário | Status | Precisão (< 0,2 mm) | Cobertura da peça |
|---|---|---|---|
| detalhe | aprovado, trecho realinhado | 99,99% | 98,75% |
| liso | revisar, trecho não mesclado | 99,99% | 64,15% |
| mesclar 2 varreduras | mesclado | 100% | 100% |

## Próximos passos

1. Rodar em scans reais e calibrar os `Gates` com o histórico.
2. Se as métricas de borda ainda deixarem passar casos ruins, treinar um classificador por ponto (PointNet++ ou similar) usando os pares antes/depois das limpezas que você já fez manualmente.
3. Para mais de 2 varreduras, trocar a mescla incremental por otimização de grafo de poses (Open3D `PoseGraph`).
