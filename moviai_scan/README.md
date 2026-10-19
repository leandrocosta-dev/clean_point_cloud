# MoviAI Scan

Limpeza, realinhamento e correção de deriva de rastreamento do Revopoint Miraco Plus.
Um sistema só, com uma linha de comando (`run.py`) para as quatro frentes:

| Frente | Comandos | Entrada | Depende de |
|---|---|---|---|
| Nuvem exportada | `limpar`, `mesclar` | PLY exportado do Revo Scan | Open3D |
| Malha | `malha` | PLY já limpo | Open3D |
| CloudCompare | `cc limpar`, `cc deriva` | PLY | CloudCompare instalado |
| Projeto do Revo Scan | `revo info`, `revo limpar`, `revo fundir`, `revo giro`, `revo aplicar` | pasta do projeto (`.revo`) | Open3D só no `revo limpar` |

Pipeline determinística: a LLM saiu do caminho crítico e virou inspetora opcional.

## Instalação

Python 3.10 a 3.12 (o Open3D não tem pacote para 3.13).

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

Para a inspeção por LLM (opcional): `pip install anthropic` e defina `ANTHROPIC_API_KEY`.

## Uso

```
# Limpar uma nuvem (exporte do Revo Scan em PLY, unidade mm)
python run.py limpar peca.ply --saida resultado/
python run.py limpar peca.ply --saida resultado/ --llm      # inspeção final por LLM
python run.py limpar peca.ply --base off                    # forçar ou desligar a detecção de base

# Mesclar varreduras separadas da mesma peça
python run.py mesclar parte1.ply parte2.ply parte3.ply --saida resultado/ --limpar-depois

# Malha Poisson da nuvem limpa
python run.py malha resultado/peca_limpo.ply peca.stl --depth 9

# CloudCompare (procura CloudCompare.exe em Program Files ou use --cc)
python run.py cc limpar a.ply b.ply --saida limpo.ply [--config configs/cloudcompare_alta_resolucao.json] [--dry-run]
python run.py cc limpar --componente _stage_x/a_COMPONENT_3.ply --saida limpo.ply   # quando a escolha automática errar
python run.py cc deriva limpo.ply --saida sem_deriva.ply [--suporte 500]

# Projeto do Revo Scan, quadro a quadro (o original NUNCA é alterado)
python run.py revo info <projeto> [--calibrar]
python run.py revo limpar <projeto> --scans 20261017_172640          # limpa quadro a quadro e propõe correção de pose
python run.py revo fundir <projeto> --scan 20261015_204805 --saida nuvem.ply [--correcao correcao.json]
python run.py revo giro <projeto> --scan S --ref 0-69 --alvo 148-247 --rampa 70-147 \
    --centro X,Y,Z --raio 22.91 --altura 46 [--base-de raw_preview_relatorio.json] --saida pasta/
python run.py revo aplicar <projeto> --correcao pasta/correcao.json   # aceita a correção do giro e a do limpar
```

Código de saída de `limpar` e `mesclar`: 0 = aprovado, 2 = revisar. Dá para usar em lote.

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

### Mescla de varreduras

`mesclar` usa o mesmo registro. Encaixe ambíguo também NÃO é mesclado (status `ambiguo`, código de saída 2).
Teste com duas partes do sintético que se encaixam por simetria: a versão anterior mesclava e entregava 70,5% de precisão
(erro p99 de 21 mm) com código 0; a atual recusa e mantém 100%.

## Projeto do Revo Scan (`revo`)

Lê o `cache/frames.dataset` (SQLite com assinatura trocada) quadro a quadro. O formato está documentado em
`moviai_scan/revo/formato.py` e no `CONTEXTO.md` da raiz. Tudo que grava vai para uma CÓPIA do projeto
(guid novo, `size` recalculado, journal consolidado, assinatura restaurada). Destino existente é recusado.

- `info`: quadros, modos, quadros com profundidade e os maiores saltos de pose, com o número que o Revo Scan mostra.
  Use para achar o trecho suspeito e conferir no Revo Scan antes de mexer.
- `limpar`: atua direto nos quadros do projeto, com o mesmo motor da limpeza de PLY. Código de saída 0 = aprovado, 2 = revisar.
  1. Calibra cada modo contra o `raw_preview.ply` (pula o scan se o erro passar de 0,6 mm) e acha o prato pelas câmeras.
  2. Funde TODOS os quadros numa grade de voxels (nuvem de trabalho de até cerca de 400 mil pontos), lembrando de que
     voxel veio cada pixel.
  3. Roda o motor: ruído, ilhas, trechos deslocados, busca de parâmetros. A peça é o objeto que mais aparece no centro
     da imagem. Um objeto separado só é tratado como trecho deslocado se a maior parte dos pontos dele vem de quadros
     que quase não veem a peça (objeto de apoio aparece junto com a peça; trecho com deriva vem de quadros perdidos).
  4. Cada pixel herda o rótulo do voxel. Na cópia, os pixels removidos são apagados de cada quadro. Poses intactas.
  5. Trecho realinhado com encaixe confiável: os quadros cuja maioria dos pontos caiu nele vão para
     `<scan>_correcao.json` (pose nova = T do encaixe @ pose), com `<scan>_fusao_antes.ply` e `<scan>_fusao_corrigida.ply`
     para conferir o detalhe não simétrico. A pose só é gravada com `revo aplicar` sobre a cópia limpa.
  6. Encaixe ambíguo: nada é corrigido, os pixels do trecho são apagados e o relatório lista os quadros (número do Revo)
     para você conferir e excluir no Revo Scan. Sai também `<scan>_trechoN_sugestao.ply`.

  Teste com gabarito (deriva injetada de 25 graus e 70 mm nos quadros 0 a 40 do scan de 15/10 já corrigido): achou
  exatamente os 41 quadros; depois do `aplicar`, erro de pose de 0,27 mm no centro da peça e os outros quadros intactos.
  Limite: só pega deriva que separa o trecho da peça (vão entre eles). Deriva gradual, em que o trecho fica encostado
  na peça (como o giro de 17,2 graus de 15/10), não é detectada: use `giro`.
- `giro` + `aplicar`: deriva de giro em torno do eixo de peça cilíndrica (trecho cego). Duas etapas de propósito:
  `giro` só estima e grava `correcao.json` com `fusao_original.ply` e `fusao_corrigida.ply`; `aplicar` grava a cópia
  depois que você conferir o detalhe não simétrico (plugue) contra o gabarito. Distância média à superfície não prova
  encaixe em corpo cilíndrico. Se o prato não for achado automaticamente, passe `--base-de` com o relatório de
  `run.py limpar` do `raw_preview.ply`. `--centro` é qualquer ponto do eixo da peça no referencial do mundo
  (pode ser marcado no CloudCompare sobre a fusão). Exemplo da correção de 15/10: `configs/correcao_difusor_20261015.json`.

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

## Limitação conhecida: resultado não reprodutível em scan real

Nos sintéticos duas execuções dão resultado idêntico. No `raw_preview.ply` real do scan de 15/10, a mesma entrada rodada
duas vezes deu custos diferentes nas tentativas do Optuna e, numa das vezes, o trecho saiu `realinhado` e na outra
`ambiguo`. A semente fixa não basta: as rotinas paralelas do Open3D (RANSAC/DBSCAN) variam entre execuções, e
`OMP_NUM_THREADS=1` não resolve. Até corrigir, rode duas vezes em caso de dúvida e trate status diferente como `revisar`.

## Próximos passos

1. Rodar em scans reais e calibrar os `Gates` com o histórico.
2. Se as métricas de borda ainda deixarem passar casos ruins, treinar um classificador por ponto (PointNet++ ou similar) usando os pares antes/depois das limpezas que você já fez manualmente.
3. Para mais de 2 varreduras, trocar a mescla incremental por otimização de grafo de poses (Open3D `PoseGraph`).
