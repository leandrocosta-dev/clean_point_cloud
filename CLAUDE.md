# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Projeto

Limpeza de nuvens de pontos e correção de deriva de rastreamento do scanner Revopoint Miraco Plus (projetos do Revo Scan 6). Peça de teste atual: difusor elétrico de repelente (cilindro de cerca de 45,8 mm, bloco do plugue, 10 rasgos a 36 graus no topo, furinho na borda oposto ao plugue).

`CONTEXTO.md` é o diário de estado do projeto: formato binário do projeto Revo, resultados por scan, erros confirmados e regras. Ler antes de mexer em poses ou em arquivos `.revo`.

## Regras do usuário

- Responder em português do Brasil, direto, sem bajulação, máxima especificidade.
- Nunca usar travessão em documentos.
- Nunca alterar o projeto original do Revo Scan. Sempre gravar uma CÓPIA (guid novo, `size` recalculado).
- Nunca gravar correção de pose sem verificar o detalhe não simétrico (plugue) contra o gabarito (scan 20261017_171853). Distância média à superfície não prova encaixe em corpo cilíndrico: essa métrica aprovou (94,8%) uma correção errada no scan 173102.
- Preferir EXCLUIR quadros ruins a corrigi-los quando a correção não for comprovada.
- Conferir com o usuário, no Revo Scan, quais quadros ele vê como problemáticos antes de alterar qualquer coisa. O Revo usa exatamente as poses gravadas, não reposiciona.

## Três frentes de código

1. **`moviai_scan/`** (Python, Open3D): pipeline determinística sobre nuvem PLY já exportada. Ponto de entrada `run.py` (subcomandos `limpar` e `mesclar`), orquestração em `moviai_scan/pipeline.py::process`.
2. **`moviai_scan/revo_investigacao/`**: scripts `revo_*.py` que leem e escrevem o projeto Revo quadro a quadro (poses em SQLite). São específicos do scan de 15/10 (eixo do cilindro, trechos de quadros e caminhos fixos no código, inclusive `/home/claude/...`). Ordem de execução no `LEIAME.md` da pasta. Não são reutilizáveis sem edição.
3. **`pipeline/`** (PowerShell + CloudCompare CLI): `Clean-PointCloud.ps1` (ICP, extração de componentes, SOR, ruído, crop, downsample, configurado por `config.*.json`), `Fix-RotationalDrift.ps1` (RANSAC de cilindros, descarta fragmentos de passada deslocada) e `fuse_mesh.py` (Poisson em Open3D). `inspect_frames*.py` são sondas descartáveis do formato binário.

`CONTEXTO.md` cita `moviai_scan/limpar_revo.py`, `moviai_scan/moviai_scan/revo_projeto.py` e `revo.py` (limpeza por quadro com calibração automática). Esses arquivos NÃO estão nesta pasta; confirmar com o usuário onde estão antes de assumir que existem.

## Comandos

```
pip install -r moviai_scan/requirements.txt        # open3d, numpy, scipy, optuna, matplotlib

# rodar a partir de moviai_scan/
python run.py limpar peca.ply --saida resultado/ [--llm] [--base auto|on|off] [--config cfg.json]
python run.py mesclar p1.ply p2.ply --saida resultado/ --limpar-depois

# testes (sintéticos, sem framework de teste)
python testes/gerar_sintetico.py testes/detalhe.ply detalhe   # deve realinhar o trecho
python testes/gerar_sintetico.py testes/liso.ply liso         # deve sair "revisar", trecho não mesclado
python run.py limpar testes/detalhe.ply --saida saida_detalhe
python testes/avaliar.py saida_detalhe/detalhe_limpo.ply testes/detalhe_verdade.ply

# CloudCompare (procura CloudCompare.exe em Program Files ou usa -CCExe)
.\pipeline\Clean-PointCloud.ps1 -Scans a.ply,b.ply -Output limpo.ply [-Config pipeline\config.highres.json] [-DryRun] [-KeepTemp] [-ResumeComponent comp.ply]
.\pipeline\Fix-RotationalDrift.ps1 -InputPly limpo.ply -Output sem_deriva.ply [-SupportPoints 500]
python pipeline/fuse_mesh.py --input limpo.ply --output malha.stl [--depth 9]
```

`run.py limpar` sai com código 0 = aprovado, 2 = revisar. Valores de referência dos testes sintéticos estão em `moviai_scan/README.md`.

## Arquitetura do `moviai_scan`

- Todos os parâmetros geométricos em `config.py` (`Params`) são MULTIPLICADORES do espaçamento médio entre pontos (mediana do vizinho mais próximo), calculado por nuvem. Não introduzir constantes em mm nesse pacote.
- Nuvens acima de `work_max_points` (400 mil) são processadas numa cópia reduzida; os rótulos voltam para a nuvem original por vizinho mais próximo (`_propagate`).
- Fluxo em `process`: diagnóstico, tentativa com parâmetros padrão, busca Optuna (semente fixa) só se reprovar nos `Gates`, rodada final com realinhamento, inspeção opcional por LLM, status, gravação.
- `clean.py::run_clean` rotula cada ponto (base, ruído, ilha, duplicado, trecho ambíguo). Trechos deslocados são detectados ANTES do descarte de ilhas, senão um pedaço real da peça vira ilha.
- `align.py::register`: FPFH + RANSAC + ICP com candidatos diversos, pontuação por costura de bordas abertas (`seam_score`) e análise de degenerescência do ICP (`stability`). Encaixe instável ou ambíguo nunca é mesclado: sai em `*_trechoN_sugestao.ply` e status `revisar`. Motivo: peça simétrica aceita encaixe espelhado com mais sobreposição que o correto.
- `inspect_llm.py`: a LLM não escolhe parâmetros. Recebe render + métricas, devolve JSON com tipos de problema, e cada tipo mapeia para um delta fixo em `AJUSTES`. No máximo 1 rodada extra. Sem `anthropic` ou `ANTHROPIC_API_KEY`, segue sem ela.
- Cada execução acrescenta uma linha em `historico_execucoes.jsonl` na pasta de saída (base para calibrar `Gates`, hoje ajustados só em dados sintéticos).

## Formato do projeto Revo Scan 6 (resumo; detalhes em `CONTEXTO.md`)

- `data/<scan>/cache/frames.dataset` é SQLite com os 16 primeiros bytes trocados por `RPAlgo_file_sys\0`. Ler: copiar e escrever `SQLite format 3\0`. Gravar: consolidar o WAL e restaurar a assinatura original.
- `FrameInfo.bin`: cabeçalho de 16 bytes + matriz 4x4 float64 (pose câmera para mundo) + cauda. `FrameRGBD.data`: máscara de bits 800x600 + profundidades float32 em mm.
- Numeração de quadro do Revo Scan = índice entre os quadros COM profundidade (FrameRGBD), na ordem (bloco, índice). Difere do índice sequencial quando há quadros sem profundidade.

## Dados

`nuvem_pontos/`, `pipeline/output/`, `image*.png` e `.claude/` estão no `.gitignore`. `moviai_scan/revo_work/` e `moviai_scan/saida_atual/` também são dados ou saídas geradas, não código.
