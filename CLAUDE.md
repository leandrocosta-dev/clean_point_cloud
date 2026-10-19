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

## Sistema único: `moviai_scan/`

Tudo passa por `moviai_scan/run.py` (subcomandos `limpar`, `mesclar`, `malha`, `cc limpar|deriva`, `revo info|limpar|fundir|giro|aplicar`). Os scripts antigos (`pipeline/*.ps1`, `fuse_mesh.py`, `revo_investigacao/`, `limpar_revo.py`, `revo_projeto.py`, `inspect_frames*.py`) foram absorvidos pelo pacote; estão no commit 7c12f4c se precisar consultar.

Módulos do pacote `moviai_scan/moviai_scan/`, por frente:
- **Nuvem PLY exportada (Open3D):** `pipeline.process` orquestra; `clean`, `align`, `metrics`, `optimize`, `render`, `inspect_llm`, `config`, `geom`.
- **Malha:** `malha.py` (Poisson).
- **CloudCompare CLI:** `cloudcompare.py` + `cloudcompare_padrao.json` (configs alternativas em `moviai_scan/configs/`).
- **Projeto do Revo Scan:** `revo/formato.py` (leitura dos quadros, calibração, gravação de cópia), `revo/prato.py` (prato pelas câmeras, amostra de quadros), `revo/limpeza.py` (limpeza direto nos quadros com o motor do PLY; propõe correção de pose por quadro), `revo/giro.py` (deriva de giro em duas etapas: estimar e aplicar; `aplicar` também grava a correção do `limpar`), `revo/info.py` (saltos de pose).

`moviai_scan/__init__.py` só importa `config`; cada subcomando importa o que precisa. Só `revo/limpeza.py` usa Open3D (pelo motor); `formato`, `prato`, `giro` e `info` rodam sem ele. Mantenha assim.

`revo limpar` funde todos os quadros numa grade de voxels, roda `pipeline.limpar_trabalho` (mesmo fluxo do PLY) e devolve o rótulo a cada pixel pela chave do voxel. `run_clean` aceita `base_fixa` (prato achado pelas câmeras) e `escolher` (regra de peça e de candidatos a trecho usando de que quadro veio cada ponto). Teste com gabarito: injetar deriva conhecida com `revo aplicar` (correção tipo `quadros`) numa cópia e conferir se o `limpar` acha os mesmos quadros e a T inversa.

## Comandos

```
# ambiente (Open3D não tem pacote para Python 3.13; o .venv da raiz usa 3.10)
python -m venv .venv && .venv\Scripts\python -m pip install -r moviai_scan/requirements.txt

# rodar a partir de moviai_scan/
..\.venv\Scripts\python run.py --help
..\.venv\Scripts\python run.py limpar peca.ply --saida resultado/

# testes (sintéticos, sem framework de teste); referência: detalhe 99,99% / 98,75%, liso 99,99% / 64,15% (status revisar)
python testes/gerar_sintetico.py testes/detalhe.ply detalhe
python run.py limpar testes/detalhe.ply --saida saida_detalhe
python testes/avaliar.py saida_detalhe/detalhe_limpo.ply testes/detalhe_verdade.ply

# CloudCompare não instalado nesta máquina: validar com --dry-run --cc <qualquer arquivo>
```

## Arquitetura da limpeza de PLY

- Todos os parâmetros geométricos em `config.py` (`Params`) são MULTIPLICADORES do espaçamento médio entre pontos (mediana do vizinho mais próximo), calculado por nuvem. Não introduzir constantes em mm nesse pacote. `Config.load` recusa chave desconhecida.
- Nuvens acima de `work_max_points` (400 mil) são processadas numa cópia reduzida; os rótulos voltam para a nuvem original por vizinho mais próximo (`_propagate`).
- Fluxo em `process`: diagnóstico, tentativa com parâmetros padrão, busca Optuna só se reprovar nos `Gates`, rodada final com realinhamento, inspeção opcional por LLM, `_status` (todo alerta implica `revisar`), gravação.
- `clean.py::run_clean` rotula cada ponto (base, ruído, ilha, duplicado, trecho ambíguo). Trechos deslocados são detectados ANTES do descarte de ilhas. `final_points` aplica as transformações dos trechos (e gira as normais) para métricas, render e gravação.
- `align.py::register`: FPFH + RANSAC + ICP com candidatos diversos, costura de bordas (`seam_score`) e degenerescência do ICP (`stability`). Encaixe instável ou ambíguo nunca é mesclado, nem na limpeza nem em `merge_scans`.
- **Não determinismo conhecido:** em scan real, duas execuções iguais podem dar status diferente (paralelismo do Open3D; semente e `OMP_NUM_THREADS=1` não resolvem). Nos sintéticos o resultado é idêntico. Ao comparar antes/depois de uma mudança, use os sintéticos.

## Formato do projeto Revo Scan 6 (resumo; detalhes em `revo/formato.py` e `CONTEXTO.md`)

- `data/<scan>/cache/frames.dataset` é SQLite com os 16 primeiros bytes trocados por `RPAlgo_file_sys\0`. Ler: copiar e escrever `SQLite format 3\0`. Gravar: consolidar o WAL e restaurar a assinatura original (`gravar_copia_scan`).
- `FrameInfo.bin`: cabeçalho de 16 bytes + matriz 4x4 float64 (pose câmera para mundo) + cauda. `FrameRGBD.data`: máscara de bits 800x600 + profundidades float32 em mm.
- Numeração de quadro do Revo Scan = índice entre os quadros COM profundidade (FrameRGBD), na ordem (bloco, índice). `RevoScan.numero_revo()` faz a conversão.

## Dados

`nuvem_pontos/`, `pipeline/output/`, `moviai_scan/revo_work/`, `moviai_scan/saida_atual/`, `image*.png`, `.venv/` e `.claude/` estão no `.gitignore`.
