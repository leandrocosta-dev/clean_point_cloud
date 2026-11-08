# clean_point_cloud

## Instalacao e verificacao

Projeto Windows/Python 3.12 com CUDA obrigatoria no processamento numerico.
Open3D/MeshLib ainda executam certas operacoes de malha e registro por features na CPU.

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe pipeline\verificar_cuda.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m ruff check pipeline gui tests setup.py scripts
.\.venv\Scripts\python.exe -m ruff format --check pipeline gui tests setup.py scripts
.\.venv\Scripts\python.exe -m pre_commit install
```

O pacote editavel permite executar scripts e importar `gui` a partir de outra pasta,
sem ajustar `sys.path` no codigo. `scripts/testes_sem_gpu.py` valida instalacao e utilitarios
sem CUDA; nao substitui a suite numerica completa. A CI Windows executa essa selecao e Ruff.

## Apoio e ciclo de revisao

Meça o plano de apoio no scan original, antes de apagar a mesa.
`limpar_por_placa --planos-apoio avaliacao.json --altura-min 1` reutiliza o plano por scan.
O limite padrao `--perda-max 0.30` recusa perdas maiores e devolve resultado parcial (codigo 2),
preservando o scan dessa tentativa. Aumentar esse limite exige conferir o plano e a perda esperada.
O corte automatico por maior area pode escolher uma face da peca depois da pre-limpeza.

`ciclo_precisao.py` oferece `iniciar`, `triagem`, `limpar`, `consenso --poses`, `qualidade --poses`, `fundir`,
`inspecionar`, `consultar`, `corrigir` e `comparar`. Limpeza e consenso publicam revisoes novas
apenas depois de auditoria; IDs, calibracoes e poses sao rastreados. Fusoes, malhas e aprovacoes
antigas nao liberam uma revisao nova. `aprovar` exige parecer 3D e comparacao da revisao atual.
`exportar-merge` continua bloqueado enquanto houver pendencias. Nenhuma etapa fecha furos por padrao.

Benchmarks e limites conhecidos: [diario/BENCHMARK_GPU_2026-11-08.md](diario/BENCHMARK_GPU_2026-11-08.md).

## Escolha das melhores regioes em 3D

`limpar_qualidade_3d.py` compara scans alinhados usando as observacoes 3D dos frames,
consistencia entre quadros, incidencia e normais. Remove somente observacoes piores
com cobertura comprovada por outra vista boa. Detalhes exclusivos, faces opostas
e regioes sem evidencia suficiente permanecem. Suspeitas de perda de rastreio sem
substituto confiavel geram pendencias que impedem aprovar o merge.

```powershell
.\.venv\Scripts\python.exe pipeline\limpar_qualidade_3d.py C:\Scans\Projeto --poses C:\Scans\poses.json --saida C:\Scans\Revisao_nova\projeto
```

Crie a pasta de trabalho `Revisao_nova` vazia; `projeto`, `qualidade_3d` e
`qualidade_3d.json` devem estar ausentes. Cada execucao precisa de sua propria pasta.
As poses devem cobrir todos os scans no mesmo referencial, em milimetros.
O HTML offline permite orbita, zoom, camadas e corte planar de uma amostra;
PLY/NPZ integrais e mascaras quadro/pixel ficam ao lado para consultas completas.
O resultado exige revisao; o algoritmo nao garante correcao de toda deformacao.
Contrato: [diario/SELECAO_QUALIDADE_3D.md](diario/SELECAO_QUALIDADE_3D.md).

## Limpeza adaptativa dos frames

O modo padrao agora e `--modo-limpeza precisao`. Ele ativa a adaptacao local
e valida propostas de remocao dos filtros de pixels e ilhas em ate quatro outros
quadros. Um descarte exige pelo menos dois quadros com espaco livre na posicao;
apoio de outro quadro, oclusao ou falta de cobertura preservam o ponto.
A selecao geometrica inicial da peca continua ativa e deve ser conferida.

Pares com vies de profundidade elevado sao sinalizados e nao autorizam descartes.
As poses e coordenadas nao sao corrigidas automaticamente. A tolerancia usa
mediana/MAD local e dispersao entre quadros, com limite temporal de ate 2x o
`--pixel-tol`. Isto nao substitui alinhamento ou calibracao fisica.

```powershell
.\.venv\Scripts\python.exe pipeline\rodar_tudo.py C:\Scans\Projeto --saida-limpo C:\Scans\Projeto_preciso --so-limpar --modo-limpeza precisao --temporal-passo 8 --temporal-min 2
```

`QUALIDADE.md` compara propostas locais, resultado temporal, bordas e cobertura
amostrada. Em `data/<ID>/`, `qualidade_precisao.json` detalha as evidencias por
quadro, e `cortes_qualidade.png` compara cortes de 1 mm na mesma amostra.
As imagens antes/depois representam as mascaras finais dos frames.

As entregas recebem IDs novos automaticamente para coexistir com o original no
Revo. `identificadores.json` registra o remapeamento. `rodar_tudo` faz isso apos
a uniao e remapeia a selecao enviada ao Revo. `--manter-ids` em `limpar_raw` e
reservado a etapas internas. Para reproduzir o filtro anterior, use
`--modo-limpeza classico --pixel-adaptativo`.

`limpar_raw.py` e `rodar_tudo.py` aceitam `--pixel-adaptativo`,
`--pixel-tol` (mm, padrao 1.0) e `--pixel-comp-min` (pixels, padrao 30).
O modo adaptativo usa mediana/MAD local e conserva pixels sem apoio suficiente.
Ele remove pontos; nao suaviza coordenadas, corrige poses ou fecha furos.
Componentes pequenos ainda sao removidos: use `--pixel-comp-min 1` para preservar
estruturas finas e avalie tambem o efeito da etapa de ilhas 3D.

```powershell
.\.venv\Scripts\python.exe pipeline\rodar_tudo.py C:\Scans\Projeto --saida-limpo C:\Scans\Projeto_limpo --so-limpar --pixel-adaptativo
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Use sempre um destino novo. Reduzir `--pixel-tol` exige validar perda de detalhes
em um scan representativo. Os testes sinteticos nao demonstram exatidao absoluta
nem eliminacao de todo ruido em dados reais. A limpeza agora remove a profundidade
dos quadros totalmente rejeitados, evitando restaurar seus pontos originais.

## Interface grafica

Duplo clique em `Abrir_GUI.bat` (ou `.venv\Scripts\python gui\app.py`). Requer `PySide6` no `.venv`.

A página **Assistente** conversa com o Claude Code instalado na máquina (`claude -p`), usando o login da sua assinatura; nenhuma chave de API é usada. No modo *Analista* (padrão) ele roda scripts e consulta a geometria 3D e as observacoes quadro/pixel; PNGs complementam a analise. O processamento longo ele propõe como etapa e só entra na fila quando você clica. O modo *Consultar* só lê.
Os formularios sao gerados lendo o `argparse` de cada script em `pipeline/`, entao scripts novos aparecem sozinhos.
