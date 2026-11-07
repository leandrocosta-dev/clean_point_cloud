# clean_point_cloud

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

A página **Assistente** conversa com o Claude Code instalado na máquina (`claude -p`), usando o login da sua assinatura; nenhuma chave de API é usada. No modo *Analista* (padrão) ele roda scripts de análise, lê os PNGs e resume o que mediu, como no CLI; o processamento longo ele propõe como etapa e só entra na fila quando você clica. O modo *Consultar* só lê.
Os formularios sao gerados lendo o `argparse` de cada script em `pipeline/`, entao scripts novos aparecem sozinhos.
