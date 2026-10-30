# clean_point_cloud

## Interface grafica

Duplo clique em `Abrir_GUI.bat` (ou `.venv\Scripts\python gui\app.py`). Requer `PySide6` no `.venv`.

A página **Assistente** conversa com o Claude Code instalado na máquina (`claude -p`), usando o login da sua assinatura; nenhuma chave de API é usada. No modo *Analista* (padrão) ele roda scripts de análise, lê os PNGs e resume o que mediu, como no CLI; o processamento longo ele propõe como etapa e só entra na fila quando você clica. O modo *Consultar* só lê.
Os formularios sao gerados lendo o `argparse` de cada script em `pipeline/`, entao scripts novos aparecem sozinhos.
