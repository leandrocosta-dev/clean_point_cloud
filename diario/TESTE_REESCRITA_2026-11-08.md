# Teste integrado da reescrita — 08/11/2026

**Resultado: execução funcional de limpeza, consenso, união e fusão; homologação de qualidade reprovada no corte automático do apoio.** A versão continuou sendo alterada por outro processo durante o ensaio. Os resultados abaixo pertencem às versões carregadas por cada comando; não certificam todas as alterações posteriores.

Evidências: `pipeline/output/TESTE_COMPLETO_REESCRITA_20261108_150536/`. Projeto real: `D:/Projects/Project20261106_210152`, três scans, 1.543 quadros. Todas as saídas foram criadas em pastas novas. Nenhum fechamento de furos foi solicitado ou executado.

## Ambiente e verificações básicas

- RTX 3060, 12 GB; CuPy 14.2; runtime CUDA 12.9. Teste NVRTC/cuBLAS: erro numérico zero. Open3D instalado continua sem CUDA.
- Política de recursos do repositório: modo máximo, prioridade alta e 24 processadores lógicos. Algumas etapas ainda usavam Open3D/SciPy na CPU na versão inicialmente executada.
- `python -m pip check`: aprovado.
- Primeira execução de `python -m unittest discover -s tests -v`: **108 testes aprovados em 9,211 s**. Log `testes_automatizados.log`.
- Sete verificações adicionais da camada `gpu.py`: vizinho, kNN, voxel, normais, RANSAC, componentes e ICP aprovados. Maior discrepância de distância do kNN contra SciPy: 0,00000197 mm. Detalhes em `gpu_numerico.json`.
- Sintaxe verificada por AST; 17 CLIs principais responderam a `--help`; GUI inicializou nove páginas e descobriu 103 itens. Estado da GUI isolado em arquivos de teste. `interface_cli_sintaxe.json`.

## Limpeza real e integridade

Comando principal: `pipeline/rodar_tudo.py ORIGINAL --auto --so-limpar --scans 20261106_210152 20261106_210505 20261106_210900 --saida-limpo NOVA`.

| Scan original | Pontos antes | Após limpeza dos frames | Após consenso de visibilidade |
|---|---:|---:|---:|
| 210152 | 13.469.478 | 10.973.434 | 10.930.254 |
| 210505 | 13.654.685 | 11.124.428 | 11.085.731 |
| 210900 | 14.230.088 | 11.956.292 | 11.920.004 |
| Total | 41.354.251 | 34.054.154 | 33.935.989 |

Consenso: `consenso_visibilidade.py LIMPO --poses poses_limpeza.json --saida NOVA --motor cuda`, parâmetros padrão (voxel/tolerância 0,5 mm, um quadro em seis). Levou 50 s; removeu 118.165 observações (0,30–0,39% por scan), sem zerar quadros. Inconclusivos foram preservados.

As poses de entrada desta comparação vieram do alinhamento previamente existente, remapeadas aos novos IDs. Isso isola a limpeza/fusão do teste independente de alinhamento inicial.

`auditar_projeto.py` aprovado entre original/limpo e limpo/consenso: FrameInfo e poses idênticos, profundidades mantidas exatas, nenhum pixel criado, integridade SQLite OK. O Merge anterior foi apenas copiado e não participa da fusão de teste. A união contém exatamente **1.543 FrameInfo e 1.543 FrameRGBD**.

Auditoria SHA-256 de todos os **57 arquivos originais**: nenhum alterado. `auditoria_final_hashes.json`.

## Falha crítica: plano de apoio após pré-limpeza

Executado o segundo passo da receita D: `limpar_por_placa.py LIMPO --altura-min 1 --voxel 1 --min-pts-voxel 3 --plano extenso`.

| Scan | Antes | Depois | Perda |
|---|---:|---:|---:|
| 210152 | 10.973.434 | 6.119.261 | 44,24% |
| 210505 | 11.124.428 | 5.364.136 | 51,78% |
| 210900 | 11.956.292 | 7.530.332 | 37,02% |

No primeiro scan o plano selecionado tem normal aproximadamente `[0,405; 0,349; -0,845]`; o plano de apoio previamente medido no scan original tem normal `[0,0562; 0,8906; 0,4512]`. São quase perpendiculares. A seleção por maior área passou a encontrar uma face da peça depois da retirada da mesa. O processo terminou com código 0 e integridade estrutural OK, embora tenha escolhido o plano inadequado.

**Esse ramo não foi usado na fusão entregue pelo teste.** É necessário estimar/preservar o plano do apoio antes da pré-limpeza e validar sua posição, orientação e recuperação em outras vistas. Integridade do dataset não comprova preservação da geometria.

## Fusão nativa, Revo e comparação 3D

- Controle CUDA sobre a entrada histórica com IDs corrigidos: **784.301 pontos em 81 s**; cobertura da entrega P7 a 0,3 mm: 99,9714%; mediana das distâncias 0,0133 mm. O caso corrigido não possui baseline registrado; estado formal `sem_base`, não uma aprovação da regressão original.
- Nova limpeza + consenso + `fundir_nitido.py --motor cuda`: **786.156 pontos em 101 s**, extensão 45,236 × 32,316 × 31,253 mm. Leitura/calibração 66,2 s, ICP 5,9 s, grades 17,2 s, mean-shift 9,3 s. Houve concorrência com outros trabalhos; não é benchmark isolado.
- Cache real: 10.930.254 observações; criação 26,206 s e reutilização 0,119 s, mesma contagem. `cache_resultado.json`.
- Revo Scan: importação da união em grupo novo `TESTE_REESCRITA_1518` funcionou; fusão a **0,1 mm em 5.560 ms**, optimizePose=0, sem divergências; **407.231 pontos**. A etapa completa de abertura/fusão levou 21,7 s. Cópia final do PLY validada após término da gravação assíncrona.
- A cópia de importação recebeu um PLY nativo válido em `fuse.ply` para evitar o problema conhecido do conversor com `fuse_file` vazio. Esse arquivo não é a referência: a referência é o novo `Fused 1108_1 (0.10mm).ply`, produzido a partir dos quadros.
- Tempos Revo e nativo abrangem trabalhos diferentes; não representam comparação direta da mesma operação. Nativo usa espaçamento final 0,07 mm e faz filtros próprios e refinamento ICP; Revo usa 0,1 mm e conserva as poses da união.

Métricas independentes calculadas com a implementação CPU de controle, extraída de `8cd1c06`, para não depender da reescrita das próprias métricas durante o ensaio:

| Comparação da nuvem nativa | Entrega P7 anterior | Nova fusão Revo |
|---|---:|---:|
| Cobertura da referência a 0,3 mm | 99,9869% | 92,0438% |
| Distância mediana nativo → referência | 0,00995 mm | 0,05158 mm |
| Distância mediana referência → nativo | 0,01002 mm | 0,04104 mm |
| Viés absoluto mediano por célula de 30 mm | 1,09 µm | 2,07 µm |

Uma componente na grade de 0,5 mm na nuvem nativa. Isso não elimina a possibilidade de sujeira encostada na peça.

**32.400 pontos do Revo (7,956%) ficam além de 0,3 mm do nativo**, e 235 pontos nativos ficam além de 0,3 mm do Revo. Estão exportados em `diferencas_so_revo.ply` e `diferencas_so_nativa.ply`; a navegação está em `comparacao_3d.html`. Não classificar automaticamente todas essas diferenças como ruído: exigem revisão da peça. Há também `inspecao_nativa.html`, consulta numérica integral e corte central de 0,5 mm. Não foi feita inspeção visual humana completa de todas as superfícies.

## Regressão e alinhamento

- Banco original com `regressao.py --incluir-caros`: P8 métricas, P8 triagem e P9 triagem aprovados; P7 falhou antes de fundir porque `poses_finais.json` usa IDs antigos e o projeto da fixture usa GUIDs novos. Caminho `data/20261106_210152/cache/frames.dataset` inexistente. Código 2.
- Apenas uma cópia da fixture foi remapeada no diretório do teste; a fusão CUDA então concluiu sem violações dos limites configurados. Não foram registradas ou substituídas bases para mascarar a falha.
- `alinhar_varios.py --escala auto` executado nos três scans: nenhum sem ligação, escala 0,17 e medianas finais de distância 0,211–0,217 mm. Porém houve discordância entre âncoras de 0,50°/2,85 mm e aviso Open3D de grafo desconectado no subconjunto de arestas certas. É uma inicialização, não homologação de alinhamento fino.
- O novo alinhamento fino completo por segmentos e sua reversão em dados reais não foram validados nesta rodada. Os testes de resposta conhecida de ICP não substituem essa etapa.

## Alterações durante o ensaio e limites da aprovação

O HEAD mudou de `8cd1c06` para `b1aec3d` durante o ensaio e módulos de alinhamento, cache, fusão, GPU e métricas continuaram sendo editados por outro processo. Nenhuma dessas alterações de produção foi feita por este teste. A lista e hashes estão na auditoria; cópias dos módulos divergentes foram guardadas em `codigo_alterado_durante_teste`.

A segunda execução da suíte e as métricas GPU posteriores passaram a demorar muito mais que a rodada inicial. Interrompidas apenas as execuções deste ensaio: suíte após 253,3 s (último teste iniciado: `test_vies_com_sinal`), métricas contra a entrega após 332,9 s e contra Revo após 281,7 s, sem JSON final. Detalhes em `encerramento_execucoes.json`. Não confundir o código de saída do wrapper PowerShell com conclusão dos testes: o log não contém o resumo final da suíte. Houve outros processos CUDA externos simultâneos; esse resultado não isola a causa nem permite certificar desempenho da nova implementação. As medições geométricas acima foram concluídas independentemente pelo controle CPU.

Prioridades antes de liberar para merge alignment: corrigir/validar a identificação do plano de apoio; reparar os IDs da fixture P7; concluir a suíte e as métricas GPU numa revisão estável; validar alinhamento fino e revisar as regiões de diferença em 3D. Não aprovar a entrega apenas por código de saída zero, contagem de pontos ou proximidade da referência anterior.
