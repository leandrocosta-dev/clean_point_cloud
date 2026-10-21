# Diario de bordo

Cada aprendizado vira uma entrada com: o que se observou, a evidencia (numeros), a decisao, a versao (tag git) e o que ficou em aberto.
O sistema deve ser reavaliado contra TODOS os projetos de `D:\Projects` a cada mudanca (`pipeline/aprender.py`), nunca so contra o ultimo projeto que falhou.

Gabarito objetivo: o `fuse.ply` que o proprio Revo Scan gerou (quando registra bem nos quadros, fitness >= 0,85) mostra o que e peca.
Metricas: **recall** = parte do fuse.ply que continua nos quadros limpos (perder isso estraga a malha); **lixo** = pontos mantidos longe do fuse.ply.

---

## v0.1 (base) - motor quadro a quadro
- Formato do `.revo` 1.3 decodificado (frames.dataset = SQLite com assinatura trocada; mascara de bits + profundidade por quadro; poses 4x4).
- Limpeza apaga pixels de profundidade em cada quadro, sem mexer nas poses; copia o projeto, original intacto.
- Pressuposto embutido: **existe uma mesa/base plana** (RANSAC) e a peca esta acima dela.
- Evidencia de que o pressuposto falha: busto sem mesa (`Project_Example`), o plano atravessou o corpo e cortou tudo abaixo, deixando uma ilha solta.

## v0.2 - harness de aprendizado em todos os projetos (`pipeline/aprender.py`)
- Inventario: 349 scans em 56 projetos de `D:\Projects`, ~220 com `fuse.ply`. Scan sem raw_preview (nos `Frame...`, tipo 5, estrutura de marcadores) nao e cenario de limpeza.
- `extrair` guarda por scan uma amostra leve (A: entende a cena; E: avalia) + o `fuse.ply` do Revo alinhado por ICP com inicio na identidade. Fitness >= 0,85 vira gabarito.
- `avaliar` mede variantes de pipeline sem gerar `.revo`: recall do fuse (perder isso estraga a malha) e lixo (>3 mm do fuse).
- Cuidado: o fuse do Revo pode incluir a mesa; o recall de scans assim e pessimista. Tratar com ressalva, olhar casos individualmente.

## Aprendizado 1 - o pressuposto "existe mesa plana" falha (busto sem mesa)
- Sintoma: `Project_Example`, plano de RANSAC (8% dos pontos) fatiou o busto em z=0; tudo abaixo foi cortado e sobrou ilha solta.
- Medida criada: `cruzamento_base` = fracao das celulas da peca com material DENSO tambem abaixo do plano. Mesa real ~0%; fatia da peca >=25% (busto 25%).
- Ressalva: o cruzamento sozinho NAO explica os piores cortes (ver aprendizado 2).

## Aprendizado 2 - a selecao da peca por "votos da mira" erra feio
- `20260403_085043`: escolheu o objeto de 6,6 mil pontos (11 miras) em vez do de 653 mil pontos (0 miras) que contem 94% do fuse. Recall do fuse 7,5%.
- Mesma causa provavel: `084717` do projeto 1011 (3% mantido) e varios scans do 1011.
- Hipotese: o maior componente deve vencer quando domina a cena; os votos so desempatam entre candidatos comparaveis ("hibrido"). Testado em `aval_02.json`.

## Resultado da avaliacao 02 (112 scans com gabarito, `aval_02.json`)
| variante | recall medio do fuse | p5 | scans que perdem >10% | lixo |
|---|---|---|---|---|
| plano + votos (regra v0.1) | 62,6% | 4% | 86 | 6,7% |
| plano + maior componente | 67,4% | 18% | 84 | 6,7% |
| plano + hibrido (dom. 0,5) | 65,9% | 15% | 86 | 6,9% |
| auto (cruzamento<=0,10) + hibrido | 70,6% | 15% | 69 | 7,1% |
| **sem base + hibrido** | **95,8%** | **80%** | **14** | **3,1%** |
- A regra de v0.1 corta >10% do que o Revo fundiu em 86 de 112 scans. Os bons resultados dos primeiros projetos (1017, 0924) eram a excecao (mesa giratoria bem definida).
- `perdas_plano.json`: da perda da regra v0.1, em media 25,3% e "na faixa do plano" e 5,8% e "acima do plano". Muitos scans (todos os `golf`, `20260927_2218xx`) tem o fuse INTEIRO rente ao plano: a peca e achatada/baixo-relevo e o RANSAC pegou a propria peca como mesa. Apagar o plano apaga o scan.
- 21 scans perdem >10% ACIMA do plano (5 perdem >50%): selecao errada do componente.
- O cruzamento do plano sozinho NAO separa mesa de peca achatada (auto_0.10 melhora pouco). Precisa de mais sinais.

## Aprendizado 3 - dois gargalos de velocidade (v0.3)
- `np.linalg.svd(..., full_matrices=False)` no RANSAC do plano: a versao completa montava uma matriz N x N (13 GB para 40 mil pontos). Ganho: de minutos para 0,07 s. Vale tambem para a limpeza de producao.
- Transformada de distancia 3D so numa caixa em volta da peca (grade da cena pode ter centenas de milhoes de voxels). Resultado identico (conferido: mesmo % mantido), de minutos para 2-8 s por variante.
- Cache de amostras: 321 scans, 7 GB (`aprendizado/cache`, fora do git). 9 scans nao tem nenhum quadro com profundidade e ficam de fora.

## Aprendizado 4 - nenhuma caracteristica da cena decide "o plano e mesa" (`caracteristicas.json`, `regra_plano.py`)
- Extraidas 16 caracteristicas por scan (fracao do plano, cruzamento, achatamento h99/extensao, razao de celulas plano/peca, altura das miras etc.) dos 321 scans.
- AUC para prever "plano inofensivo" (recall do plano >= recall sem base - 2%): melhor 0,72 (h01), `achatamento` 0,70; `cruzamento` so 0,30 (invertido). Nenhuma separa bem.
- Busca de regra com limiar unico e penalidade pesada por perder peca, com validacao leave-one-project-out: a melhor regra e **nunca usar o plano** (saldo 0; usar sempre = -650).
- Quando o plano e inofensivo (22 de 112 scans) ele limpa so 0,28% de pontos a mais em media (max 36%, mesas giratorias grandes). Ganho pequeno, risco alto.
- Decisao: o padrao passa a ser **sem base + selecao hibrida**. O plano nao decide mais nada sozinho. Tentativa seguinte (aval_03): usar o plano apenas como "periferia da mesa" (faixa do plano longe da peca) e so se a peca tem altura real acima dele.
- Hipotese de causa da falha das caracteristicas: a mesma geometria (grande plano + peca em cima) aparece em mesa giratoria e em peca achatada sobre bancada; so o contexto de captura (rotacao da mesa, movimento da camera) diferencia. Ideia futura: usar o movimento das poses (camera orbitando a peca) como sinal.

## Aprendizado 5 - o plano nao pode decidir nada; "periferia da mesa" tambem nao (`aval_03.json`)
- Melhor regra por caracteristica = nunca usar o plano (ver aprendizado 4). Padrao novo: **sem base**.
- Teste da "periferia da mesa" (apagar a faixa do plano longe da peca, so quando a peca tem altura): recall cai de 95,8% para 86,7%..93,3% em todas as configuracoes (saldo sempre negativo).
- Ressalva importante: o `fuse.ply` do Revo inclui a base plana da peca (ex.: placa com cupulas do `194310`); o gabarito NAO consegue validar remocao de mesa. Decisao: remocao de mesa fica como opcao explicita (`periferia_mm`), desligada por padrao. O padrao so remove o que esta solto, longe ou isolado.

## Aprendizado 6 - a peca aparece partida em varios componentes grandes (`aval_04.json`)
- Causa dos piores cortes do modo sem base (`164145`, `golf/183319`, `180335`): amostra esparsa e fendas partem a peca; o codigo so mantinha componentes com voto da mira. O fuse tinha 664x606x349 mm e o mantido 289x237 mm.
- Correcao: `grandes_frac` = todo componente com >= x do maior entra, com ou sem voto.
| grandes_frac | recall | p5 | scans <90% | lixo |
|---|---|---|---|---|
| 0 (antes) | 95,79% | 80% | 14 | 3,07% |
| **0,02** | **98,76%** | **93%** | **4** | 5,02% |
| 0,05 | 97,98% | 88% | 8 | 4,00% |
| 0,10 | 97,78% | 87% | 10 | 3,46% |
| 0,35 | 96,75% | 84% | 12 | 3,23% |
- Escolhido 0,02: perder peca estraga a malha; sobra solta se limpa depois. Custo: mais pontos soltos mantidos (lixo 3,1% -> 5,0%).

## Aprendizado 7 - busto sem mesa (caso reportado) resolvido
- `Project_Example` com os padroes novos: 100% dos pontos mantidos (so saem pixels voadores e a ilha de ruido abaixo do busto). Antes (v0.1) cortava tudo abaixo do plano.

## Estado: v0.5 - padrao = sem base + selecao hibrida (dominancia 0,5) + componentes grandes (0,02) + filtro de pixels + ilhas 2%

## Sessao 2026-10-19/20 (projeto `Project20260613_160228`, painel de carro) - o que NAO repetir
- LER ESTE DIARIO E `git log` ANTES DE QUALQUER INVESTIGACAO. Uma sessao redescobriu do zero o formato do frames.dataset (SQLite com magic `RPAlgo_file_sys`, mascara LSB + floats) que ja estava na v0.1 e perdeu horas.
- O usuario exige limpar **nos quadros** (`frames.dataset`) e entregar o `.revo` completo. Limpar `fuse.ply`/`fuse_mesh.ply`/`raw_preview.ply` nao serve: o Revo Scan abriu a copia e mostrou a nuvem igual a original (ele refunde dos quadros e nao grava nada na pasta). `build_revo_project.py` e `clean_revo.py` (so .ply) sao beco sem saida para a entrega; nao usar.
- Ambiente: CloudCompare nao instalado (`Clean-PointCloud.ps1` nao roda). 24 threads. `scatter` do matplotlib em nuvem grande estourou a memoria e travou o Claude Code (Bun) 2x: usar `histogram2d`.
- Rodar: `.venv/Scripts/python.exe -I pipeline/limpar_raw.py <projeto> --saida <pasta nova>`.

## Aprendizado 8 - remocao de mesa nos quadros (`--periferia-mm`), com auditoria independente do fuse (2026-10-20, projeto `Project20260613_160228`)
- Sintoma: padrao v0.5 deixa a mesa inteira (95-99% dos pontos mantidos). Nos scans 161826/162307 o "objeto" da peca media 724x680 mm e 785x556 mm, mas a peca tem ~320x300 mm: o componente da peca vem COLADO na mesa.
- Causa de a 1a tentativa de periferia falhar (so ~20% a menos): a pegada era calculada com todos os voxels do componente acima de 4 mm do plano; relevo e marcadores da mesa entravam na pegada e a protegiam. Correcao: pegada = so o CORPO (h > `per_altura` 15 mm) + relevo baixo ate `per_alargue` mm do corpo (cobre base alargada), com abertura morfologica; corte vale por ULTIMO (depois da referencia do fuse, que inclui a mesa e a ressuscitaria); trava: so quando o plano NAO atravessa a peca (cruzamento <= 0,10).
- `pipeline/auditar_limpeza.py` compara original x limpo pixel a pixel (sem fuse.ply) e mede o que foi removido por altura acima da mesa, os removidos dentro da coluna vertical do corpo mantido e a distancia 3D deles ate a superficie mantida.
- ARMADILHA da auditoria: `RevoScan.pontos(k, passo=N)` escolhe indices dentro da lista de pixels VALIDOS; apos a limpeza a lista encolhe e a amostra desloca (parecia que 50% do corpo foi cortado, era artefato). Comparar sempre com passo=1.
- Resultado (periferia 8 mm, banda 12, alargue 20): removido acima de 15 mm = 0,8-0,9% (so filtros de pixel voador padrao); a mais que a v0.5 acima de 15 mm = 4.141 e 3 pontos, zero dentro da coluna do corpo; mesa ate 4 mm: 85% removida; sobra colar de ~25-30 mm.
- Risco NAO medido: o Revo Scan pode precisar da geometria da mesa/marcadores para registrar/fundir (ver docstring de `limpar_fuse`). So o usuario testa abrindo o `.revo` e fundindo.
- Rodada final (`--periferia-mm 5 --periferia-banda 12 --periferia-alargue 10`, projeto inteiro): 160310 manteve 94,4% (periferia DESLIGADA pela trava: cruzamento 24%; extensao final 321x304x261 mm = a peca); 161826 56% (335x347x232); 162307 44% (365x301x286). Auditoria: acima de 15 mm a mais que a v0.5 = 0 / 4.141 / 3 pontos; 4.141 sao 0,1% do corpo, a <=1 mm da superficie mantida (cobertura duplicada, geometria preservada).
- Comando de producao usado: `python pipeline/limpar_raw.py <projeto> --periferia-mm 5 --periferia-banda 12 --periferia-alargue 10 --saida <pasta>` e depois `python pipeline/auditar_limpeza.py <original> <limpo> <scan> --referencia <limpeza v0.5>`.
- Em aberto: (1) o Revo Scan fundir bem sem a mesa/marcadores; (2) testar em outros projetos de `D:\Projects` (so 1 projeto validado); (3) tornar a periferia padrao so depois de rodar `aprender.py` com a nova regra; (4) origem dos 4.141 pontos de corpo a mais removidos em 161826.

### Validacao no Revo Scan (2026-10-20)
O usuario abriu `Project20260613_160228_limpo_sem_mesa` (limpo nos frames com `--periferia-mm 5 --periferia-banda 12 --periferia-alargue 10`), refez a fusao e a malha saiu perfeita. Ou seja: o Revo funde bem sem a mesa e sem os marcadores neste projeto. Validado em 1 projeto (3 scans); falta rodar `aprender.py` nos demais antes de tornar a periferia padrao.

### Aprendizado 9 - amostra da periferia em outros projetos (2026-10-21)
`pipeline/avaliar_periferia.py` compara, sobre o cache de `aprender.py`, limpeza padrao x padrao+periferia (5/12/10), sem usar o fuse.ply (que inclui a mesa). Armadilhas:
- A altura da mesa tem de usar `periferia["n"]/["d"]` (d_ref), NAO `seg["d"]`: com o d errado a mesa aparece a 83-100 mm e o "corpo perdido" sai 50% (falso).
- "Corpo perdido acima de 15 mm" e 0 por construcao (a regra so mata |h| <= banda). A metrica que vale: removidos DENTRO da projecao vertical do corpo mantido.
- Falha real achada: scan de peca achatada (Project20260605_141933/20260605_142220): a regra tirou 93% (dentro do corpo 39%). Correcao: trava `per_corpo_frac` (0,10) em `segmentar_peca`: so liga se >=10% do volume mantido passa de `per_altura` (15 mm) acima do plano. Nesse scan passou a manter 99%.
- Resultado parcial com a trava (18 de 41 scans, amostra uniforme): periferia ligou em 10; removidos dentro do corpo: 0 a 199 pontos (<=0,027% do corpo) em todos; nenhum corte grande. Rodada interrompida por memoria (um worker chegou a 36 GB: grade enorme em cenas de ~1 m; ver pendencia).
- Memoria: rodar no maximo 2 jobs; `ProcessPoolExecutor(max_tasks_per_child=1)`. Scans de cena grande (1 m+) estouram RAM em `segmentar_peca`.
- Pendente: concluir os 23 scans restantes da amostra; investigar o estouro de memoria da grade; scan Project20260613_143330/144748 (peca baixa, 336x286x66 -> 270x286x46 mm, 33% mantidos) merece olhar visual.
- Reteste do projeto Project20260613_160228 com a trava `per_corpo_frac` (v0.6.1): contagem de pontos mantidos IDENTICA a entrega validada no Revo nos 3 scans (65.643.732 / 36.689.519 / 15.188.560). A trava nao muda o caso bom.

### Aprendizado 10 - memoria e amostra completa da periferia (2026-10-21)
- Estouro de RAM: a grade da peca (voxel 1 mm) em cena de ~1 m passa de 1e9 voxels (36 GB num processo). Correcao em `segmentar_peca`: `MAX_VOXELS` (3e8) e `MAX_VOXELS_TRANS` (1e8); o voxel cresce x1,25 ate caber e loga "cena grande". Scan que levava 4+ min e >17 GB: 84 s e ~5 GB, resultado equivalente (dentro do corpo 141 vs 199 pontos).
- Amostra uniforme de 41 scans de 12 projetos (`avaliar_periferia.py --limite 40`, `aprendizado/periferia_01.json`): 0 erros; periferia ligou em 25 e a trava a desligou em 16 (plano atravessa a peca ou peca achatada). Removidos DENTRO da projecao do corpo: mediana 0,002%, p95 0,24%, max 0,36% (Project20261011_165456/20261011_184157, peca baixa ~130 mm); nenhum acima de 0,5%.
- Periferia continua opt-in (`--periferia-mm`): so 1 projeto foi fundido no Revo (malha perfeita). Tornar padrao depois de mais um teste de fusao pelo usuario em peca diferente.

## Automacao do Revo Scan 6 (2026-10-21) - `pipeline/revo_ui.py`
Objetivo: rodar a pipeline inteira (limpar -> importar -> fundir -> malha -> exportar) por codigo.
**Nao tentar de novo**: nao ha CLI nem API. `RevoScan6.exe <arquivo>` so aceita `.revox` (com `.revo` mostra "Cannot directly open non-revox extension files"). As DLLs de algoritmo (`RPAGlobalFusion.dll`, `RPAMeshRecon.dll`, ...) exportam C++ mangled (std::shared_ptr, Eigen, std::function): nao chamaveis por ctypes sem headers/shim MSVC. Strings do exe: `QCommandLineParser`, `QLocalServer` (instancia unica), `BatchProcessList` (so UI).
**O que funciona**: UI Automation (Qt expoe a arvore com automation ids `RvApplication.*`), pywinauto no `.venv` (`pip install pywinauto pillow`).
- Biblioteca interna do Revo: `%APPDATA%/Revopoint/RevoScan6/Projects/<guid>/` (um scan por pasta, .revox). `File > Load Project` (dialogo nativo #32770, filtro `*.revox *.revo`) IMPORTA os scans do .revo (copia os dados; ~3 s). Os resultados da fusao/malha ficam gravados na biblioteca, nao na pasta de origem.
- Fluxo medido (scan de 383 quadros, ~1 M de pontos): abrir scan 5-7 s; Fusion>Apply 6-9 s (`Fused <data>_n (0.41mm)`); Meshing>Apply 10 s (`Mesh <data>_n`); Export Model 2 s -> `.ply`.
- Sinais de fim: nova linha na lista do modelo (ListItem) para fusao/malha; janela `RvMessageBox` "Model Export Complete" (fica FORA da janela principal; bloqueia cliques ate Confirm).
- Armadilhas: (1) o `.invoke()` de botoes Qt nao faz nada, usar `click_input()`; (2) paineis Fusion/Meshing sao QMenu cujo conteudo nao esta na arvore UIA: clicar o Apply por posicao (Fusion 497,324; Meshing 950,433 relativos a janela maximizada 1918x999); (3) os cartoes da biblioteca nao tem texto na UIA: filtrar pela caixa de busca (QLineEdit) e dar duplo clique no 1o cartao; (4) botoes de dialogos nativos em Windows pt-BR: "&Abrir", "Selecionar pasta"; `.click()` do win32 nao age no seletor de pasta, usar `click_input()`; (5) o mouse real e usado: nao mexer no PC durante a execucao.
- **Armadilha de dados**: o .revo limpo copia `fuse.ply`, `fuse_mesh.ply` e `mark.ply` ORIGINAIS (com a mesa). Ao importar, o Revo mostra esses itens como `fuse`/`fuse_mesh` ao lado dos novos, e o Export Model os lista tambem. Primeira exportacao automatica pegou o `fuse_mesh` antigo (724x733x438 mm, mesa incluida) em vez da malha nova (332x343x225 mm, so a peca). `exportar()` agora marca so os itens "Fused*"/"Mesh*" pelo nome. Para entregar limpo, convem nao copiar esses .ply antigos no `limpar_raw` (nao alterado ainda).
- Teste de ponta a ponta no scan 20260613_161826: fusao 9 s, malha 10 s, nuvem 1.219.987 pontos (332x343x225 mm), malha 1.217.681 vertices / 2.409.932 faces, bate com o painel do Revo.
- Efeito colateral: a biblioteca do usuario ganhou 3 scans importados (135 -> 138) e dois deles ganharam fusao/malha.
- Pendente: expor parametros (distancia de pontos 0,41 mm, Remover marcadores/Isolados, qualidade da malha) no modulo; orquestrador `limpar_raw` -> `revo_ui` por projeto; lidar com muitos scans na lista (busca por nome ja cobre); detectar falhas/modais inesperados.

### Parametros de fusao/malha sem mouse e orquestrador (2026-10-21) - `revo_ui.parametros`, `pipeline/revo_pipeline.py`
- **Os valores do painel vem do `.revox` do scan na biblioteca**, bloco `param_template.postProcessParam` (o `postProcessParamQuick` e ignorado). O Revo le o arquivo na 1a vez que carrega o scan na sessao; reabrir o mesmo scan na mesma sessao reaproveita o estado (gravar de novo nao vale). O `.revo` de origem nao tem `.revox`: ele nasce na importacao, entao grava-se depois de `Load Project` e antes de abrir o scan, sem reiniciar o Revo.
- Mapeamento verificado (log + captura do painel): `pointPitch` (+`isPointPitchSetted=true`, senao o Revo recalcula o automatico) = Point Distance; `meshMode` 0 = Fast Meshing, 1 = High-Quality (Poisson); `optimizeLevel` 0/1/2 = None/Low/High (so no modo rapido; o Revo regrava 0 no modo alto); `meshEdgeLength` (+`isMeshLenSetted`) = Grid Size (so no alto); `enableAutoFillHoles` = Hole Filling; `enableClearMarker` = Remove marker points; `enableExtractIsolation` = Remove Isolation (ligado confirmado; desligado so pelo log nao foi isolado).
- **Fim de etapa e conferencia pelo log** (`%LOCALAPPDATA%/Revopoint/RevoScan6/logs/RevoScan6.log`): `point cloud fuse with pointPitch: X` / `point cloud fuse end, duration`; `meshingHelper] - meshMode:..optimizeLevel:..fillHoleEnabled:..meshEdgeLength:..` / `triMesh meshing end, duration, result: 0`; `Project opened:<pasta>\<guid>.revox` (diz qual copia abriu). A lista de modelos e virtualizada (so linhas visiveis na arvore UIA): contar linhas deixa de funcionar com muitos itens.
- Painel muda de altura (Hole Filling, marcadores, modo da malha): o Apply e achado na captura da janela (retangulo azul 0,136,255, ~200x28 px), nao por posicao fixa. Meshing so habilita com nuvem fundida selecionada (a fusao seleciona a nova).
- Biblioteca tem varias copias com o mesmo guid (importacoes repetidas): `carregar_projeto` devolve a pasta das copias novas e `abrir_scan` confere a pasta pelo log. Cada importacao acrescenta pastas na biblioteca do usuario.
- Armadilhas: dialogo de arquivo as vezes nao fecha com o clique (ler o texto de volta, tentar `click` e `click_input`); caixa de erro tambem se chama "Load Project" (botao OK); `taskkill` sem /F nao fecha o Revo com scan aberto, `principal.close()` fecha; fusao de scan com `can_fuse_optimize` usa otimizacao de pose e leva ~50 s.
- Teste (projeto limpo, 3 scans, 0,5 mm / malha alta grade 0,5 / furos / marcadores): `Fused ... (0.50mm).ply` + `Mesh ... (0.50mm).ply` nos 3, sem divergencia entre pedido e usado. Um scan falhou ao abrir quando o Revo ja o tinha aberto de uma execucao anterior; sozinho passou.
- Ainda nao feito: encadear `limpar_raw` -> `revo_pipeline`; `limpar_raw` ainda copia `fuse.ply`/`fuse_mesh.ply`/`mark.ply` antigos; cliques ainda usam o mouse real (abrir, Apply, exportar); janela maximizada e rotulos pt-BR.

### Encadeamento limpeza -> Revo (2026-10-21) - `pipeline/rodar_tudo.py`
- `limpar_raw --sem-ply-antigos` nao copia `fuse.ply`/`fuse_mesh.ply`/`mark.ply` (traziam a mesa). Verificado: o Revo importa o scan sem esses arquivos e funde/gera malha normalmente (2 de 3 scans no teste; o 3o falhou por causa da busca, abaixo).
- `rodar_tudo.py` roda `limpar_raw` e depois `revo_pipeline` no `.revo` limpo.
- Falha achada e corrigida: a busca da biblioteca demora a filtrar; clicar antes abre o 1o cartao da lista inteira (scan errado). `abrir_scan` agora espera `Scan(N) >= 1`; a conferencia da pasta pelo log ja barrava o erro.
- Revo travou ("Nao responde", 2,3 GB) depois de horas de sessao: a pipeline reinicia o Revo a cada scan e `RevoUI.abrir` recusa um Revo travado (IsHungAppWindow). Reinicio por scan verificado (3 de 3 scans, Revo fechado no fim); a deteccao de travamento nao foi exercitada de novo.
- **"There is no active desktop required for moving mouse cursor" nao era tela apagada nem bloqueio**: a sessao estava ativa, sem LogonUI. O primeiro plano era o Gerenciador de Tarefas (elevado) e janela elevada em foco bloqueia o mouse de processo sem elevacao (UIPI): `SetCursorPos` e `mouse_event` nao movem o cursor e `SetForegroundWindow` nao troca o foco. Solucao que funcionou: `Shell.Application.MinimizeAll()` (equivale a Win+D). `revo_ui.manter_tela_ativa()` faz isso (alem de `SetThreadExecutionState` e 1 px de mouse para acordar o monitor) e `restaurar_janelas()` desfaz; `RevoUI.abrir` chama a primeira.
- Teste final do encadeamento: limpeza dos 3 scans (`--sem-ply-antigos`, periferia 5/12/10) + Revo (0,5 mm, malha alta grade 0,5, furos): 3 de 3 com `Fused`/`Mesh` `.ply` exportados e sem divergencia entre pedido e usado.
- **Lingua do Revo = ambiente de quem o inicia** (QLocale; o ambiente da sessao tem `LANG=en_US.UTF-8` e o Revo abriu em `en_US` o dia todo; uma execucao sem ele abriu em `pt_BR` e a automacao nao achou "Load Project"). `RevoUI.abrir` agora inicia o Revo com `LANG`/`LC_ALL=en_US.UTF-8` e `biblioteca()` avisa se encontrar um Revo ja aberto em portugues.
- Projeto novo `Project20261017_204257` (2 scans pequenos, peca ~50x78x68 mm, sem periferia, parametros automaticos do Revo): limpeza 95% mantidos nos dois, Revo fundiu a 0,10 mm (automatico) e gerou malha rapida/Low; `.ply` exportados, sem divergencia. Sem `--ponto` o Revo usa o passo automatico, que em pecas pequenas e o minimo (0,10 mm) e gera arquivos grandes (27 MB a nuvem).

## Uniao de scans da mesma peca (2026-10-21) - `pipeline/uniao_scans.py`
**CORRECAO (mesmo dia): o resultado descrito abaixo como "validado" estava ERRADO.** O usuario viu no Revo as duas nuvens do mesmo lado; no projeto a peca foi escaneada de cima e depois de baixo (peca virada). Metricas boas (fitness, rmse, cobertura, espessura local, "todo ponto explicado por A ou B") NAO detectam isso: valem para as duas solucoes simetricas. Ver "Correcao" no fim desta secao; as medidas e a abordagem por frames continuam validas, o criterio de escolha do alinhamento mudou.
Caso: `Project20261017_204257`, a peca foi escaneada em duas partes (scans `204349` e `204711`).
- **Abordagem**: unir no nivel dos frames, nao nas malhas. Um scan novo com os 500 frames de A + os 499 de B, cada pose de B multiplicada por T (B->A). O Revo funde tudo de uma vez (sem costura de malhas) e o fluxo `revo_pipeline` segue igual.
- Formato: `FrameInfo` so usa a coluna `bin` (as outras ficam NULL). `bin` = 16 B de cabecalho + pose 4x4 float64 (camera->mundo, linha a linha) + cauda de marcadores em coordenadas da CAMERA (registros de 8 float32: posicao, normal, qualidade, diametro): a cauda nao muda com T. `FrameRGBD` e copiado com `id` deslocado (maxA + 2). Calibracao (`param/*.bin`) das duas partes era identica (conferir antes de unir). `raw_preview.ply` = A + B transformada (posicao e normal).
- Alinhamento: nuvem tirada dos proprios frames limpos (nao do `.ply` exportado, para nao depender do referencial), FPFH+RANSAC (5 sementes) + ICP ponto-plano, depois REFINO em 0,5/0,25/0,15 mm: o ICP a 1 mm deixava 0,65 mm de erro e errava ~2 graus / 6 mm; apos o refino rmse 0,14 mm. O criterio de confianca usa as sementes ANTES do refino (0,03 graus / 0,12 mm) e a cobertura nos dois sentidos (87% de B em A, 76% de A em B a 2 mm); o programa recusa gravar se as sementes discordam ou a cobertura e <50%.
- Resultado no Revo (passo 0,2 mm, malha rapida): fusao 4 s, Fused 330 mil pontos, extensao 48,4x77,6x70,5 mm (uniao esperada 48,7x77,5x71,3; A sozinha 48,7x77,7x67,6). Todo ponto da nuvem fundida fica a <0,5 mm de A ou de B: 24,6% so A, 59,4% ambas, 16,0% so B, 0% nenhuma. Espessura local (plano em 40 vizinhos, voxel 0,3): A 11 um mediana, B 8, fundida 19 (p99 249 um), A+B empilhadas SEM fundir 53 (assinatura de desalinhamento): o Revo fundiu em camada unica. Ressalva: fundida a 0,2 mm e as partes a 0,1 mm, comparacao nao e 100% justa. Visualmente a face que tinha falha de dados em A saiu completa (B preencheu).
- O Revo aceitou o scan unido sem mexer no cabecalho dos frames (`hdr0` do 1o frame de B continua 0) e sem reclamar dos marcadores de B; a otimizacao de pose do Revo (`can_fuse_optimize`) nao foi isolada, so o resultado final foi medido.
- Limites: validado em 1 par; exige sobreposicao real entre as partes (aqui ~80%). Peca com pouca sobreposicao ou simetrica pode alinhar errado: ai o recurso certo e alinhamento manual/por marcadores no Revo.
- Armadilha: o Revo aberto pelo usuario (pt_BR, com scan aberto) foi tomado pela automacao; `RevoUI.abrir` agora recusa sessao fora do ingles.
- **Correcao: simetria da peca.** Peca em disco (46,6 mm de diametro, ~40 mm de espessura, aba lateral): cada scan ve a PROPRIA face de extremidade + o corpo lateral + a aba. A area em comum (corpo+aba) e simetrica por giro de 180 graus em torno do eixo do disco, entao existem duas solucoes quase iguais: "faces no mesmo lado" (planos das faces coplanares: z -11,7 e -11,8 mm) e "peca virada" (planos a 38-39 mm um do outro). O ICP preferiu a errada (mais inliers: 0,65-0,69 contra 0,42-0,60 a 0,3 mm, porque as faces quase planas coplanares tambem contam) enquanto a certa tem erro menor (0,087-0,114 contra 0,133-0,142 mm). Fitness e rmse nao decidem.
- `alinhar` agora gera candidatos (RANSAC + 4 giros de 180 graus dos eixos principais), refina cada um (1/0,5/0,25/0,15 mm), mede a separacao entre o maior plano de A e o de B alinhada e classifica `mesmo` (<1 mm) / `opostos` (>40% da extensao) / `intermediaria`; `--lados opostos|mesmo|auto` (padrao `opostos`). Aborta se nenhum candidato com bom encaixe tem a relacao pedida. A informacao "peca virada" vem de quem escaneou; nao se deduz so da geometria.
- Resultado corrigido (`pca-y`, separacao 39,2 mm, rmse 0,114): Revo fundiu em 4 s; a nuvem fundida bate com A+B alinhadas (ICP fitness 1,00, rmse 0,065 mm, deslocamento 0,02 graus / 0,05 mm: o Revo manteve o alinhamento); A e B 100% cobertas; nenhum ponto longe das duas. Nos eixos da peca: paredes laterais coincidem (cilindro de A centro (5,33; 1,74) raio 23,55; de B centro (5,37; 1,64) raio 23,49), uniao 73,6 x 49,1 x 42,9 mm (A 38,6; B 41,5 ao longo do eixo). Vista de um lado a nuvem mostra a face de A, do outro a face de B. A caixa orientada (OBB) enganou: nao usar para comparar uniao.
- Pendente: o usuario validar a fusao/malha no Revo (aberto o `.revo` unido).

