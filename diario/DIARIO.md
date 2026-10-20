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
