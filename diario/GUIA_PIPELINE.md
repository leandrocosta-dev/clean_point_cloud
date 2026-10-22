# GUIA DO PIPELINE - leia isto PRIMEIRO (atualizado em 2026-10-22, tag v1.9+)

Objetivo do usuario: a partir dos scans do Revo Scan 6 (`frames.dataset`), entregar um projeto `.revo` com cada frame limpo
e, quando a peca foi escaneada em varios scans, uma peca COMPLETA, alinhada e limpa (hoje: engenharia reversa no NX).
Historico detalhado, numeros e erros em `diario/DIARIO.md`; este guia e a receita e a lista do que NAO repetir.

## 0. Regras do usuario (nao quebrar)
- Limpar no nivel dos FRAMES (`frames.dataset`), nunca em `.ply`; o Revo refunde dos quadros. Entregar `.revo` completo.
- Nunca tocar no projeto original (D:\Projects\...): sempre copia em pasta nova (os scripts recusam destino existente).
- Responder em portugues. Anotar aprendizados no diario do repo sem duplicar. Nao perguntar; checar e testar por conta.
- Usar a maquina toda, MAS ver secao 8 (instabilidade): prioridade baixa e <= 12 threads.
- "Nao alucinar": toda afirmacao com medida; dizer o que nao foi verificado.

## 1. Mapa dos scripts (`pipeline/`)
| script | para que serve |
|---|---|
| `limpar_raw.py` | limpeza por quadro (pega a peca por marcadores/geometria, tira pixels voadores e ilhas). `--sem-ply-antigos` (nao copia fuse/mesh/mark.ply com a mesa), `--periferia-mm 5 --periferia-banda 12 --periferia-alargue 10` (remove mesa; opt-in), `--jobs`. Pula scan com erro de calibracao > 2,5x espacamento (copia intacto) |
| `revo_ui.py` | automacao do Revo por UI Automation (pywinauto): `RevoUI.abrir/carregar_projeto/abrir_scan/fundir/malha/exportar/fechar`, `parametros()` (grava no `.revox`), `manter_tela_ativa()` |
| `revo_pipeline.py` | importa um `.revo`, grava parametros, funde, gera malha e exporta por scan (`--ponto --malha rapida|alta --grade --furos --marcadores --manter-isolados --reusar --so-exportar`) |
| `rodar_tudo.py` | cadeia: limpar_raw -> (`--unir A B --lados opostos`) -> revo_pipeline. Para 1 scan ou 1 par |
| `uniao_scans.py` | une 2 scans (alinhamento por candidatos + `--lados`) ou N scans com poses por segmento (`unir_varios`, `copias`, selecao) |
| `alinhar_varios.py` | alinhamento INICIAL de N scans por crescimento com ancoras (FPFH+RANSAC isolado, `isolado()`), saida `poses_iniciais.json` |
| `alinhar_segmentos.py` | segmentos de N quadros -> centroides de 1 mm -> lacos ICP robusto -> grafo de poses (so como partida; o bom e `alinhar_direto`) |
| `alinhar_direto.py` | OTIMIZACAO CONJUNTA ponto-a-plano de todos os segmentos (o que realmente alinha); varias etapas com `--tol-ini/--tol-fim` |
| `avaliar_segmentos.py`, `avaliar_por_scan.py` | vies normal por celula de 30 mm (unica metrica de alinhamento valida) |
| `limpar_por_consenso.py` | limpeza dos quadros por consenso entre scans ALINHADOS (apoio >= 2 outros scans e <= 15 mm do nucleo) |
| `remapear_segmentos.py` | refaz k0/k1 e centroides quando a limpeza apagou quadros |
| `cobertura.py`, `selecionar_segmentos.py` | ordem gulosa de construcao e conjunto minimo de segmentos que cobre a superficie SOLIDA |
| `montar_pecas.py` | monta o scan unico (frames dos segmentos escolhidos, pose do segmento) + `.revo`; `--selecao`, `--copias` |
| `avaliar_fusao.py`, `faltando.py`, `buracos.py` | avaliar a fundida; o que dos segmentos NAO esta na fundida (`faltando.py` e o util; `buracos.py` por contorno nao serve p/ casca aberta) |
| `fundir_proprio.py` -> `refinar_nuvem.py` -> `uniao_revo_propria.py` -> `malha_poisson.py` | fusao propria completa, suavizacao, uniao com a nuvem do Revo (cinza/laranja), malha de Poisson |
| `confianca.py` | mapa de confianca: em quantos scans independentes cada ponto se apoia |
| `limpar_fragmentos.py` | tira componentes isoladas pequenas (chapa solta a 82 mm) da nuvem e das malhas; `lacunas.py` = relatorio/imagem dos contornos abertos da malha |

Pre-requisitos: `.venv` do projeto (open3d 0.20 CPU, numpy, scipy, pywinauto, pillow, psutil, matplotlib). Scripts com
ProcessPool exigem `if __name__ == "__main__"`. Revo: `C:\Program Files\Revo Scan 6\RevoScan6.exe` 6.4.1.

## 2. Receita por tipo de projeto
**A. Um scan:** `python pipeline/rodar_tudo.py <proj> --saida-ply PASTA [--periferia-mm 5 --periferia-banda 12 --periferia-alargue 10] --ponto 0.1 --malha alta --grade 0.1 --furos`
**B. Duas partes da mesma peca (virada de cima/baixo):** igual + `--unir A B --lados opostos` (so o usuario sabe se foi virada: `mesmo` se so girou sobre a mesa). Validado pelo usuario (difusor).
**C. Muitos scans da mesma peca (Onix, 20 scans), ordem exata:**
1. `limpar_raw.py <proj> --saida X_limpo --sem-ply-antigos` (cenario pouco: remove 1-16%).
2. `alinhar_varios.py X_limpo --saida X_alinh --scans <todos menos de outra calibracao>` -> `poses_iniciais.json` (~25 min).
3. `alinhar_segmentos.py ... --quadros 15` (so para gerar `segmentos_indice.json` + centroides; 1 rodada basta) e depois `alinhar_direto.py <pasta_seg> --poses ... --saida direto*.json` em 3 etapas: (6->1,2 mm, 14 it), (3->1, 24 it, sigma-odo 1,0, amostra 20000), (2->0,8, 30 it, k-por-no 6). Vies caiu 367 -> 114 um.
4. `limpar_por_consenso.py X_limpo <pasta_seg> direto.json --saida X_limpo2` (84% dos pontos) -> `remapear_segmentos.py` (novos centroides) -> `alinhar_direto.py` mais uma etapa (vies 104 um).
5. Fundir: `selecionar_segmentos.py ... --redundancia 1 --apoio-min 12` -> `montar_pecas.py --selecao` -> `revo_pipeline.py --ponto 0.4`. Fusao Revo: ~12 M pontos, satura; perde ~15% da superficie.
6. Completar: `fundir_proprio.py X_limpo2 direto.json --saida F --voxel 0.4` -> `refinar_nuvem.py F` -> `uniao_revo_propria.py <fused_revo.ply> F/nuvem_refinada.ply --dist 2.5` -> `malha_poisson.py nuvem_completa.ply --profundidade 11` -> `confianca.py`. Entrega: pasta com LEIA-ME.md (ver `Onix_parachoque_traseiro_ENTREGA`).

## 3. NAO REPETIR (custou horas)
- Alinhamento com ICP de scan inteiro e escolher so por fitness/rmse: peca simetrica tem solucao espelhada (difusor: faces no mesmo lado). Gerar candidatos + medir separacao dos planos das faces + `--lados`.
- Metricas enganosas: distancia ponto-a-ponto/fitness a 0,3-0,5 mm (mede espacamento e ruido de ~0,3-0,5 mm por ponto bruto); espessura local por PCA (cresce com o raio); consistencia 3D em mm RMS do grafo (ICP desliza em superficie lisa). Use SO o vies por celula (mediana COM SINAL por celula de 30 mm) e cortes ampliados.
- Grafo de poses do Open3D: arestas CERTAS precisam formar grafo conexo, senao o resultado piora ("Certain-edge subset ... not connected"). O refino por pares fica longe do otimo; usar `alinhar_direto`.
- RANSAC em todos os pares (190) nao escala e rodar em paralelo nao ajuda (CPU saturada): crescimento por ancoras, cada registro em processo isolado com repeticao.
- Nuvens de pontos BRUTOS dos frames sao ruidosas: promediar em voxels (centroides) antes de medir/alinhar.
- Revo reotimiza poses na fusao se o scan tem `scan_mode 0`: no scan montado usar `scan_param.scan_mode = 1` e `can_fuse_optimize = false` (ja em `unir_varios`) -> log `isOptimize: 0`.
- Revo descarta ~11% dos pontos (mesmo sozinhos; nao e `minFrameCount`, nem isolados, repetir quadros nao muda) e produz superficie NITIDA (parede fina em 2 linhas); fusao simples por voxel borra a parede. Por isso a uniao: Revo onde ha, acrescimos so a > 2,5 mm do Revo (1,2 mm deixa "penugem").
- Apoio por scans sozinho nao limpa (suportes/chapas fixos na peca passam); precisa tambem da proximidade ao NUCLEO (>= 6 scans). Deteccao de planos nao serve (centro do para-choque e plano).
- Apagar quadros desloca indices k0/k1: remapear. `FrameRGBD.weights` e NULL; `FrameInfo` so usa a coluna `bin` (pose 4x4 float64 camera->mundo nos bytes 16..144; cauda = marcadores em coordenadas da camera).
- Hash de arestas com indices int32 estoura: usar int64. Redirecionamento do shell: `sleep` solto e bloqueado; esperar com `until ...; do sleep; done` (<= 290 s por chamada) e `run_in_background`.

## 4. Revo Scan 6 por UI (fatos verificados)
- Sem CLI/API; so UI Automation. Library `%APPDATA%\Revopoint\RevoScan6\Projects\<uuid>\<guid>.revox`. Importar = `Load Project` (dialogo #32770; reler o texto digitado e tentar `.click` e `.click_input`; caixa de erro tambem se chama "Load Project").
- Valores dos paineis vem de `param_template.postProcessParam` do `.revox` (gravar DEPOIS da importacao e ANTES de abrir o scan; lido so na 1a abertura por sessao): `pointPitch`+`isPointPitchSetted`, `meshMode` 0 rapida/1 alta, `optimizeLevel` 0/1/2 = None/Low/High, `meshEdgeLength`+`isMeshLenSetted`, `enableAutoFillHoles`, `enableClearMarker`, `enableExtractIsolation`+`enableEraseIsolatedPoint` (o log `extractIsolation: 0.005` e constante).
- Fim de etapa e valores usados: pelo LOG `%LOCALAPPDATA%\Revopoint\RevoScan6\logs\RevoScan6.log` (`point cloud fuse with pointPitch`, `fuse end, duration`, `meshingHelper] - meshMode`, `triMesh meshing end ... result: 0`, `Project opened:<pasta>`). A lista de modelos e virtualizada (so linhas visiveis).
- Apply do painel: achar o retangulo azul (0,136,255) na captura (a altura muda); Meshing so habilita com nuvem fundida selecionada. Exportar logo apos malha grande trava: fechar e reabrir o Revo antes (`revo_pipeline` ja faz). Scan montado grande demora > 5 min para abrir (espera 20 min).
- Lingua = ambiente de quem inicia (LANG=en_US.UTF-8); sessao do usuario em portugues e RECUSADA pela automacao (nao mexer nela; pedir que feche). `taskkill` sem /F nao fecha com scan aberto: `principal.close()`.
- Mouse: janela ELEVADA em foco (Gerenciador de Tarefas) bloqueia o mouse (`SetCursorPos` falha, "no active desktop"): `manter_tela_ativa()` acorda a tela, mantem ligada e minimiza janelas (Win+D) e `restaurar_janelas()` desfaz. Janela "Fusion (Nao esta respondendo)" de Revo antigo tambem segura o foco: matar e esperar o processo sumir.
- `Import Model` (menu lateral da biblioteca; *.stl *.obj *.ply) abre malha propria; `Data Import` e so transferencia do scanner. Vistas: so Perspectiva/Ortogonal; orbita por arrasto. Cria item novo na biblioteca (a biblioteca do usuario acumula copias de teste: nao apagar sem pedir).

## 5. Medidas de referencia
- Onix (20 scans): vies por celula 367 -> 104 um (p90 229); uniforme ao longo da peca (100-143 um). Fundida Revo 12,6 M pontos; uniao 15,3 M; malha Poisson 14,6 M triangulos; cobertura 98,7% (Revo sozinho 85%). Confianca: 92,4% da superficie final confirmada por >= 2 scans a < 1 mm; dos 2,7 M pontos acrescentados 99,5% confirmados (so 0,5% de um scan so); Revo 90,8% (>= 2 scans).
- Difusor (2 partes, `--lados opostos`): Revo 0,1 mm / malha alta: validado pelo usuario (MeshInspector).
- Tempo tipico: Revo fusao 1-3 min e malha 1,5 min para ~1000 quadros; fusao propria 8 min; Poisson prof. 11 em 15 M pontos 70 s.

## 6. O que e verificavel e o que nao (dizer sempre)
Medido: consistencia entre scans (vies), cobertura contra os scans, confianca por apoio, topologia da malha, malha abre no Revo.
NAO medido: exatidao absoluta (precisa de referencia CMM/peca padrao), abertura no NX, validacao visual do usuario.

## 7. Estado em 2026-10-22
- Difusor (`Project20260613_160228`/`20261017_204257`): entregue e aprovado.
- Para-choque Onix (`D:\Projects\Project20260921_185107`): entrega em `Desktop\Onix_parachoque_traseiro_ENTREGA` (v1.9); falta conferencia visual do usuario. Pastas de trabalho em `Desktop\_onix_*` (podem ser apagadas depois da aprovacao; `_onix_seg_c` tem segmentos/poses `direto4.json`).
- Segundo projeto pedido pelo usuario: `D:\Projects\Project20260607_154154` (7 scans, 1,5 GB, mesma calibracao do Onix, scan_mode misto, 2 scans sem fusao). Limpeza em `Desktop\_p2_limpo` (limpar_raw, 3 jobs); proximos passos = receita C (alinhar_varios -> segmentos -> direto -> consenso -> Revo -> propria).

## 8. Maquina (i7-13700K, 64 GB, GTX 1650 4 GB sem CUDA no Open3D)
Instabilidade sob carga sustentada em todos os nucleos: segfaults aleatorios do Open3D (3221225477/139), `cKDTree.query` devolvendo indice impossivel, VS Code/PC caindo; Windows registrou WHEA e "memoria defeituosa". Sempre: `baixa_prioridade()` (BELOW_NORMAL), OMP <= 3-4 threads x 4 processos, tarefas pesadas em processo isolado com repeticao (`isolado()`), validar resultados (`vizinho()`), checkpoints em disco, esperas curtas. Recomendar ao usuario BIOS/microcode Intel e teste de memoria. GPU nao compensa.

## Tiras da MESA que sobrevivem ao consenso (achado de 2026-10-22, usuario viu na malha do Onix)
- Causa: a mesa onde a peca estava apoiada fica FIXA em relacao a peca nos scans da MESMA pose de repouso; scans consecutivos
  (ex.: 194205..195038) se apoiam entre si e passam em `limpar_por_consenso` (>=2 outros scans). O usuario confirma: no scan com a
  peca virada pra cima essas aparas nao existem.
- Como achar: (1) planos grandes e finos (resid ~0,2 mm) em componentes soltos/quase soltos; (2) pontos apoiados SO por um grupo de
  scans consecutivos (agrupar scans por co-apoio em pontos com 2..5 scans) e paralelos a esse plano. Scripts de exploracao ficaram em %TEMP% (suporte_mask, t7).
- Resultado v2: removidos 95 mil pontos (0,62%) da mesa do grupo 194205-194821/195038 (plano normal ~(0,54;0,76;0,36) no ref. PCA). Outras tiras
  (outros grupos de pose) NAO foram provadas; apoio "so de um grupo" tambem pega superficie REAL vista so por esse grupo (canto esquerdo), entao nao
  remover cegamente por apoio.
- Melhor solucao futura: limpar a mesa ANTES do consenso, por scan, ajustando o plano da mesa nos quadros originais (referencia = scan virado pra cima).
- ATUALIZACAO (mesmo dia): janela de scans da mesma pose = 193159..195345 (9 scans, 19:31-19:53). Chapas finas FORA do casco (z 62-87 no ref. PCA, y -145..-220)
  vistas so por esses scans e por nenhum dos 11 anteriores: removidas (suporte exclusivo do grupo + normal ~ normal da mesa + conexo a semente flutuante).
  Teste que funcionou: "nenhum scan de outra pose enxerga esse ponto" + chapa que flutua acima do casco. v4 = 14,99 M pontos, malha 14,4 M tri; entrega atualizada.
  Placa de controle real (supp. por todos os scans) NAO foi removida. Ainda falta confirmacao visual do usuario.
- Verificacao das placas removidas (mesa): das colunas (x,y) das placas que scans de OUTRAS poses cobrem (35% dos pontos), so 6,6% tem superficie
  na altura da placa; em 93% essas poses veem superficie em outra altura mas nada ali => a placa nao existe nessas poses (consistente com mesa).
  Nos 65% restantes as outras poses nao olham a regiao: ali a remocao se apoia so na geometria (chapa plana flutuando acima do casco, normal da mesa).
  Consequencia: a regiao da grade inferior ficou com buracos (sem dado real de outras poses).

## Projeto 2 (D:\Projects\Project20260607_154154): sao DUAS PECAS, nao uma
- Miniaturas dos scans (`Desktop\_p2_alinh\miniaturas.png`): {154154, 154453, 154644} = uma peca (lateral/para-lama + frente);
  {155001, 155502, 160710, 161037} = outra (painel largo ~1400x600). `alinhar_varios` com os 7 so ligou 3; com os 4 sozinhos liga todos
  (concordancia entre ancoras 0,06 graus / 0,6-1 mm; saida `Desktop\_p2_alinhB`, referencia 160710). Licao: se sobram scans "sem ligacao", olhar
  miniaturas antes de afrouxar limiares; podem ser OUTRA peca.
- CORRECAO: os "buracos" da grade inferior do Onix nao sao falta de dado. Na regiao (x -470..360, y -225..-140, z<60) os 19 scans tem 2,86 M pontos
  e 99,2% deles tem vizinho a < 2,5 mm na nuvem final (so 23,8 mil orfaos). As aberturas retangulares sao as aberturas REAIS da grade; antes pareciam
  tampadas porque as chapas da mesa ficavam na frente na vista traseira. (Teste: pontos dos scans sem vizinho na nuvem final, por regiao.)

## Furos da malha Onix: causa = aparo do Poisson (2026-10-22)
- A malha v5 tinha 293 furos (3-60 mm de raio); 285 tinham dado de scan dentro. Reincorporar pontos "orfaos" (1,36 M confirmados por >=2 scans) NAO ajudou
  (309 furos) e 310 mil deles eram a mesa removida (filtrar!). Causa: `malha_poisson.py` descarta vertices com densidade < quantil 0,02 e a > 1,5 mm da nuvem.
  Poisson depth 11 da mesma nuvem: aparo (0,02; 1,5) -> 317 furos; (0,005; 3) -> 154; (0; 5) -> 84 (mas 5 mm fecha furos pequenos reais). Usar `--quantil 0.005 --dist-max 3`.
- Scans do pára-choque de OUTRO DIA existem: `D:\Projects\Project20260920_150020` (20260920_153720 cobre a face externa; 82% a < 3 mm do modelo; 153912 tambem) e
  `Project20260920_143823/20260920_145309`. Nao foram incorporados (os 20 scans ja cobrem quase todos os furos). `20260922_083006` e um teste RGB do usuario: ignorar.
- Varredura de todos os scans de D:\Projects (extensao por scan): `Desktop\_varredura_scans.csv`.
- v8: v7 + `o3d.t.geometry.TriangleMesh.fill_holes(hole_size=16)` (so liga vertices de borda existentes, +~1000 triangulos): furos 146 -> 41. Dos 146 da v7: 118 tinham dado de >=3 scans, 14 de 1-2, 14 sem dado nos 20 scans.

## MeshInspector / MeshLib na pipeline (2026-10-22, pedido do usuario)
- O usuario confere as malhas no MeshInspector (e num visualizador 3Dconnexion); NAO usa NX. `C:\Program Files\MeshInspector\MeshInspector\MeshInspector.exe` abre o .ply por argumento.
  Cores: azul = face de frente, amarelo-esverdeado = verso. Painel Information: Triangles, Vertices, Holes, Components.
- `pip install meshlib` (ja no .venv): o motor do MeshInspector. `pipeline/malha_meshlib.py` reproduz o painel EXATAMENTE (v8: 980610 tri, 491002 vert, 538 furos, 384 componentes)
  e repara (componentes < 2000 faces, arestas multiplas, furos <= 20 mm): v8 -> 71 furos, 1 componente. `malha_poisson.py` agora chama isso no fim (grava `*_reparada.ply`; aparo padrao 0,005/3 mm).
- Armadilha: minhas contagens antigas (Open3D, "furos de 3-60 mm") davam 41-97 onde o MeshInspector dava 538: arestas nao-manifold (104) viram componentes/furos nele. SEMPRE medir com `malha_meshlib.py`.
- `pipeline/meshinspector_ui.py` (print da janela via ctypes) existe mas nao foi validado: ele faz taskkill do MeshInspector; usar so com o usuario de acordo.
- Cobertura real: pontos dos scans confirmados por >=3 scans a > 2 mm da malha: v5 3,44%; v8 3,31%; malha da nuvem com orfaos (v6) 2,75% (concentrados na calha, bordas da grade, cantos = estruturas finas/interna).
- Faces invertidas: 10% dos triangulos da v8 tem normal oposta a da nuvem (agrupados na calha/cantos/grade) = as manchas amarelas do MeshInspector e brancas do visualizador do usuario.

## Manchas "de verso" (amarelo no MeshInspector / branco no 3Dconnexion) = normais da nuvem orientadas pra dentro (2026-10-22)
- Sintoma: v5, v8 e v9 pareciam IDENTICAS no visualizador do usuario, com manchas brancas na calha, grade, cantos, apesar de furos 293 -> 41 -> 73 nas minhas contagens.
- Reproduzir a vista por codigo: `python pipeline/vista_malha.py malha.ply --saida vista.png` (raios ortograficos por tras; azul = frente, amarelo = verso, branco = nada).
  v8/v9: 11% dos pixels com malha eram VERSO; em 64% deles ha so UMA camada (folha de verso), nao pele externa faltando.
- Causa: normais da fusao orientadas pelo lado de onde a camera viu (`visada`); onde so scan visto por dentro mediu a parede, o Poisson sai de verso.
- Correcao: `python pipeline/orientar_normais.py nuvem.ply --saida nuvem_orientada.ply` (z-buffer ortografico, celula 1 mm, tol 0,8 mm — NAO 3 mm, que pega a pele interna da parede;
  25 direcoes num cone de 65 graus; vira p/ fora a normal de ponto visivel de fora que nenhuma direcao aprova): 9,7% das normais viradas.
  Depois `malha_poisson.py` (aparo 0,005/3 + reparo MeshLib) e decimar p/ 1 M. v10: verso na vista de tras 10,9% -> 0,02%; MeshLib 61 furos, 3 componentes, 982 mil tri.
- Ordem nova do final da receita C: ... `uniao_revo_propria` -> `orientar_normais` -> `malha_poisson` (gera `*_reparada.ply`) -> `malha_meshlib` -> `vista_malha` (verso < 1%).
- O usuario NAO usa NX: confere no MeshInspector e num visualizador 3Dconnexion. Validacao do usuario so da v5 ("limpo"); v10 ainda nao conferida por ele.
- Maior scan do Onix: 20260921_185120 (1,58 M tri, 1,17 m2, cobre 79% da nuvem final); scans de outro dia: Project20260920_150020 (153720, 153912), Project20260920_143823/145309.

## Malha unica vs uniao (2026-10-22): a uniao estava PIOR que o melhor scan sozinho
- Usuario comparou lado a lado: malha do scan 185120 (Revo, lisa) vs v10 (pontilhada, bordas esfiapadas). Rugosidade local (RMS de plano em r=2,5 mm, vertices da malha): scan unico 22 um;
  v5 524 um; v10 580 um. Causa 1: a malha da uniao tem DUAS PELES (externa + interna, ~3 mm) porque scans virados pra cima veem o lado de dentro; Poisson mistura as duas.
  Causa 2 (restante): espessura da nuvem fundida de 20 scans (desencontro fino entre scans ~0,3 mm), que o Revo nao mostra num scan so.
- Correcao parcial: `orientar_normais.py --so-visiveis` grava so a pele externa (z-buffer, tol 0,8 mm): 8,4 M pontos; Poisson + reparo: 44 furos, 16 componentes; vista de tras 99,98% frente.
  Rugosidade 380 um; com Taubin (30 it, lambda 0,5, mu -0,53) 214 um (deslocamento medio 0,4 mm). Taubin 60: 181 um.
- AINDA NAO RESOLVIDO: chegar perto dos 22 um do scan unico. Caminho provavel: escolher, por regiao, a superficie de UM scan (o melhor alinhado) em vez de promediar 20 (sem espessura),
  usando a malha do scan de maior cobertura (185120, 79%) como base e acrescentando os outros so onde ela nao cobre ("construcao por cobertura" pedida pelo usuario).
- SOLUCAO (v12): Poisson em profundidade 10 (nao 11) sobre a nuvem so da pele externa: rugosidade 126 um (v11 214; v10 580; scan unico 22), 0,00% de verso na vista de tras,
  MeshLib 27 furos / 2 componentes / 978 mil tri. Mesmo aparo (0,005 / 3 mm) e reparo MeshLib. Poisson depth 9 apagou tudo com o aparo de 3 mm (nao usar sem relaxar).
  Construcao por cobertura com a malha de UM scan nao compensou: o 185120 se encaixa na nuvem final com 0,5 mm RMS (deriva propria) e cobre so 48% da pele externa a < 1 mm (`base_por_cobertura.py`).
  Receita final do Onix: nuvem_completa -> `orientar_normais --so-visiveis` -> `malha_poisson --profundidade 10` -> decimar p/ 1 M -> `malha_meshlib` -> `vista_malha` (verso ~0%).

## DESCOBERTA CENTRAL (2026-10-22): 14 dos 20 scans do Onix sao da face INTERNA; construcao por cobertura (v13)
- Malha original (fuse_mesh.ply) de cada scan, alinhada a nuvem externa: orientacao CONCORDA com a normal p/ fora em 6 scans (185120, 185633, 185733, 191925, 192042, 192128 = 18:51-19:21)
  e DISCORDA (3-18%) nos outros 14 (19:02 em diante... 190200, 190944, 192440, 192628, 193159, 193719, 194205..195709): eles viram a parede POR DENTRO. Alinhados a face externa ficam ~1,6 mm
  deslocados (espessura da parede) -> camadas duplas, pontilhado, verso. Isso explica v5..v12 (rugosidade 120-580 um) vs scan unico (22 um). Teste: `construcao_por_cobertura.py` imprime a concordancia por scan.
- Receita certa p/ a face EXTERNA: `construcao_por_cobertura.py <proj> <pasta_seg> <direto.json> <T_seg_revo.npy> <nuvem_externa.ply> --saida P --margem 0 --so-externos`
  (base = scan externo de maior cobertura apos aparar a <=1,5 mm da nuvem externa; guloso com so os pedacos que preenchem lacunas, abertura 3 mm, realinhamento local por pedaco).
  Depois `malha_meshlib.py montada.ply --saida m.ply --min-faces 1500 --furo-max 25`. Resultado v13: base 185733 + 185120, 185633, 192128, 191925; cobertura da pele externa 80,9%;
  rugosidade mediana 22 um (= scan unico), verso 0,17%, 96 furos, 34 componentes, 933 mil tri. Poisson sobre as pecas NAO ajuda (528 um: amplifica degraus de emenda).
- Os 19% restantes so tem dado nos scans INTERNOS. Os 14 internos servem p/ uma malha da face interna SEPARADA (ou p/ a espessura da parede), nao p/ preencher a externa.
- Como o usuario achou o problema: comparou lado a lado a malha do scan unico (lisa) com a uniao (pontilhada) no 3Dconnexion e disse "precisa mudar o conceito: maior scan como base e inserir so os que preenchem lacunas".

## ESTADO ao fim da sessao (2026-10-22) - RETOMAR DAQUI
- Entrega atual: v13 (`Desktop\Onix_parachoque_traseiro_ENTREGA\...para_enviar.ply`): lisa (22 um), mas com VAZIOS (aparo --tol-limpa 1.5 cortou areas boas do scan). O usuario comparou com o scan unico 185120 e disse que o scan unico "sobressai em muito".
- Em andamento: `Desktop\_onix_cob6` (log `_onix_cob6.log`): construcao por cobertura com `--tol-limpa 5.0 --tol-lacuna 2.0 --so-externos --apenas <6 externos> --extras _onix_extras.json`. Ao terminar: montar (`remove_duplicated*`, `malha_meshlib --min-faces 1500 --furo-max 25`),
  medir rugosidade/vista_malha/furos, comparar com o scan unico e v13; so entao trocar a entrega e enviar.
- Scans do dia 20/09 (153720, 153912) tambem sao INTERNOS (orientacao 4%): nao preenchem a face externa.
- O usuario nao usa NX; confere no 3Dconnexion Viewer / MeshInspector. Nenhuma versao apos a v5 foi aprovada por ele.
- Projeto 2: grupo B (painel) e grupo A entregues; ver memoria projeto2-duas-pecas.
- v14 (cob6: `--tol-limpa 5.0 --tol-lacuna 2.0 --so-externos --apenas <6 externos>`): base 185733 + 185120 (+374 mil tri), 185633, 192128, 191925; cobertura 80,7%; reparo MeshLib `--min-faces 1500 --furo-max 25`:
  1,33 M tri, 70 furos, 34 componentes, rugosidade mediana 21 um, verso 0,08%; vista de tras sem os vazios da v13 (aparo de 1,5 mm cortava areas boas; 5 mm so tira mesa/cenario). Entregue em `Desktop\Onix_parachoque_traseiro_ENTREGA`.
- Teste de soldagem de emendas na v14 (Open3D `merge_close_vertices`): eps 0,4 -> 64 furos/30 comp.; eps 0,8 -> 53 furos/22 comp. mas perde 36% dos triangulos (colapsa detalhe). NAO usar; ficar com a v14 como esta.

## Fechar furos so onde ha dado (v15) e o que o usuario aprovou
- O usuario viu a v14 e disse "gostei do resultado ate aqui" (aprovacao da linha: so scans EXTERNOS + construcao por cobertura). A v15 foi gerada depois e ainda nao foi vista por ele.
- Furos grandes da v14 (70 no MeshLib) NAO sao todos aberturas reais: os de perimetro 6,8 m e 5,8 m sao a borda da casca. Teste decisivo: simular `fillHole` numa copia e medir a distancia (p90) dos triangulos do remendo a nuvem externa. Remendo a < 5,5 mm da nuvem = ha superficie medida (aparo apagou) -> fechar; remendo no ar = abertura real -> manter. Resultado: 65 de 70 fechados, restam 5 (bordas). Script: `pipeline/fechar_furos_com_dado.py`.
- Usar `fillHoleNicely` com `triangulateOnly=False` (MeshLib); `FillHoleNicelySettings`, nao `...Params`. `findHoleRepresentiveEdges` (sem o "ta").
- ARMADILHA: vista por tras da v15 mostra verso 3,9% (v14: 0,08%) porque remendos cobrem reentrancias. Inverter faces do remendo uma a uma pelas normais da nuvem QUEBRA a topologia (8.497 furos, 6.746 componentes) -> nao fazer. Pendente: orientar o remendo como um todo (por componente) ou reduzir o verso de outro modo.
- Escrever malha com Open3D apos mexer em indices pode dividir vertices; gravar sempre pelo MeshLib e validar com `malha_meshlib.py`.
- Capturar tela do MeshInspector: `meshinspector_ui.py` falhou com "empty image" quando a janela estava minimizada; funcionou chamando ShowWindow(h,9) e (h,3) e SetForegroundWindow antes de GetWindowRect. O usuario NAO quer prints enviados a ele quando esta operando o PC (ele mesmo olha).
- RESOLVIDO (v15b): dos 64 remendos da v15, os que discordam (>50% das faces) das normais da nuvem externa sao os que viram verso; removendo so esses (por componente, via arestas entre faces novas) o verso volta a 0,02% e a cobertura da vista sobe de 70,1% (v14) para 73,2%, furos 70 -> 36, componentes 34. Fluxo: `fechar_furos_com_dado.py` -> remover remendos discordantes -> `malha_meshlib.py --min-faces 1500 --furo-max 0`. Arquivo: `parachoque_malha_completa_v15b_final.ply` (1,40 M tri).
