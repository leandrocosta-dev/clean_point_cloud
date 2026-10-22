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
