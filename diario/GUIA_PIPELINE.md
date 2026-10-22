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
