# Plano de continuação: GPU em todo o software e padrões de engenharia (2026-11-08)

Documento de retomada para outra sessão ou outro modelo. Leia primeiro `AGENTS.md` e as seções 00, 0, 3 e 8 de `diario/GUIA_PIPELINE.md`.

## 1. Objetivo do usuário
1. O sistema deve ser rápido, preciso e funcionar em qualquer projeto de scan (Revo Scan 6, formato antigo `.dph`, Creality).
2. **Todo o cálculo numérico roda na GPU** (RTX 3060 12 GB, CuPy 14.2 + CUDA 12.9 na `.venv`). A CPU fica só com:
   - leitura e gravação de arquivos e do SQLite;
   - automação do Revo por interface;
   - funções do Open3D e da MeshLib sem equivalente em `pipeline/gpu.py`: FPFH, RANSAC por features, PoseGraph, Poisson, TSDF e operações de malha.
3. Recursos no máximo: prioridade alta e 24 threads (`pipeline/recursos.py`, modo `maximo`). A instabilidade antiga é histórica, anterior ao upgrade da GPU.
4. Padrões de engenharia:
   - pacote instalável;
   - biblioteca comum sem duplicação;
   - ruff;
   - `logging`;
   - testes e regressão;
   - commits pequenos por componente.
5. Regras que não podem ser quebradas:
   - nunca alterar `D:\Projects\...` nem `C:\Users\pc\Desktop\*_ENTREGA`; sempre gravar uma cópia nova;
   - limpar nos frames (`frames.dataset`), não em `.ply`;
   - não fechar furos por padrão;
   - responder em português;
   - toda afirmação vem com medida, e o que não foi verificado é dito.

## 2. Estado do repositório
- Branch `melhorias-gpu-engenharia`, a partir de `main`. Remoto: `origin` (GitHub `leandrocosta-dev/clean_point_cloud`). Nada foi enviado com push.
- Autor dos commits: `claude <noreply@anthropic.com>`, passado por variáveis de ambiente, porque o git local não tem identidade:
  ```powershell
  $env:GIT_AUTHOR_NAME='claude'; $env:GIT_AUTHOR_EMAIL='noreply@anthropic.com'; $env:GIT_COMMITTER_NAME='claude'; $env:GIT_COMMITTER_EMAIL='noreply@anthropic.com'
  ```
  Fim de cada mensagem de commit: `Co-Authored-By: <modelo> <noreply@anthropic.com>`. Assunto no padrão `pipeline: ...`, `gui: ...`, `guia: ...`, `tests: ...`.
- Commits já feitos neste branch, do mais antigo ao mais novo:
  1. `a0ef049` ciclo de precisão, fusão CUDA experimental, merge do Revo e auditorias
  2. `c261ac8` orçamento de recursos e portões de segurança
  3. `59c2e08` triagem, métricas, inspeção 3D e banco de regressão
  4. `d7cccac` consenso por visibilidade
  5. `cfbeed0` `fundir_nitido` na GPU e cache de observações
  6. `6c2fc02` scripts descartados em `pipeline/legado/`
  7. `61fe486` gui: prioridade alta, triagem, receita D
  8. `8cd1c06` guia: plano e resultados
  9. `b1aec3d` pacote instalável (`pyproject.toml`, `setup.py`), `pipeline/gpu.py`, `pipeline/comum.py`
- Ambiente: `.\.venv\Scripts\python.exe -m pip install -e .` já rodado, então os módulos de `pipeline/` e `pipeline/revo_frames/` são importáveis pelo nome de topo. O ruff está instalado na `.venv`.
- Testes: `.\.venv\Scripts\python.exe -m unittest discover -s tests`. Depois do commit 9 eram 127 passando.

## 3. O que já existe (não refazer)
| Item | Onde | Situação |
|---|---|---|
| Orçamento de recursos | `pipeline/recursos.py` | `REVO_RECURSOS=maximo` (padrão) ou `conservador`; `REVO_TETO_THREADS` |
| Camada GPU obrigatória | `pipeline/gpu.py` | Veja a API abaixo. `REVO_PERMITIR_CPU=1` só para depuração e testes de referência |
| Kernels de vizinhança | `pipeline/vizinhos_gpu.py` | `Grade` (mean-shift, PCA, contagem, mais próximo, knn) |
| Utilitários comuns | `pipeline/comum.py` | `sha256`, `ler_ply`/`gravar_ply`, `salvar_json_atomico`, `configurar_log`, `destino_novo` |
| Triagem automática | `pipeline/triagem.py` | Acerta P3, P4, P8 e P9. **Falha no P5**: escolhe a placa-gabarito elevada |
| Portões | `limpar_raw`, `limpar_por_consenso`, `alinhar_direto`, `rodar_tudo` | Código 2 = resultado parcial; consenso euclidiano; passo LM aceito só se melhora; reversão pelo viés |
| Exclusões de suporte | `limpar_raw --exclusoes` | Prevalecem sobre o fuse e sobre a evidência temporal |
| Consenso por visibilidade | `pipeline/consenso_visibilidade.py` | P9: 093808 10%→1,1%, 094322 9,6%→0,2%, 093504 25,9%→9,5% sem confirmação |
| Métricas, inspeção 3D, regressão | `metricas.py`, `consulta3d.py`, `regressao.py`, `bench/` | Bases: `p8_metrica_entrega`, `p8_triagem`, `p9_triagem`. `p7_fundir_nitido` definido, sem base |
| `fundir_nitido` na GPU | `--motor cuda` | P7: 78 s (GPU) contra 446 s (CPU); diferença mediana de 3,4 µm |

API de `pipeline/gpu.py`: `xp`, `em_gpu`, `para_gpu`, `para_cpu`, `memoria_vram`, `liberar_memoria`, `tamanho_bloco`, `blocos`, `agrupar`, `somar_por_grupo`, `chaves_voxel`, `decodificar_voxel`, `reduzir_voxel`, `Indice`, `vizinho_mais_proximo` (idx=-1 fora do `dist_max`), `contar_no_raio`, `knn` (exato até k=64), `normais_pca`, `remover_soltos`, `ransac_plano` (mesmo contrato de `revo_projeto.plano_ransac`), `icp_ponto_plano` (Tukey), `componentes_conexos_voxel`. Funções aceitam numpy ou cupy e devolvem no tipo da entrada.

## 4. Em andamento quando este documento foi escrito: fase 2, migração para a GPU
Cinco agentes, cada um com arquivos exclusivos, sem commit. **Ao retomar, verifique o que cada grupo deixou** com `git status` / `git diff --stat` e rode os testes. Se um grupo ficou incompleto, termine-o seguindo as mesmas instruções.

Instruções comuns a todos os grupos:
- trocar `cKDTree`, `scipy.ndimage`, `estimate_normals`, `voxel_down_sample`, `registration_icp` ponto-plano, `evaluate_registration`, `compute_point_cloud_distance` e laços numpy pesados pela `gpu.py`;
- manter os dados na GPU entre etapas;
- preservar regras e limiares;
- medir as diferenças contra a versão anterior (`git show HEAD:arquivo` numa pasta temporária);
- registrar o que ficou na CPU e por quê;
- não mexer em `sys.path.insert`, `print` nem formatação (isso é a fase 3).

| Grupo | Arquivos | Validação obrigatória |
|---|---|---|
| 1 Limpeza | `revo_frames/revo_projeto.py`, `revo_frames/precisao.py`, `limpar_raw.py`, `limpar_por_placa.py`, `limpar_por_fuse.py`, `limpar_por_consenso.py`, `limpar_por_merge_revo.py`, `componentes.py`, `corrigir_perda_rastreio.py`, `auditar_limpeza.py`, `avaliar_periferia.py`, `avaliar_corte_apoio.py`, `triagem.py`, `tests/test_gpu_limpeza.py` | `limpar_raw D:\Projects\Project20261107_074043 --ext-max-mm 100 --so-peca --min-mantido 0.02 --sem-ply-antigos` antes e depois: pontos por scan com diferença menor que 0,1%, mais o tempo; `regressao.py --caso p8_triagem p9_triagem` ok |
| 2 Alinhamento | `alinhar_varios.py`, `alinhar_segmentos.py`, `alinhar_direto.py`, `uniao_scans.py`, `alinhar_interna_por_ponte.py`, `avaliar_merge_revo.py`, `aprender.py`, `consenso_visibilidade.py` (GPU obrigatória), `avaliar_segmentos.py`, `avaliar_por_scan.py`, `remapear_segmentos.py`, `cobertura.py`, `selecionar_segmentos.py`, `montar_pecas.py`, `tests/test_gpu_alinhamento.py` | `alinhar_varios --escala auto` no P8 (limpo + placa 1 mm) antes e depois; mediana de encaixe ~0,09-0,11 mm. Consenso por visibilidade no P9 com os mesmos pixels removidos de `pipeline/output/consenso_vis_p9_v4` |
| 3 Fusão | `fundir_nitido.py` (GPU padrão e obrigatória, ICP multivista via `gpu.icp_ponto_plano`), `fusao_superficies.py`, `comparar_fusao_cuda.py`, `cache_observacoes.py`, `fundir_proprio.py`, `refinar_nuvem.py`, `suavizar_nuvem.py`, `uniao_revo_propria.py`, `orientar_normais.py`, `confianca.py`, `limpar_artefatos.py`, `limpar_fragmentos.py`, `ciclo_precisao.py`, `inspecao_fusao.py`, `rodar_ciclo_precisao.py`, `tests/test_gpu_fusao.py` | P7: `fundir_nitido pipeline\output\Project20261106_210152_acesso_corrigido --poses pipeline\output\bench_nitido_poses_p7.json` contra `bench_nitido_cuda_p7` (784.301 pontos) e `bench_nitido_cpu_p7`. Rodar 2× para medir o cache. Registrar a base de `p7_fundir_nitido` |
| 4 Malha e medição | `metricas.py` (rugosidade em lote na GPU), `consulta3d.py`, `construcao_por_cobertura.py`, `base_por_cobertura.py`, `completar_com_poisson.py`, `sem_atravessar.py`, `corrigir_interna_suave.py`, `limpar_interna_com_externa.py`, `faltando.py`, `avaliar_fusao.py`, `rugosidade_interna.py`, `malha_poisson.py`, `vista_malha.py`, `lacunas.py`, `fechar_furos_pequenos.py`, `remover_remendos_discordantes.py`, `metrica_malha.py`, `fuse_mesh.py`, `malha_meshlib.py`, `tests/test_gpu_malha.py` | `regressao.py --caso p8_metrica_entrega` ok (834.686 pontos; rugosidade ~38,5 µm); `consulta3d comparar` das nuvens v1/v2 do P8 = 6.639 sumidos a 0,05 mm |
| 5 Formatos antigos | `revo_antigo/*.py`, `creality/*.py`, `build_revo_project.py`, `clean_revo.py`, `criar_projeto_biblioteca.py`, `preparar_importacao.py`, `auditar_projeto.py`, `relatorio_qualidade.py`, `abrir_no_revo.py`, `verificar_cuda.py`, `legado/*.py`, `tests/test_gpu_antigos.py` | Dados `.dph`, se existirem em `D:\Projects` ou no Desktop; senão, testes sintéticos |

Checagem de conclusão da fase 2. Este comando mostra o que ainda usa CPU:
```powershell
Get-ChildItem pipeline -Recurse -Filter *.py | ? { $_.FullName -notmatch '\\(output|__pycache__)\\' } | Select-String -Pattern 'cKDTree|query_ball_point|estimate_normals|voxel_down_sample|registration_icp|compute_point_cloud_distance|scipy\.ndimage' | Group-Object Path | % { "$($_.Count) $($_.Name)" }
```
Só podem sobrar usos justificados: referência CPU em testes ou atrás de `REVO_PERMITIR_CPU`.

Commits da fase 2, um por grupo:
- `pipeline: limpeza dos quadros na GPU`
- `pipeline: alinhamento na GPU`
- `pipeline: fusao e nuvem na GPU (fundir_nitido GPU obrigatoria)`
- `pipeline: malha, cobertura e metricas na GPU`
- `pipeline: formatos antigos e creality na GPU`

## 5. Fase 3: padrões de engenharia (depois da fase 2)
Faça em commits separados e rode a suíte depois de cada passo.
1. **Remover os `sys.path.insert`** (cerca de 67) de `pipeline/`, `gui/` e `tests/`, agora que o pacote é editável. Garantir que `python pipeline/x.py` e a GUI continuam funcionando, inclusive os scripts executados como processo pela GUI (`gui/runner.py`).
2. **ruff:** `ruff check --fix` e depois `ruff format` em `pipeline/`, `gui/` e `tests/`. Corrigir à mão o que sobrar (F401 imports mortos, B, E). Reavaliar o `ignore` do `pyproject.toml` (hoje E501 e I001).
3. **logging:** trocar os cerca de 346 `print` por `comum.configurar_log`, começando pelos scripts principais (`limpar_raw`, `rodar_tudo`, consenso, `alinhar_*`, `fundir_nitido`, `fusao_superficies`, `triagem`, `revo_pipeline`). Manter a saída legível na GUI, que lê o stdout: o handler de console deve ir para stdout.
4. **Deduplicação:** os `carregar` ×8, `metricas` ×4, `sha256` ×3, `nuvem` ×4 e `processar` ×4 devem usar `comum.py`/`gpu.py` onde forem equivalentes. `baixa_prioridade` deve vir só de `recursos.py`.
5. **Tratamento de erros:** trocar os 26 `except Exception` por exceções específicas onde for possível. Manter os que protegem a interface ou processos isolados, com comentário.
6. **Anotações de tipo** nas funções públicas de `gpu.py`, `comum.py`, `recursos.py`, `revo_projeto.py`, `metricas.py` e `triagem.py`.
7. **Hook de pré-commit e CI:**
   - `.pre-commit-config.yaml` local com ruff e testes rápidos;
   - opcionalmente, um workflow do GitHub Actions só com ruff e os testes que não precisam de GPU. Testes de GPU são pulados sem CUDA: usar `unittest.skipUnless`.
8. **GUI:**
   - conferir na tela, com prints, a página Ambiente (seletor de prioridade) e os formulários dos scripts novos (`triagem`, `consenso_visibilidade`, `metricas`, `consulta3d`, `regressao`, `--motor`);
   - remover a opção de motor CPU dos formulários, se ainda aparecer.

## 6. Fase 4: revisão final e entrega
1. Suíte completa e `--help` de todos os scripts (catálogo da GUI: `gui/catalog.carregar()` sem erro).
2. Regressão: `.\.venv\Scripts\python.exe pipeline\regressao.py` (todos os casos, incluindo `p7_fundir_nitido`).
3. Benchmarks antes e depois da GPU, documentados em `diario/BENCHMARK_GPU_<data>.md`: tempo total e por etapa de `limpar_raw` (P8), `alinhar_varios` (P8), `consenso_visibilidade` (P9), `fundir_nitido` (P7) e `metricas` (P8).
4. Atualizar o `GUIA_PIPELINE.md`:
   - seção 00: GPU obrigatória, `pip install -e .`;
   - seção 1: tabela de scripts com o backend;
   - seção 8.
   Atualizar também o `DIARIO.md` (resultados medidos, sem duplicar) e o `AGENTS.md`: comandos de build/teste com `pip install -e .` e ruff; teste de GPU obrigatório.
5. Commits finais. Push e PR **só se o usuário pedir**. Corpo do PR: mudança de processamento, scripts afetados, dados de validação, e terminar com `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

## 7. Pendências do plano original que continuam abertas
- **Triagem no P5** (`D:\Projects\Project20261026_192833`, arco suspenso sobre placa-gabarito furada): escolhe o gabarito elevado como peça. Investigar o critério do foco das câmeras e a distribuição de altura.
- **Validação real que nunca foi feita:**
  - consenso euclidiano, `alinhar_direto` novo (aceitação LM, Tukey auto, reversão) e `alinhar_varios --escala auto` em projeto grande. As pastas de segmentos do Onix foram apagadas; refazer a receita C em `D:\Projects\Project20260921_185107` se o usuário quiser;
  - `--cache` de observações em dado real;
  - comportamento de VRAM em peça grande (Onix, 20 scans).
- **Nada foi fundido no Revo nem conferido visualmente pelo usuário:** a limpeza do P9 por visibilidade (`pipeline/output/consenso_vis_p9_v4`) e a fusão GPU do P7.
- **Arquivos travados:** em `pipeline/output/Project20261106_210152_precisao_{v2,v3,v4,final}` ficam 12 `frames.dataset` com ACL privada (cerca de 1 GB). Só saem com PowerShell como administrador: `takeown /F <pasta> /R /D S` e depois `Remove-Item -Recurse -Force <pasta>`.
- **Ciclo orquestrado:** `ciclo_precisao.py` ainda não chama a triagem nem o consenso por visibilidade. A integração faz parte da Fase 4 do plano original: limpar, fundir, inspecionar, corrigir, refundir e liberar para o merge.
- **Revo por interface:** continua sendo a etapa mais lenta e frágil. A fusão nativa na GPU só substitui o Revo depois de passar o comparativo da auditoria (`diario/ANALISE_PRECISAO_2026-11-08.md`).

## 8. Dados de referência para validação (somente leitura)
| Caso | Entrada | Referência |
|---|---|---|
| P7 (peça de 45 mm, detalhes finos) | `pipeline/output/Project20261106_210152_acesso_corrigido` + `pipeline/output/bench_nitido_poses_p7.json` | `C:\Users\pc\Desktop\Projeto7_ENTREGA\nuvem_nitida.ply`; `pipeline/output/bench_nitido_{cpu,cuda}_p7` |
| P8 (moldura de 47 mm, aprovada) | `D:\Projects\Project20261107_074043` | `C:\Users\pc\Desktop\Projeto8_ENTREGA\peca_fusao_nitida.ply` (834.686 pontos) |
| P9 (8 poses, massinha) | `C:\Users\pc\Desktop\Projeto9_ENTREGA\Project20261108_092049_limpo_v4_IMPORTAR` + `_avaliacao_v4_merge08\poses_final.json` | `pipeline/output/consenso_vis_p9_v4`; tabela da v5 em `Projeto9_ENTREGA\..._limpo_v5\LEIA-ME.md` |
| Triagem | `D:\Projects\Project20261025_164458` (P3), `...174228` (P4), `...20261026_192833` (P5), `...20261107_074043` (P8), `...20261108_092049` (P9) | P3, P4, P8 e P9 corretos; P5 incorreto |
