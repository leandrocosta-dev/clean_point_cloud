# GPU e engenharia — validacao de 08/11/2026

RTX 3060 12 GB, i7-13700K (24 processadores logicos), Windows, CuPy 14.2/CUDA 12.9. Resultados nao medem exatidao dimensional absoluta. A CPU continua responsavel por I/O, controle, pequenos sistemas por pose e operacoes Open3D/MeshLib sem equivalente CUDA. Os benchmarks abaixo usam destinos novos e referencias somente leitura.

Evidencias: `pipeline/output/RETOMADA_PARALELA_20261108/`. Controle anterior extraido de `e5aace7` em `controle_anterior/`; os fontes atuais nao foram revertidos para executar a comparacao.

## Medicoes seriais nesta retomada

| Etapa | Antes / controle CPU | GPU atual | Observacao |
|---|---:|---:|---|
| Limpar frames P8, quatro scans | 128,250 s | 40,283 s | 3,18x; mesmas opcoes e entradas |
| Metricas P8 (PLY 834.686 pontos) | 5,421 s | 1,764 s | 3,07x; mesmo raio e amostra |
| Fusao P7, cache novo | — | 20,665 s | 784.082 pontos, 0/3 calibracoes/pontos reutilizados |
| Fusao P7, segunda leitura | — | 17,071 s | 784.082 pontos, 3/3 calibracoes/pontos reutilizados |

Tempos da tabela sao de parede, incluindo iniciar o Python. Os relatorios internos da fusao marcam 19/15 s. Uma rodada de cada; nao sao distribuicoes estatisticas. O sistema tinha outras sessoes abertas. Comparacoes de velocidade com maior rigor exigem rodadas isoladas/repetidas.

Arquivos: `benchmark_limpeza.json`, `benchmark_metricas.json`, `benchmark_fusao_cache.json`. Frio/quente mantiveram a mesma cobertura e distancias geometricas residuais abaixo de 1e-9 mm (`cache_fusao_comparacao.json`).

## Preservacao de dados e geometria

Limpeza do P8 (`--ext-max-mm 100 --so-peca --min-mantido 0.02 --sem-ply-antigos`):

| Scan | Antes | Mantidos CPU | Mantidos GPU | So GPU / so CPU |
|---|---:|---:|---:|---:|
| 075313 | 14.824.408 | 14.277.032 | 14.277.022 | 481 / 491 |
| 075710 | 16.872.976 | 15.776.605 | 15.776.068 | 288 / 825 |
| 080053 | 12.200.307 | 11.425.205 | 11.425.191 | 125 / 139 |
| 080329 | 32.326.995 | 31.078.112 | 31.077.267 | 21 / 866 |

Maior diferenca de contagem menor que 0,004%, dentro do limite de 0,1% do plano. A auditoria integral tambem compara mascaras: contagem semelhante sozinha nao prova equivalencia. Poses/FrameInfo e valores z retidos identicos ao original; nenhum pixel criado; SQLite OK; nenhum quadro vazio. `auditoria_p8_gpu_cpu.log` e `benchmark_limpeza_gpu_atual/auditoria_integridade.json`.

Metricas P8: rugosidade mediana CPU 38,5462478 um, GPU 38,5459666 um, com raio aproximadamente 1,21185 mm; uma componente em ambos. O raio inclui curvatura e nao representa a rugosidade de uma face plana isolada.

Regressao completa com os quatro casos existentes, incluindo o caro P7: **aprovada**, codigo 0. P8 metrica, P8 triagem, P9 triagem e P7 fusao. O caso P7 agora usa poses indexadas pelos GUIDs da entrada e uma base registrada com limites geometricos verificados; nao foi substituida base para ocultar regressao nesta retomada. P7 levou 19,7 s nessa rodada.

## Alinhamento e consenso

- Alinhamento inicial P8, migracao comparada com execucao anterior: medianas finais aproximadamente 0,192–0,223 mm em ambos. O valor de 0,09–0,11 mm previsto no plano nao foi reproduzido nessa etapa; nao atribuir esse resultado a ela.
- Refino direto GPU real: 150 segmentos, 717 pares candidatos, 146 elos de odometria, 14 passos aceitos; 43 s. Viés por celula de 30 mm nos 60 pares reservados: **59,829 -> 27,722 um**. Nao houve reversao por piora. `p8_direto_gpu.json`/`.log`.
- Artefatos da fase 2 P9 foram conferidos: contagens finais e votos iguais entre `g2_consenso_p9_orig` e as rodadas GPU. Tempo registrado do controle 32,2 s e da rodada GPU C 29,3 s; foram produzidos pela outra sessao, sem reproducao serial independente nesta retomada. Nao confundir igualdade de contagem com prova integral das mesmas mascaras.

## Corte do apoio e seguranca

A receita anterior podia escolher uma face da peca depois de retirar a mesa. O teste automatico de 1 mm removeria 44,09% no primeiro scan P7: agora foi recusado com codigo 2 e copia intacta.

Usando planos previamente medidos no original (`--planos-apoio avaliacao.json --altura-min 1`):

| Scan | Antes | Depois |
|---|---:|---:|
| 210152 | 10.973.434 | 10.852.385 |
| 210505 | 11.124.428 | 10.774.080 |
| 210900 | 11.956.292 | 11.752.217 |

Cada scan terminou em 5–6 s. A auditoria confirmou poses e profundidades preservadas. Esse corte e intencional e exige recuperar a faixa por outra vista; nao prova eliminacao de todo suporte encostado. O limite de perda padrao e 30%; elevar o limite nao valida o plano automaticamente.

## Engenharia e testes

- Instalacao editavel, imports externos sem `sys.path.insert`, bibliotecas numericas compartilhadas, logging de progresso no stdout e JSON de maquina preservado.
- Ruff/format no codigo ativo, pre-commit local e CI Windows configurada. Treze testes de infraestrutura passaram numa venv sem CuPy/Open3D, sem skips. CUDA ausente pode pular testes numericos, erros de kernel nao.
- GUI: oito testes, nove paginas, prioridades, presets, Runner fora da raiz e formularios por subcomando. Bibliotecas/outputs deixaram de aparecer como ferramentas.
- Suite integrada com a nova selecao de qualidade: **254 testes aprovados em 40,575 s**, sem skips (`suite_final_qualidade_v2.log`). Ruff check/format passou nos 159 arquivos ativos. A rodada anterior tinha 227 testes e concorria com uma fusao real; os tempos das suites nao sao benchmark comparavel.
- `gpu.agrupar` de linhas usa ordenacao lexicografica/cumsum na GPU. Foi eliminado o caminho de CuPy 14.2 `unique(axis=0)`, que fazia comparacoes Python repetidas e bloqueava os testes.
- Componentes por raio usam conectividade exata, sem limite artificial de 64 vizinhos; testes incluem ponte entre grupos densos e limiar inclusivo.

## Limites ainda abertos

Triagem P5 continua escolhendo o gabarito elevado: a rodada atual classificou peca grande apoiada, enquanto a receita manual validada era suspensa. Nao usar essa recomendacao automaticamente sem conferir. VRAM/qualidade em Onix com vinte scans e a CI hospedada ainda nao foram validadas. A interface Revo permanece dependente de mouse/estado da aplicacao. Conferencia dimensional e visual pelo usuario continua necessaria.

A selecao das melhores regioes de scans, solicitada durante esta rodada, tem contrato em `SELECAO_QUALIDADE_3D.md` e validacao propria: dispersao espacial pode ser detalhe real, portanto nao deve ser apagada apenas por parecer mais rugosa que outra vista.

## Selecao regional e ciclo real

Os modulos `medicoes_qualidade_3d`, `selecao_qualidade` e `limpar_qualidade_3d` consultam observacoes 3D reais, com origem quadro/pixel e hashes. A evidencia temporal e separada da rugosidade espacial. Testes incluem detalhe exclusivo de 0,08 mm em um ou dois pixels, faces opostas, empates, fontes circulares, mascaras obsoletas e colisao de destinos. Suspeitas sem cobertura boa permanecem como pendencia; isso nao demonstra eliminacao de toda deformacao.

Ensaio ponta a ponta, sem mocks: copia reduzida do P7 com **32 quadros por scan, 96 no total**, mantendo FrameInfo e profundidades originais dos quadros selecionados. Foram analisados **2.061.642 pixels**, removidos **199** (51/37/111 por scan), em **15,023 s**. Poses, profundidades mantidas e SQLite passaram na auditoria; nenhum quadro vazio. **Seis pendencias**, estado `revisao_pendente`, sem aprovacao de merge. Nao extrapolar essa medida para os 1.543 quadros do projeto completo. Evidencias: `qualidade_amostra_p7_execucao.json`, `qualidade_3d.json` e `qualidade_3d/`.

O HTML `qualidade_3d/navegacao_amostrada.html` usa Plotly/WebGL embutido, com orbita, zoom, camadas e corte planar. A navegacao limita a 20 mil pontos por camada e declara a amostragem; PLY/NPZ integrais permanecem disponiveis. Um HTML amostrado nao prova que regioes ocultas foram limpas: decisoes usam consultas numericas integrais.

A repeticao apos a protecao contra colisao encontrou uma tentativa duplicada de criar a pasta reservada. A execucao falhou e reverteu somente suas proprias saidas, sem publicar projeto. Corrigido o escritor e adicionado teste que grava os artefatos reais em pasta reservada. A nova execucao ponta a ponta passou em **14,329 s**, com os mesmos 199 pixels removidos e seis pendencias; poses/profundidades/SQLite OK. HTML offline de 7,16 MB e quatro camadas NPZ foram gerados pela propria CLI, sem regeneracao manual: `qualidade_amostra_final/`. A suite tambem foi repetida depois dessa correcao.

O ciclo P7 completo anterior a esse novo estagio passou por limpeza, consenso, fusao nativa, tres fusoes reais no Revo e comparacao. Revo em 0,1 mm, sem otimizar poses nem fechar furos: 2.014/2.168/2.024 ms internos, retorno 0, sem divergencias. A importacao usa copia isolada e um `fuse.ply` inicial valido; a referencia de comparacao e o Fused gerado depois pelo proprio Revo. Este caminho nativo por superficies e diferente do benchmark `fundir_nitido` acima. O ciclo permanece aguardando revisao 3D, sem aprovacao/exportacao de merge. Logs: `ciclo_p7_revo.log`, `ciclo_p7_comparar.log`, `inspecao_ciclo_real.json`.
