# Auditoria de precisão — 2026-11-08

Escopo: leitura do guia, histórico, limpeza de frames, evidência temporal, consenso entre scans, otimizador conjunto, fusão própria, refino e orquestração. Execução dos testes existentes e verificações numéricas pequenas. Não é uma certificação de todos os scripts do repositório. Nenhum scan foi reprocessado nesta auditoria e nenhum algoritmo foi alterado.

## Conclusão

O projeto tem uma base útil: limpeza dos dados brutos, preservação das profundidades retidas, validação temporal conservadora, alinhamento conjunto e auditoria dos arquivos. Entretanto, a proteção de detalhes não é uniforme entre as etapas. Não há evidência suficiente para garantir precisão máxima ou ausência de ruído em peças arbitrárias.

O objetivo de evolução deve ser reduzir ruído com perda de geometria medida, separando observações confirmadas, contraditas e inconclusivas. Falta de observação não demonstra que uma superfície é falsa. Passo de exportação de 0,1 mm também não demonstra exatidão de 0,1 mm.

## Achados em ordem de prioridade

### 1. Crítico: consenso pode eliminar a peça inteira com poucos scans

Em `pipeline/limpar_por_consenso.py`, `--nucleo` tem padrão 6, sem validar a quantidade de scans. Com menos de seis scans, nenhum voxel pode formar o núcleo. A grade fica vazia, `ok &= perto` rejeita todos os pontos dos quadros abrangidos pelos segmentos e o limiar de 1.500 pontos apaga esses quadros. Mesmo com seis ou mais scans, o núcleo pode ficar vazio por falta de sobreposição. Isso decorre do código; não foi executado sobre dados reais nesta auditoria.

Correção prioritária: validar viabilidade antes de criar a entrega, detectar núcleo vazio e interromper com diagnóstico. Não reduzir automaticamente o apoio exigido: isso muda a confiança. Não tratar limiares específicos do para-choque Onix como receita universal.

### 2. Alto: proteção temporal começa depois da segmentação

Em `pipeline/revo_frames/revo_projeto.py:771`, `mascara_manter` define a base. Em :843 somente `base & ~ok` passa pela evidência temporal. Em :853 o resultado continua limitado à base. Assim, um detalhe real excluído pela segmentação nunca é resgatado pelo modo `precisao`.

Correção proposta: registrar separadamente descartes de cenário e propostas geométricas ambíguas; permitir revisão multivista das propostas ambíguas. Recortes de mesa deliberados precisam de máscara e relatório próprios. Não liberar cenário inteiro sem critério.

### 3. Alto: consenso confunde ausência de apoio com ruído

O consenso exige apoio de outros scans e proximidade ao núcleo. Não consulta as imagens de profundidade das outras câmeras para distinguir ausência de cobertura, oclusão e espaço livre observado. Uma nervura interna ou borda vista por apenas uma posição pode ser real e será rejeitada pela regra atual. Scans repetidos na mesma posição de apoio também podem confirmar a mesa entre si, como já registrado no guia.

Correção proposta: usar voxels apenas para localizar candidatos e verificar a decisão por reprojeção em câmeras de outros scans, com poses alinhadas. Contar diversidade de posições da peça e de visadas, além do número de scans. Preservar e sinalizar o inconclusivo.

### 4. Alto: tolerâncias do consenso não são distâncias euclidianas

`VOX = 2.0` e duas grades deslocadas permitem correspondência dentro de uma célula cuja diagonal é 3,464 mm. Isso não significa que todo par a essa distância corresponde, mas prova que a célula não impõe um limite rígido de 1 mm. Além disso, a dilatação do núcleo usa vizinhança 26-conexa: cinco passos de 3 mm alcançam centros a 25,981 mm na diagonal, embora o parâmetro seja apresentado como raio de 15 mm. `max(1, ...)` também aplica uma dilatação quando o raio é zero.

Correção proposta: após a busca por voxels, verificar distância física e compatibilidade de superfície; usar distância euclidiana para a vizinhança do núcleo e validar unidades/limiares. Valores novos exigem ensaio comparativo, pois mudariam a receita existente.

### 5. Alto: a cadeia principal não integra a validação multisscan completa

`pipeline/rodar_tudo.py` limpa por scan, opcionalmente une um par e aciona o Revo. Não integra a sequência de N scans com alinhamento conjunto, avaliação independente, consenso e nova avaliação. Portanto executar a cadeia principal não garante que a limpeza entre scans descrita no guia foi aplicada.

Também verifica o código de saída de `limpar_raw`, mas este pode terminar normalmente copiando scans com status `erro`, `pulado` ou `revisar`. Esses estados precisam aparecer como resultado parcial e impedir uma aprovação automática de qualidade.

### 6. Alto: otimização aceita atualizações sem demonstrar melhora

Em `pipeline/alinhar_direto.py`, as poses recebem todas as atualizações calculadas. Não há busca de passo, rejeição da atualização por aumento de custo ou avaliação independente antes de salvar. A conectividade e as direções pouco observáveis também não são verificadas explicitamente nesse otimizador.

O corte robusto Tukey é fixo em 0,5 mm, mesmo quando a busca de correspondências começa em 6 mm: pares encontrados podem ter peso zero se o resíduo normal for grande. Isso pode reduzir a capacidade de refino a partir de uma inicialização imperfeita; o efeito depende dos dados.

O RMS exibido divide a soma ponderada dos resíduos por uma contagem que inclui resíduos de peso zero quando o par tem algum peso positivo. Não é um RMS ponderado normalizado pela soma dos pesos e pode aparentar melhora com rejeição de correspondências.

Correção proposta: custo e normalização explícitos, número efetivo de correspondências, verificação de conectividade/condicionamento, passos limitados e aceitação pela melhora. Conservar a melhor pose anterior e comparar o viés por células de 30 mm em dados reservados antes da aprovação.

### 7. Médio: proteção temporal conservadora permite ruído persistente

`ValidadorTemporal` consulta até quatro quadros a distâncias fixas de índices. Um apoio em qualquer pixel da janela 3×3 impede descarte; o espaço livre exige os nove pixels válidos. Isso protege contornos, mas pode conservar fantasmas persistentes e usar evidência de câmeras praticamente iguais. Não corrige poses suspeitas.

Evolução proposta: selecionar visadas por cobertura e separação geométrica; estimar incerteza por distância/incidência/registro; investigar regiões com resíduos persistentes em vez de apenas aumentar o limiar. Não diminuir globalmente `pixel-tol` para forçar mais remoção.

### 8. Alto para detalhes finos: média e suavização não têm garantia geométrica

`fundir_proprio.py` agrega por voxel sem separar explicitamente superfícies opostas. `refinar_nuvem.py` reagrupa em 0,6 mm, ajusta planos com raio de 1,8 mm e permite deslocamento de até 0,5 mm por passada. Limitar deslocamento não garante conservação de uma aresta, abertura ou parede mais fina que essa escala. A afirmação no docstring de que não colapsa detalhes finos é mais forte que a validação demonstrada.

Evolução proposta: separar camadas por normal/profundidade/visada, limitar influência entre faces opostas e preservar descontinuidades. Entregar a nuvem medida limpa junto à versão suavizada, com mapa de deslocamento. A malha deve ser avaliada separadamente; furos permanecem opt-in.

### 9. Médio: preview e frames podem representar seleções diferentes

O preview aplica a máscara de ilhas diretamente em `revo_projeto.py:902`, enquanto os frames em modo precisão podem recuperar esses mesmos candidatos pela evidência temporal. Isso pode mostrar ausência no preview onde ainda há dados brutos válidos. Reconstruir o preview a partir das máscaras finais tornaria a inspeção coerente.

### 10. Limitações adicionais confirmadas no código

- `popcount` do consenso conta apenas 48 bits. Verificação executada: bit 0 retorna 1; bit 48 retorna 0. A partir do 49º scan a contagem deixa de representar todos os scans; deslocamentos maiores também encontram o limite do inteiro usado. Validar limite ou substituir a representação.
- `--jobs` do consenso não impõe teto de 12 threads e não configura todos os backends numéricos. A cadeia precisa aplicar o orçamento de recursos de forma uniforme, respeitando a instabilidade documentada da máquina.
- A cobertura de 0,5 mm do modo precisão compara o resultado com a peça já segmentada e usa cada oitavo pixel. É uma medida útil desse estágio, mas não detecta detalhes removidos antes nem toda perda submilimétrica.

## Sequência de evolução recomendada

### Proposta avaliada: fusão nativa com CPU/GPU

O usuário propõe substituir a fusão dependente da interface do Revo por código nativo, visando agilidade e precisão. A base já existe em três caminhos: `fundir_proprio.py` (centroides por voxel), `fundir_nitido.py` (pesos de incidência e mean-shift na normal) e o TSDF do formato antigo em `revo_antigo/tsdf_scan.py`. Nenhum deles deve ser promovido automaticamente a substituto universal do Revo.

Verificação local em 2026-11-08: CPU i7-13700K, 16 núcleos/24 processadores lógicos; `nvidia-smi` informa RTX 3060, 12.288 MiB de VRAM e driver 621.14. A descrição antiga da GTX 1650 no guia não representa a GPU atual. Open3D instalado: 0.20.0, `o3d.core.cuda.is_available()` retorna False. Portanto a GPU NVIDIA atual não está disponível para os kernels CUDA desse ambiente. Não foram instaladas bibliotecas nem alterados drivers nesta auditoria.

A documentação oficial atual do Open3D descreve integração TSDF em VoxelBlockGrid com kernels CPU/CUDA e suporte de instalação/compilação CUDA no Windows. Isso demonstra uma rota técnica, não compatibilidade já testada nesta máquina nem aceleração garantida. Referências: https://www.open3d.org/docs/latest/tutorial/t_reconstruction_system/integration.html e https://www.open3d.org/docs/latest/getting_started.html . O suporte Windows acelerado deve ser validado em ambiente separado antes de mudar o ambiente existente.

Limitações específicas encontradas: a fusão simples não separa faces antes da média; o motor nítido usa um único K calculado para todos os scans, carrega todos os pontos e centros de câmera na RAM, aplica ICP sem aceitação independente de melhora e filtros de densidade/planaridade que exigem controle de preservação de detalhes. O TSDF existente atende ao formato antigo e tem histórico de camadas duplas na fusão multisscan; não reutilizar sua receita conjunta como padrão para frames.dataset. As medições históricas de rugosidade do motor nítido não equivalem à exatidão física de poucos micrômetros.

Arquitetura candidata para implementação:

1. Leitor em blocos dos frames, com calibração por scanner/modo, máscaras de suporte e proveniência. Preservar os originais e manter coordenadas/unidades explícitas.
2. Fusão individual por scan antes do merge, com poses recebidas como entrada. Otimização de poses é uma etapa separada, avaliada e versionada.
3. Motor de amostras de superfície orientadas (surfels) com múltiplas camadas por região, ponderação por incerteza/incidência, consenso robusto e proteção de arestas/faces opostas. Adaptar o motor nítido como candidato de comparação, sem assumir que a direção média de visada basta para resolver toda parede fina.
4. Grade espacial esparsa e processamento em blocos, com CPU para leitura/decodificação e kernels compilados CPU/GPU para projeção, associação e acumulação. Evitar laços Python por ponto. Medir separadamente custo de transferência RAM/VRAM.
5. Cache versionado de observações e dependências espaciais para refazer só regiões afetadas quando possível. Não subtrair simplesmente uma média anterior para desfazer uma fusão robusta: invalidar e recomputar os blocos dependentes com suas observações. Alterações de poses podem exigir reintegração ampla; bordas entre blocos precisam de vizinhança e verificação de continuidade.
6. Saída de nuvem com normais, confiança, proveniência e dados inconclusivos separados; malha opcional e posterior. Integração com o ciclo inspeção 3D → correção nos frames → refusão → liberação para merge.

Uso de recursos: aproveitar CPU, GPU, RAM e armazenamento conforme perfil medido, mantendo interface responsiva e limites de memória. A autorização para aproveitar recursos não remove o teto vigente de 12 threads nem a prioridade baixa devido ao histórico de instabilidade. GPU pode ser testada independentemente desse teto; aumentar carga de CPU exige primeiro validar estabilidade. Utilização máxima não é critério de qualidade ou velocidade por si só.

Plano de aceitação: comparar Revo, CPU nativa e GPU nativa com exatamente os mesmos frames/máscaras/poses e parâmetros registrados. Medir tempo total incluindo I/O e transferências, pico de RAM/VRAM, cobertura por região, resíduos e viés por células, preservação de aberturas/arestas/espessuras, observações excluídas e repetibilidade. Comparar também refusão localizada e reconstrução integral. Hardware não garante precisão; a substituição depende de passar esse comparativo, mantendo o Revo disponível como referência durante a transição. Integração CUDA e novo motor ainda não foram implementados ou benchmarkados nesta auditoria.

### Requisito esclarecido: ciclo de limpeza após fusão e antes do merge

O usuário exige que cada scan passe por limpeza de frames, fusão, inspeção 3D e, havendo resíduo, nova limpeza dos frames e nova fusão antes de ser liberado para Merge Alignment. Resíduos flutuantes, abas artificiais, suportes e sujeira encostada na peça são pendências de qualidade, mesmo quando conectados ao corpo principal. Uma aba real da peça deve continuar protegida; conectividade ou espessura, isoladamente, não determinam identidade.

Fluxo requerido por scan: frames originais preservados → cópia limpa versionada → fusão correspondente à versão → inspeção 3D de mantidos/removidos → regiões de resíduo identificadas → transformação dessas regiões para as observações dos frames de origem → correção localizada e auditada → nova fusão → nova inspeção. Repetir enquanto houver pendência corrigível; se a classificação permanecer ambígua ou as iterações não melhorarem, marcar revisão pendente e não liberar automaticamente para o merge. Concluir uma iteração ou atingir um limite não equivale a aprovação.

As máscaras devem carregar identidade do scan, versão dos dados, coordenadas e transformação usadas na inspeção. Se a fusão alterar poses ou referencial, recalcular o mapeamento; não aplicar uma seleção espacial antiga sobre coordenadas novas sem verificação. Proximidade de um resíduo ao corpo não autoriza apagar uma vizinhança inteira: proteger as observações da superfície válida e registrar a evidência da separação. Toda correção deve gerar nova saída e comparação com o original, sem deslocar profundidades retidas inadvertidamente.

Alterar frames invalida a fusão/malha anterior e qualquer merge dependente. Esses artefatos devem ser marcados desatualizados e reconstruídos antes da entrega. Depois do Merge Alignment, uma segunda avaliação verifica incompatibilidades entre scans; resíduo descoberto ali retorna ao scan responsável, seguido de refusão e reavaliação/reexecução do alinhamento dependente. O merge não deve ser o único detector de sujeira.

Existem peças parciais desse mecanismo: `limpar_por_fuse.py` reaplica proximidade de uma fusão aos frames; `avaliar_merge_revo.py` e `limpar_por_merge_revo.py` detectam candidatos após merge e retornam aos frames. Não constituem o ciclo requerido: a primeira depende de uma fusão de referência completa e já limpa (pode conservar lixo encostado ou cortar dado real ausente nela); as outras dependem de um merge já feito e usam falta de confirmação como parte da classificação. Precisam das salvaguardas de visibilidade e pertencimento descritas nesta auditoria.

Critério de liberação proposto: nenhuma região identificada como resíduo permanece sem resolução, suportes identificados excluídos, detalhes medidos preservados nas comparações, inspeção 3D registrada, integridade dos frames aprovada e fusão atualizada para a mesma versão. Isso é um controle verificável de pendências, não prova matemática de ruído zero. A orquestração e o bloqueio automático ainda não foram implementados nesta auditoria.

### Requisito esclarecido: inspeção dos dados em 3D

O usuário exige acesso à geometria tridimensional durante a limpeza para que a decisão não dependa apenas do que uma imagem 2D mostra. O processamento atual já opera em 3D em várias etapas (reconstrução dos frames, árvores de vizinhança, registro, consenso e confiança); isso não equivale a uma interface completa de inspeção 3D para a LLM. `vista_malha.py` gera uma imagem por primeiros impactos de raios em uma direção: superfícies atrás desses impactos permanecem ocultas nessa imagem. Cortes e projeções existentes ajudam, mas não demonstram revisão de todo o volume.

Requisito de arquitetura: fornecer ferramentas consultáveis pela LLM sobre a geometria real, com seleção por caixa/esfera/plano, cortes em qualquer eixo, componentes, vizinhos, normais, distâncias e correspondências entre scans. Cada região deve preservar origem (scan, quadro e pixel), estado de classificação e motivo da remoção. Expor à LLM resultados estruturados dessas consultas e imagens locais complementares; não depender de inserir milhões de coordenadas no contexto.

A inspeção humana deve permitir órbita, zoom, isolamento por scan/componente, ocultação de suporte/camada, planos de corte e sobreposição antes/depois, incluindo os pontos removidos. Transparência e múltiplas vistas são auxiliares; não substituem cortes e consultas geométricas que alcançam estruturas internas. Uma visualização reduzida pode ser usada para navegação, mas decisões e auditoria precisam acessar os dados na resolução original em blocos.

Antes de aprovar uma limpeza, percorrer espacialmente todas as regiões com dados e registrar quais foram avaliadas, quais mudaram e quais permanecem ambíguas. Distinguir: geometria medida oculta na vista atual (consultável em 3D), superfície ocluída em um scan mas medida em outro (recuperável) e superfície nunca medida (desconhecida). Nenhuma interface 3D torna observável uma região que o scanner não capturou. Não reconstruí-la automaticamente como se tivesse sido medida.

Essa interface e essa verificação de cobertura de inspeção são propostas de evolução; não foram implementadas nesta auditoria.

### Requisito esclarecido: entregar somente a peça

O usuário confirmou nesta auditoria que alguns scans usam elementos para manter a peça posicionada. Suportes, gabaritos, fixadores e elementos de posicionamento devem ser totalmente descartados, mesmo quando persistentes e confirmados por vários scans. Esta é uma decisão de pertencimento à peça, separada da detecção de ruído de medição.

Achado adicional: `mascara_manter` preserva regiões sob a pegada/coluna da peça e também faz `ok |= np.isfinite(d)` para pontos próximos do `fuse.ply`. Se esse fuse contém um suporte, a referência pode recuperar o suporte junto com detalhes reais. A exclusão explícita de elementos de posicionamento deve prevalecer sobre essas regras de recuperação e sobre o apoio temporal. O código atual só aplica a exclusão de periferia depois dessa recuperação; ela não descreve suportes arbitrários.

`limpar_por_placa.py` fornece uma solução específica para placa e objetos baixos, mas seu corte por altura não é uma solução geral para grampos, hastes e suportes que encostam na peça. A própria face plana da peça pode confundir a seleção de plano. Não aplicar corte por altura universal para cumprir esse requisito.

A evolução precisa manter máscaras identificadas de peça/suporte por scan, usar mudanças de posição e observações da mesma região sem suporte para separar os objetos e aplicar a exclusão nos frames originais antes da fusão. Onde o suporte escondeu a peça, registrar falta de medição; usar outras posições quando disponíveis. Não inventar a superfície encoberta nem fechar a lacuna automaticamente. Se houver ambiguidade de identidade no contato, uma identificação visual localizada pode ser necessária; consenso geométrico sozinho não resolve todos os casos.

1. Bloquear consenso inviável/núcleo vazio, estados parciais silenciosos e parâmetros fora de limites.
2. Cobrir por evidência multivista os descartes geométricos ambíguos e registrar o motivo de cada remoção.
3. Integrar alinhamento de N scans com avaliação antes/depois e reversão se piorar.
4. Substituir decisão de consenso por voxels por verificação física com visibilidade e diversidade de poses.
5. Validar fusão que separa faces e suavização que preserva arestas, mantendo uma saída sem suavização.

Critérios de comparação: viés normal com sinal por células de 30 mm, distribuição e p90/p95 dos resíduos, cobertura por região, conservação de aberturas e nervuras, deslocamentos introduzidos e apoio por posições independentes. Viés pequeno sozinho pode ocultar ruído de sinais opostos e não mede todas as direções de erro. Reservar quadros/visadas para avaliação, sem reutilizá-los como única fonte de decisão. Exatidão absoluta requer referência dimensional física.

## Validação executada nesta auditoria

- `.\.venv\Scripts\python.exe -m unittest discover -s tests -v`: 18 testes aprovados; avisos de arquivos abertos sem fechamento explícito em `revo_projeto.py`.
- `limpar_raw.py --help` e `rodar_tudo.py --help`: código de saída zero usando o Python da `.venv`.
- Verificações numéricas de `popcount` e das diagonais descritas acima. São verificações pequenas, não ensaios de qualidade de um scan.
- O Python local funciona nesta sessão, apesar da limitação de ambiente registrada em entradas históricas.
- Não foram refeitas fusão, malha ou inspeção visual no Revo. Não foi medida precisão dimensional real. Os arquivos e projetos originais permaneceram intactos. Alterações preexistentes na árvore de trabalho foram preservadas.
