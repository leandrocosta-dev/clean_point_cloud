# Tutorial: limpar nuvem de pontos do Revo Scan no CloudCompare

Fonte: fórum Revopoint, tópico "[Tutorial] How to use Cloud Compare to clean up your messy point cloud/scan data", autor PUTV, abril de 2022.
Link: https://forum.revopoint3d.com/t/tutorial-how-to-use-cloud-compare-to-clean-up-your-messy-point-cloud-scan-data/12121

Este arquivo reúne o texto do tutorial traduzido e os valores que aparecem só nas imagens. Versão do CloudCompare usada nos prints: v2.12.0 (Kyiv), 64 bits.

---

## 1. O problema

- Escanear a mesma peça muitas vezes ("over scan") gera dados demais, e a fusão de pontos não consegue processar direito.
- O resultado é uma nuvem "fofa", cheia de quadros mal fundidos. A malha gerada dela sai com caroços e ruído, e o arquivo fica enorme.
- A regra do autor: a malha nunca fica melhor que a nuvem. "Garbage in, garbage out."
- O método resolve cerca de 95% dos casos, segundo o autor. Não é garantido.

## 2. Preparação

1. Baixe o CloudCompare, gratuito para Windows e Mac: https://www.danielgm.net/cc/
2. No Revo Scan, faça a fusão normalmente.
3. Abra no CloudCompare a **nuvem de pontos fundida** que está nos arquivos do projeto. **Não abra a malha.**
4. Formato recomendado: **PLY binário**. O Revo Studio abre PLY binário e ASCII.

## 3. Passo a passo

### Passo 1. Selecionar a nuvem
No painel **DB Tree**, clique na nuvem carregada para selecioná-la.

### Passo 2. Filtro SOR (Statistical Outlier Removal)
Menu: **Tools > Clean > SOR filter**

Função: remove os pontos soltos e os quadros extras que não foram fundidos direito.

Valores do print (janela "Statistical Outlier Removal"):

| Campo | Valor |
|---|---|
| Number of points to use for mean distance estimation | **6** |
| Standard deviation multiplier threshold (nSigma) | **1.00** |

Fórmula mostrada na janela: max distance = average distance + nSigma × std. dev.

Clique em **OK**.

Resultado: uma nova nuvem com o sufixo **"Cloud.clean"**. As bordas ficam menos "fofas" e os quadros soltos somem.

### Passo 3. Filtro de ruído (Noise)
Selecione a nova nuvem **"Cloud.clean"** (ignore a original).

Menu: **Tools > Clean > Noise filter**

Valores do print (janela "Filter noise"):

| Grupo | Campo | Valor |
|---|---|---|
| Neighbors | Modo | **Radius (Sphere)** (não usar "Points (kNN)") |
| Neighbors | Raio | **0.138415** no print. É o valor que o CloudCompare preenche sozinho para aquela nuvem; mantenha o valor automático da sua. |
| Neighbors | Points (kNN), desativado | 6 |
| Max error | Modo | **Relative** |
| Max error | Valor relativo | **1.00** |
| Max error | Absolute, desativado | 1.000000 |
| Opção | Remove isolated points | **Marcado** |

Ordem indicada no print: 1) marcar "Remove isolated points", 2) escolher "Radius (Sphere)", 3) clicar em **OK**.

Resultado: nova nuvem com o sufixo **"Cloud.clean.clean"**. Essa é a nuvem pronta para gerar a malha.

### Passo 4. Gerar a malha (Poisson)
Com **"Cloud.clean.clean"** selecionada:

Menu: **Plugins > PoissonRecon**

Valores do print (janela "Poisson Surface Reconstruction"):

| Campo | Valor |
|---|---|
| Octree depth (selecionado) | **8** |
| Resolution (não selecionado) | 0.843935 |
| interpolate cloud colors | **Marcado** |
| Aba Density: output density as SF | **Marcado** |

Ordem indicada no print: 1) Octree depth = 8, 2) marcar "output density as SF", 3) marcar "interpolate cloud colors", 4) **OK**.

Orientações do autor sobre o nível de octree:
- **8**: densidade normal de malha. Exemplo do autor: vaso.
- **9 a 10**: mais detalhe, no máximo. Exemplo do autor: rosto.
- Comece pelo 8. Ele já pode demorar em peças com muitos detalhes. Se precisar, gere outra malha com nível maior.
- Níveis altos podem criar micro ruídos.
- O "output density as SF" é opcional, mas recomendado se a peça **não tem fundo** (é aberta embaixo).

Texto de ajuda da própria janela (aba Density): a densidade serve para reduzir a extensão da malha para cobrir só a área da nuvem. Na malha gerada, ajuste o valor "min displayed" do SF nas propriedades até a parte visível ficar como você quer, e depois exporte uma nova malha com "Edit > Scalar fields > Filter by Value" (o SF "density" pode ser apagado depois).

Referência de tempo e tamanho do console no print:

| Nível | Triângulos | Vértices |
|---|---|---|
| 10 | 5.704.715 | 2.854.813 |
| 8 | 343.982 | 172.077 |

### Passo 5. Entender as cores
A malha aparece com cores (azul, verde, amarelo, vermelho). Elas mostram a **densidade da nuvem** (scalar field "Density", escala "Blue>Green>Yellow>Red"). A malha já está bem mais limpa, sem caroços.

### Passo 6. Remover a malha extra (peças sem fundo)
O Poisson sempre fecha a peça. Se a peça não tem fundo, ele cria uma "tampa" falsa, que aparece com densidade baixa.

No painel **Properties** da malha, em **SF display params**, aba **Display ranges**, arraste o controle da esquerda do histograma para a direita até a malha extra sumir.

Valores do print:

| Campo | Valor |
|---|---|
| Faixa mínima original | 5.40033500 |
| displayed (máximo) | 10.85259628 |
| Limite inferior ajustado | 2.0000000 (lido em print de baixa resolução) |
| saturation | 10.85259628 |

Cuidado: se arrastar demais, você reabre furos reais da peça.

### Passo 7a. Peça com fundo ou sem furos
Basta exportar a malha. Antes, troque a cor:

Painel **Properties** da malha > linha **Colors** > troque de **Scalar field** para **RGB** (ou **None** se o scan não tiver cor). Depois use o botão de salvar da barra de ferramentas.

Dados da malha no print: 343.982 faces, Scalar Field Count = 1, Active = Density.

### Passo 7b. Peça sem fundo (exemplo do bolo)
Menu: **Edit > Scalar fields > Filter By Value**

Valores do print (janela "Filter by value"):

| Campo | Valor |
|---|---|
| Range (mínimo) | **5.40033484** |
| Range (máximo) | **10.85259628** |
| Botão | **Export** (não usar "Split") |

Isso gera uma malha nova só com a faixa de densidade escolhida, sem a tampa falsa.

### Passo 8. Salvar
- Na malha nova, troque a cor de SF para **RGB**, ou **None** se o scan não tiver cor.
- Salve a malha.
- Opcional: exporte também a nuvem limpa ("Cloud.clean.clean") e use no lugar da nuvem suja no seu projeto do Revo Scan.
- Formato: **PLY binário**.

Segundo o autor, todos os detalhes da malha são preservados.

## 4. Dicas das respostas do tópico

- O CloudCompare não é intuitivo no começo, mas fica fácil depois que você sabe onde estão os botões.
- Cada filtro (SOR, Noise) cria uma nuvem nova e mantém a anterior. Se limpar demais, volte para a versão de antes.
- Pergunta sobre formato: PLY, sempre em **Binary**.
- Não dá para limpar quadros individuais antes da fusão; o autor diz que isso não é viável. Uma volta de 360° tem cerca de 24 quadros.
- Para juntar varreduras parciais, o autor prefere o CloudCompare ao MeshLab e tem outro tutorial sobre isso.
- Para evitar o problema: escanear uma volta de 360° por ângulo, mais devagar, sem repetir passadas.

## 5. Resumo rápido (cola)

| Etapa | Menu | Parâmetros |
|---|---|---|
| SOR | Tools > Clean > SOR filter | 6 pontos, nSigma 1.00 |
| Ruído | Tools > Clean > Noise filter | Radius (Sphere) automático, Relative 1.00, Remove isolated points marcado |
| Malha | Plugins > PoissonRecon | Octree 8 (até 10), output density as SF, interpolate cloud colors |
| Tirar tampa falsa | Properties > SF display params, depois Edit > Scalar fields > Filter By Value | Range pela densidade, botão Export |
| Exportar | Properties > Colors = RGB (ou None) | PLY binário |

## 6. Complementos (não estão no tutorial)

O tutorial limpa ruído, mas não separa a peça da mesa e dos apoios. Para isso, o próprio CloudCompare tem:

- **Edit > Segment** (ícone de tesoura): recorte manual em 3D, girando a vista livremente. Desenhe o contorno, escolha manter dentro ou fora e confirme.
- **Plugins > RANSAC Shape Detection** (ou Tools > Fit > Plane): detecta o plano da mesa para você apagar.
- **Tools > Segmentation > Label Connected Components**: separa a nuvem em grupos desconectados, como o Selecionar Conexão (B) do Revo Scan. Fique só com o grupo da peça.

Ordem sugerida para as peças da MoviAI: fundir no Revo Scan, exportar PLY, RANSAC (tirar mesa), Connected Components (tirar apoios soltos), SOR, Noise, Poisson.
