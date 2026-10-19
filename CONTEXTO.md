# Contexto para continuar no Claude Code (17/10/2026)

Projeto: sistema MoviAI de limpeza e correcao de deriva de rastreamento para o Revopoint Miraco Plus.
Peca de teste: difusor eletrico de repelente (cilindro de cerca de 45,8 mm, bloco do plugue, 10 rasgos a 36 graus no topo, furinho na borda oposto ao plugue).

## Objetivo
Ler o projeto do Revo Scan quadro a quadro, detectar e corrigir deriva de rastreamento e gravar uma COPIA do projeto (.revo) para o Revo Scan fundir. Nunca alterar o original.

## Formato do projeto Revo Scan 6 (versao 1.3), confirmado
- `data/<scan>/cache/frames.dataset`: SQLite com os 16 primeiros bytes trocados por `RPAlgo_file_sys\0`. Para ler, copiar e gravar `SQLite format 3\0` nos 16 bytes. Para gravar de volta, restaurar a assinatura original.
- Tabela FrameInfo, coluna `bin`: cabecalho 16 bytes (4 x uint32/int32) + matriz 4x4 float64 (pose camera para mundo) + cauda.
  - hdr[1]: o byte 2 ((hdr[1]>>16)&0xff) e o MODO do quadro = indice da pasta de calibracao (0 = param, 1 = param1).
  - hdr[0]: em scans sem marcador = inicio do bloco de armazenamento (0, 499, 998). Em scans com marcador = numero do grupo de rastreamento (muda quando o scanner se reposiciona).
  - Cauda em scans com marcador: lista de marcadores (x, y, z, nx, ny, nz, qualidade, raio 3,0) em float32.
- Tabela FrameRGBD, coluna `data`: largura, altura, n (uint32) + mascara de bits 800x600 (ordem little) + 4 bytes + n float32 (profundidade em mm).
- ID do quadro: bits altos = bloco de armazenamento de 500 indices (NAO e perda de rastreamento), bits baixos = indice. Ordem temporal: (bloco, indice).
- Calibracao do modo usado nos scans de modo unico (estavel em 4 projetos): fx 1140, fy 1141, cx 233, cy 319,5. Ponto: X=(u-cx)*z/fx, Y=-(v-cy)*z/fy, Z=-z. O cx e mesmo deslocado em relacao a metade do Pl.bin. `revo.py` ajusta automaticamente contra o raw_preview.ply (erro 0,3 mm).
- `.revo`: campo `size` = soma dos arquivos do projeto sem o .revo. Em copias, gerar `guid` novo.
- `property.rvproj`: `scan_mode` 1 = rastreamento por marcadores, 0 = por geometria. `accuracy_type` NAO indica perto/longe.

## Resultado do scan de 15/10 (ja resolvido)
Quadros #70 a #147 viam so o cilindro; o giro em torno do eixo derivou 17,2 graus e o plugue duplicou. Correcao: rotacao em torno do eixo, linear de 0 a 17,2 graus no trecho cego e constante depois. Validado: plugue unico, 10 rasgos nitidos, plugue oposto ao furinho (foto do usuario). Projeto corrigido gerado; ainda falta o usuario fundir no Revo Scan.
Scripts dessa investigacao: pasta `revo_investigacao` dentro do moviai_scan.zip enviado no chat.

## Scans de validacao desta pasta (17/10)
1. 20261017_171853: marcadores no prato, mesa automatica, tripe fixo. 286 quadros, um objeto de 74x49x38 mm (difusor em pe). Reposicionamento no quadro #154 (salto de 12,4 graus). E o GABARITO.
2. 20261017_172640: sem marcadores, scanner NA MAO (corrigido pelo usuario; trajetoria livre, raio 66 a 192 mm). 538 quadros, passos suaves (max 2,15 graus).
3. 20261017_172322: sem marcadores, mesa automatica, tripe (corrigido pelo usuario; orbita circular R 153,6 mm, residuo 0,8 mm, varredura 434 graus, 74 graus de sobreposicao no fim). 627 quadros, passos suaves (max 1,75 graus).
4. 20261017_173102: com marcadores, longe e depois perto (pequena perda no perto). Quadros #0-224 modo 0, #225-321 modo 1, ultimo bloco modo 0 com varios grupos. Saltos de coordenadas nas trocas de modo (ate 178 graus e 505 mm).

## Resultado da validacao dos 4 scans (17/10)
- 171853 (gabarito), 172322 e 172640: sem deriva de rotacao em torno do eixo (teste por janela de 12 quadros contra quadros distantes no tempo, rotacao otima 0 +-0,5 graus, erro 0,07 a 0,25 mm). Vao sem alteracao. Limite do teste: so cobre rotacao em torno do eixo do cilindro.
- Objetos extras nos scans 172322 e 172640 (cupula, bloco) sao pecas de apoio de rastreamento do usuario.
- 173102: no modo perto (seq 225 a 321, 97 quadros) os marcadores sairam da camera. Esses quadros nao tem profundidade (FrameRGBD so tem 297 dos 394) e nao tem marcadores. Depois da perda o rastreamento recomecou em outro referencial: corridas continuas 322-359 e 365-381 ficaram deslocadas cerca de 59 graus e 160 mm no referencial (nuvem a 11 a 13 mm da certa), e 382-393 olhavam para fora da peca. A corrida 360-364 e os grupos 1 e 2 (seq 0-224) estao corretos.
- Correcao: pose = C @ pose, com C obtida do salto de pose entre quadros consecutivos (359->360 e 364->365; as duas estimativas concordam dentro de um passo de camera), refinada por ICP (0,7 graus, 3,4 mm). Quadros 382-393 encadeados pelo salto 381->382 (C4). Resultado: 94,8% dos pontos a menos de 1 mm (mediana 0,23 mm) contra 20% antes.
- Armadilha: ICP dos marcadores com limiar inicial largo (40 mm) alterna entre dois minimos locais; usar limiar apertado partindo da pose gravada. Com 3 a 4 marcadores a rotacao fica livre em torno da linha entre eles, entao marcador sozinho nao detecta o erro; a geometria detecta.
- Residuo: R1/R2/R3 contra G1/G2 fica em mediana de 0,9 a 1,2 mm (G1 contra ele mesmo: 0,2). Refino rigido adicional nao melhora (0,93 para 0,86 mm), provavelmente ruido do conjunto distante.
- Pacote: Desktop/Project20261017_171402_corrigido (guid novo, size recalculado, sem -shm/-wal, original intocado). Scripts em validacao_4scans/.
- raw_preview.ply e preview.png do 173102 no pacote continuam os antigos (nao regenerados).

## Pendente (o que sobrou)
- RESOLVIDO: os objetos extras nos scans 2 e 3 (cupula de cerca de 50 mm e bloco de cerca de 30 mm) sao pecas de apoio de rastreamento colocadas pelo usuario. Nao sao deriva. Sao as ancoras de geometria desses scans; recortar so o difusor (acima do plano, cilindro de cerca de 46 mm) nas metricas.
- Depois: linha do tempo da orientacao das paredes do plugue por quadro (familias de direcao mod 90), detectar duplicacao, medir correcao e comparar com o gabarito (scan 1).
- Generalizar: deteccao de trecho cego pela analise de estabilidade (degenerescencia do ICP), modo da mesa pela trajetoria da camera, validacao automatica antes de gravar.

## Regras do usuario
Responder em portugues do Brasil, direto, sem bajulacao, maxima especificidade. Nunca usar travessao em documentos.

## ERRO CONFIRMADO NO PACOTE CORRIGIDO (17/10, 19h)
- O pacote Project20261017_171402_corrigido esta ERRADO no scan 173102. Nao usar.
- O usuario fundiu no Revo Scan: o trecho corrigido saiu inclinado e deslocado. O Revo usa exatamente as poses gravadas, nao reposiciona.
- Causa: a correcao C_final (58 graus, 163 mm) tem eixo de rotacao 64 graus fora do eixo do cilindro e foi aplicada nas sequencias 322 a 381, que estavam CORRETAS no original. Contra os quadros bons (seq 0 a 224): so 36% dos pontos a menos de 1 mm, plugue com mediana de 5 mm. A metrica de 94,8% usada na validacao nao detectava o erro (armadilha do cilindro: distancia a superficie nao prova o encaixe).
- Calibracao do modo 0 no 173102 ficou fraca (preview com 4.854 pontos, erro de 0,97 mm, fy 1152).
- Numeracao do Revo Scan = indice entre os quadros COM profundidade (FrameRGBD), na ordem (bloco, indice). No 173102: Revo 0 a 224 = seq 0 a 224; Revo 225 a 296 = seq 322 a 393. Seq 225 a 321 (modo 1) nao tem profundidade.
- O usuario selecionou no Revo o trecho que perdeu a referencia: quadros Revo 285 a 296 = seq 382 a 393. SO esses sao problematicos.
- Solucao adotada: projeto original + excluir os quadros Revo 285 a 296 no proprio Revo Scan.

## REGRAS A PARTIR DAQUI
1. Nunca gravar correcao de pose sem verificacao especifica do detalhe nao simetrico (plugue) contra o gabarito. Distancia media a superficie nao vale como prova em corpo cilindrico.
2. Preferir EXCLUIR quadros ruins a corrigi-los quando a correcao nao for comprovada.
3. Conferir com o usuario no Revo Scan quais quadros ele ve como problematicos antes de alterar qualquer coisa.
4. A limpeza por quadro (base, prato, pecas de apoio) ainda NAO existe no codigo.

## LIMPEZA POR QUADRO IMPLEMENTADA (17/10, 19h30)
- Codigo: moviai_scan/moviai_scan/revo_projeto.py e moviai_scan/limpar_revo.py (sem Open3D).
- Uso: python limpar_revo.py C:\Users\pc\Desktop\Project20261017_171402 --scans 20261017_172640
- Faz: calibra cada modo contra o raw_preview (pula o scan se erro > 0,6 mm), acha o plano do prato, separa objetos acima dele em voxels de 1 mm, escolhe a peca pela mira da camera, apaga em cada quadro os pixels fora da peca. Poses intactas (conferido byte a byte). Consolida o WAL antes de restaurar a assinatura.
- Primeira saida: saida/Project20261017_171402_limpo (so o scan 20261017_172640). Pontos 88,6 M -> 18,2 M (21%). Peca = objeto 106 (117 de 176 miras). AGUARDANDO o usuario fundir no Revo Scan.

## REFATORACAO EM SISTEMA UNICO (19/10)
- Tudo virou o pacote moviai_scan com um CLI so: `python run.py <comando>` dentro de moviai_scan/. Ambiente em .venv (Python 3.10).
- Equivalencias: limpar_revo.py -> `run.py revo limpar`; revo_fix/revo_write -> `run.py revo giro` (estima, grava fusoes antes/depois e correcao.json, nao toca no projeto) + `run.py revo aplicar` (grava a copia); revo_seg/revo_jump -> `run.py revo info`; revo_fit -> `run.py revo info --calibrar`; fuse_mesh.py -> `run.py malha`; Clean-PointCloud.ps1 e Fix-RotationalDrift.ps1 -> `run.py cc limpar` e `run.py cc deriva`. Codigo antigo no commit 7c12f4c.
- Validado: `revo limpar` grava o mesmo conteudo da versao anterior (391 linhas iguais no 20261015_204805); `revo giro` no scan de 15/10 estima 17,30 graus (17,25 com a calibracao fixa antiga; o original deu 17,20); rampa por quadro identica a fixes.npy; `revo aplicar` confere as poses relendo a copia e o original fica intacto (hash).
- Correcao de bug: `mesclar` mesclava encaixe ambiguo. Agora recusa (status ambiguo, saida 2).
- Problema conhecido: `limpar` em PLY real nao e reprodutivel (paralelismo do Open3D). Na mesma entrada, o trecho do raw_preview de 15/10 saiu realinhado numa execucao e ambiguo em outra.
- frameFlag.json de cada scan tem `disabled_index: []`. Provavel mecanismo do Revo Scan para excluir quadros; NAO confirmado.
