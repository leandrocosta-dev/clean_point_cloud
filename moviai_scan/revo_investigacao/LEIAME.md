# Investigacao: projeto Revo Scan do difusor (17/10/2026)

Scripts usados para decodificar o projeto do Revo Scan 6 e corrigir a deriva de rastreamento.
Ainda sao especificos desta peca (eixo do cilindro, direcoes das paredes e trechos de quadros fixos no codigo).

## Formato do projeto (versao 1.3)
- `cache/frames.dataset`: SQLite com a assinatura de 16 bytes trocada por `RPAlgo_file_sys\0`.
  Para abrir, copiar e escrever `SQLite format 3\0` nos primeiros 16 bytes. Para gravar, restaurar a assinatura original.
- Tabela `FrameInfo`: coluna `bin` (164 bytes) = cabecalho de 16 bytes + matriz 4x4 em float64 (pose camera para mundo) + 20 bytes.
- Tabela `FrameRGBD`: coluna `data` = largura, altura, n (uint32) + mascara de bits 800x600 (ordem little) + 4 bytes + n valores float32 (profundidade em mm).
- ID do quadro: bits altos = segmento de rastreamento, bits baixos = indice.
- Calibracao ajustada empiricamente contra o raw_preview: fx 1140,4 | fy 1141,0 | cx 232,8 | cy 319,5.
  Ponto do pixel: X = (u-cx)*z/fx, Y = -(v-cy)*z/fy, Z = -z.
- `.revo`: campo `size` = soma dos arquivos do projeto sem o .revo. Gerar `guid` novo em copias.

## Diagnostico do difusor
- Quadros #70 a #147 do segmento 0 viam so o cilindro: o giro em torno do eixo ficou sem referencia e acumulou 17,2 graus.
- Correcao: rotacao em torno do eixo, linear de 0 a 17,2 graus em #70 a #147 e 17,2 graus constantes em #148 a #247.

## Ordem de execucao
revo_probe, revo_fit (calibracao) > revo_pose (validacao) > revo_seg, revo_plug, revo_jump, revo_timeline (diagnostico)
> revo_fix (correcao e fusao) > revo_val (antes/depois) > revo_write (grava projeto corrigido) > revo_stl (malha)
