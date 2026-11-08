# CUDA instalado e primeiras comparações — 2026-11-08

## Ambiente funcionando

- RTX 3060 com 12 GB; i7-13700K e aproximadamente 64 GB RAM.
- Instalados na `.venv`: CuPy 14.2.0 e componentes CUDA Toolkit/runtime 12.9 (`requirements-cuda.txt`). Driver NVIDIA existente preservado.
- Compilação NVRTC, redução e produto matricial cuBLAS executados na GPU. Comparação da matriz com NumPy: erro máximo 0 no ensaio 16×16.
- Open3D 0.20.0 continua CPU: o wheel oficial Windows testado tem `BUILD_CUDA_MODULE=False`. CUDA funciona pelo CuPy; instalar o runtime não converte automaticamente os algoritmos antigos para GPU.
- `cupy.show_config()` confirma runtime 12090, NVRTC 12.9, compute capability 8.6 e bibliotecas cuBLAS/cuFFT/cuRAND/cuSOLVER/cuSPARSE disponíveis. O aviso de ausência de raiz CUDA única permanece no CuPy 14.2 com os componentes CUDA 12 distribuídos em diretórios separados; os testes de compilação e execução passaram. Existe também um NVCC 12.1 antigo instalado no sistema; não foi usado para compilar este motor, que usa NVRTC 12.9.

Instalação reproduzível, conforme o mecanismo de componentes CUDA do [CuPy](https://docs.cupy.dev/en/latest/install.html):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
.\.venv\Scripts\python.exe pipeline\verificar_cuda.py
```

## Experimento executado

Script novo: `pipeline/comparar_fusao_cuda.py`, disponível também no catálogo da GUI como benchmark experimental. Implementa a mesma agregação por voxel em NumPy/CPU e CuPy/CUDA, leitura de frames em cópia temporária, saída nova e cache de coordenadas. A chave do cache considera SHA-256 do dataset, preview usado na calibração e versão do preparo. Não usa a referência Revo para construir a nuvem.

Entrada: `pipeline/output/Project20261106_210152_UNIDO_FINAL/data/7f889b8c-4033-480a-a785-a87f0b3030d2`. São 1.543 quadros e 34.054.154 observações; SHA-256 do dataset `75b189d00aa8aa72772c16aa0063395c3f90ddc1c9693e6d6ac5f51cd14227a4`. Mesmas coordenadas e poses para CPU e GPU, sem ICP. Voxel 0,1 mm e mínimo de três observações.

Referência geométrica: `pipeline/output/Project20261106_210152_ENTREGA_UNIDA/nuvem_unida_limpa.ply`, 407.095 pontos. É a fusão histórica do Revo a 0,10 mm após exclusão de 13 pontos esparsos; não é padrão dimensional físico. A referência histórica não foi gerada novamente nesta rodada. O arquivo de entrega também contém uma revisão posterior dos frames (33.864.614 observações); a entrada nativa escolhida é a versão anterior (34.054.154) citada no histórico da fusão. A correspondência integral com os frames efetivamente usados pelo Revo não foi comprovada nesta rodada. Não apresentar este ensaio como comparação controlada definitiva entre motores.

## Tempos medidos

Uma execução de cada configuração; números iniciais, sem distribuição de repetibilidade:

| Configuração | Fusão com transferências | Execução com cache, PLY e avaliação |
|---|---:|---:|
| CPU, blocos de 2 milhões | 6,27 s | 8,82 s |
| GPU, blocos de 2 milhões | 6,48 s | 16,16 s, incluindo preparo inicial de 6,75 s |
| CPU, bloco até 40 milhões | 4,48 s | 7,04 s |
| GPU, bloco até 40 milhões | 0,60 s | 3,51 s |

O preparo inicial dos frames/calibração levou 6,75 s e foi reutilizado pelos demais ensaios. O aquecimento CUDA medido separadamente no ensaio otimizado foi 0,13 s. Tempos totais acima não incluem iniciar o interpretador/importar todos os módulos; não comparar com tempo de interface de outro programa.

Ganho observado GPU/CPU no agrupamento de bloco grande: 7,51 vezes; execução com cache aproximadamente 2 vezes. Não é ganho medido contra o Revo. Aumentar o bloco eliminou muitas reagregações; não é uma melhoria na classificação de ruído. O bloco deve caber na VRAM: a execução GPU amostrou aproximadamente 3,56 GiB no pool CuPy e 1,87 GiB de RSS. Esses valores não são picos contínuos nem incluem toda a VRAM usada pelo desktop. Nenhum processo alheio foi encerrado. CPU em prioridade baixa, orçamento configurado de até 12 threads por backend.

CPU e GPU produziram 1.323.020 pontos. Os PLY otimizado CPU e GPU têm SHA-256 iguais. Esse teste demonstra equivalência numérica neste conjunto; não demonstra que as superfícies estão corretas.

## Comparação geométrica inicial com Revo

- Distância nativo → referência: mediana 0,0870 mm; p95 0,2805 mm; 4,595% acima de 0,3 mm.
- Distância referência → nativo: mediana 0,0531 mm; p95 0,0742 mm; 0,258% acima de 0,3 mm. Logo 99,742% dos pontos da referência têm amostra nativa a até 0,3 mm.
- Resíduo normal absoluto, limitado aos pares a menos de 0,3 mm: mediana 0,0713 mm; p95 0,1855 mm. Essa restrição é importante: os pontos distantes ficam fora dessa estatística, mas aparecem nas distâncias acima.
- Medianas assinadas em nove células de 30 mm: 0,0060 a 0,0593 mm, com orientação das normais da referência. Não representam exatidão absoluta, nem validação entre scans independentes.

O maior número de pontos nativos não demonstra maior detalhe: média por voxel conserva camadas de observações ruidosas e pode misturar faces. Não classificar automaticamente os 4,595% como lixo: podem conter dados reais ausentes na referência. Este motor é baseline experimental, não foi ativado no pipeline de produção nem substitui `fundir_nitido.py` ou o Revo.

## Reprodução

```powershell
.\.venv\Scripts\python.exe pipeline\comparar_fusao_cuda.py pipeline/output/Project20261106_210152_UNIDO_FINAL/data/7f889b8c-4033-480a-a785-a87f0b3030d2 --referencia pipeline/output/Project20261106_210152_ENTREGA_UNIDA/nuvem_unida_limpa.ply --saida pipeline/output/benchmark_novo_gpu --motor cuda --bloco 40000000
```

Trocar `--motor cuda` por `--motor cpu` e usar outra saída para comparar. Relatórios JSON e nuvens desta rodada: `pipeline/output/benchmark_cuda_20261108/{cpu,gpu,cpu_bloco40m,gpu_bloco40m}`. Dados gerados ficam ignorados pelo Git.

Validação: 20 testes aprovados, incluindo equivalência CPU/CUDA e independência do agrupamento em blocos, `--help` e leitura dos argumentos pelo catálogo GUI; `pip check` sem dependências quebradas. Persistem avisos antigos de arquivos sem fechamento explícito nos testes de frames. Não houve fusão nova no Revo ou inspeção visual 3D nesta rodada, nem benchmark de malha. Os projetos originais e as entregas anteriores foram preservados.

Próxima comparação necessária: produzir nova fusão Revo da mesma cópia congelada e confrontá-la com um candidato nativo que separe camadas e preserve detalhes, incluindo cortes 3D de nervuras/aberturas e ruído encostado. Tempo e qualidade precisam ser aprovados separadamente.
