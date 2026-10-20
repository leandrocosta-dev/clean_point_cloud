# Diario de bordo

Cada aprendizado vira uma entrada com: o que se observou, a evidencia (numeros), a decisao, a versao (tag git) e o que ficou em aberto.
O sistema deve ser reavaliado contra TODOS os projetos de `D:\Projects` a cada mudanca (`pipeline/aprender.py`), nunca so contra o ultimo projeto que falhou.

Gabarito objetivo: o `fuse.ply` que o proprio Revo Scan gerou (quando registra bem nos quadros, fitness >= 0,85) mostra o que e peca.
Metricas: **recall** = parte do fuse.ply que continua nos quadros limpos (perder isso estraga a malha); **lixo** = pontos mantidos longe do fuse.ply.

---

## v0.1 (base) - motor quadro a quadro
- Formato do `.revo` 1.3 decodificado (frames.dataset = SQLite com assinatura trocada; mascara de bits + profundidade por quadro; poses 4x4).
- Limpeza apaga pixels de profundidade em cada quadro, sem mexer nas poses; copia o projeto, original intacto.
- Pressuposto embutido: **existe uma mesa/base plana** (RANSAC) e a peca esta acima dela.
- Evidencia de que o pressuposto falha: busto sem mesa (`Project_Example`), o plano atravessou o corpo e cortou tudo abaixo, deixando uma ilha solta.

## v0.2 - harness de aprendizado em todos os projetos (`pipeline/aprender.py`)
- Inventario: 349 scans em 56 projetos de `D:\Projects`, ~220 com `fuse.ply`. Scan sem raw_preview (nos `Frame...`, tipo 5, estrutura de marcadores) nao e cenario de limpeza.
- `extrair` guarda por scan uma amostra leve (A: entende a cena; E: avalia) + o `fuse.ply` do Revo alinhado por ICP com inicio na identidade. Fitness >= 0,85 vira gabarito.
- `avaliar` mede variantes de pipeline sem gerar `.revo`: recall do fuse (perder isso estraga a malha) e lixo (>3 mm do fuse).
- Cuidado: o fuse do Revo pode incluir a mesa; o recall de scans assim e pessimista. Tratar com ressalva, olhar casos individualmente.

## Aprendizado 1 - o pressuposto "existe mesa plana" falha (busto sem mesa)
- Sintoma: `Project_Example`, plano de RANSAC (8% dos pontos) fatiou o busto em z=0; tudo abaixo foi cortado e sobrou ilha solta.
- Medida criada: `cruzamento_base` = fracao das celulas da peca com material DENSO tambem abaixo do plano. Mesa real ~0%; fatia da peca >=25% (busto 25%).
- Ressalva: o cruzamento sozinho NAO explica os piores cortes (ver aprendizado 2).

## Aprendizado 2 - a selecao da peca por "votos da mira" erra feio
- `20260403_085043`: escolheu o objeto de 6,6 mil pontos (11 miras) em vez do de 653 mil pontos (0 miras) que contem 94% do fuse. Recall do fuse 7,5%.
- Mesma causa provavel: `084717` do projeto 1011 (3% mantido) e varios scans do 1011.
- Hipotese: o maior componente deve vencer quando domina a cena; os votos so desempatam entre candidatos comparaveis ("hibrido"). Testado em `aval_02.json`.
