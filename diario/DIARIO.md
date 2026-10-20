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
