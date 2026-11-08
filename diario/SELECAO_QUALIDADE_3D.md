# Selecionar as melhores regioes de cada scan

Requisito de 08/11/2026: a limpeza deve consultar geometria 3D e a origem quadro/pixel, identificar deformacoes/perda de rastreio e priorizar a regiao boa de outra vista, preservando a cobertura da peca.

## Contrato de seguranca geometrica

- Originais/entregas sao somente leitura. A selecao e aplicada nos frames de uma revisao nova, auditada antes da publicacao.
- Comparar scans somente com poses fornecidas e validadas, no mesmo referencial e em milimetros. Deslocamento entre scans nao prova sozinho que um deles esteja errado.
- Qualidade por regiao combina dispersao local em 3D, consistencia/quantidade de quadros, incidencia e sinais de rastreio. Contagem de pontos isolada nao decide qualidade.
- Uma superficie de referencia deve ter evidencias temporais suficientes, normal confiavel, incidencia aceitavel e qualidade significativamente melhor. Nao atribuir qualidade boa apenas porque o scan e mais denso.
- A substituicao exige cobertura real em 3D pelo scan melhor, compatibilidade de normais com sinal e ausencia de ambiguidade. Faces opostas de paredes finas nao sao intercambiaveis.
- Rejeitar automaticamente apenas as observacoes comprovadamente redundantes e piores. Pontos exclusivos, bordas/ressaltos com normal ambigua e partes sem substituto confiavel permanecem, com pendencia quando houver suspeita de qualidade.
- A fonte escolhida nunca pode ser removida simultaneamente por uma escolha circular. Empates permanecem. Preservar os pontos bons; nao suavizar coordenadas nem preencher furos.
- Perda de rastreamento sem correspondencia confiavel gera pendencia/bloqueio de merge; nao inventar uma transformacao para ocultar deformacao. A selecao conservadora nao promete reparar todo rastreio perdido.
- Depois de aplicar a mascara, verificar a cobertura e a integridade dos frames. Qualquer perda de cobertura ou erro preserva a revisao ativa anterior.

## Artefatos revisaveis

- Relatorio por scan/regiao com score, ruido, incidencia, quadros, origem preferida, motivos, removidos/preservados e pendencias.
- Nuvens de regioes preferidas, redundantes ruins e sem substituto; consultas 3D integrais e HTML de navegacao com amostragem explicitada.
- Mascara de observacoes vinculada a hashes do dataset/calibracao/poses e indices quadro/pixel. Nao aplicar mascara produzida para outra revisao.
- Revisao nova invalida fusao, malha, comparacao, inspecao e aprovacao de merge. A avaliacao de qualidade sozinha nao aprova o resultado.

## Validacao minima

1. Regiao ruidosa sobreposta a uma boa: retirar somente a redundante ruim, manter a referencia boa e a cobertura.
2. Detalhe exclusivo da vista ruim: preservar e sinalizar, sem presumir recuperacao.
3. Faces opostas proximas: nenhuma substituicao entre elas.
4. Scans empatados/sem evidencia temporal/incidencia rasante/normais ambiguas: nao remover por prioridade arbitraria.
5. Rastreio suspeito e deslocamento sem correspondencia: pendencia, sem contaminacao de um merge aprovado.
6. Aplicacao preserva FrameInfo, poses, profundidades e pixels retidos; falha/mascara desatualizada nao publica revisao.

Limites numericos e de desempenho devem ser registrados com os testes. Exatidao absoluta e conferencia dimensional continuam dependendo de uma referencia fisica.
