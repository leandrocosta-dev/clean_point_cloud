# pipeline/legado — scripts que NÃO devem ser usados

Estão aqui para manter o histórico e permitir reproduzir as medidas antigas. Nenhuma receita do
`diario/GUIA_PIPELINE.md` depende deles. A GUI mostra todos na categoria "Legado (não usar)".
Os imports para módulos de `pipeline/` foram ajustados para continuar funcionando a partir desta pasta.

| script | por que está aqui | referência no guia |
|---|---|---|
| `corrigir_interna_local.py` | a mediana por vértice (raios de 25/10/5 mm) deixou a interna áspera: rugosidade de 36 para 423 µm. Foi substituído por `corrigir_interna_suave.py` | "CORRECAO: `corrigir_interna_local.py` DEIXOU A INTERNA ASPERA" |
| `encaixar_espessura.py` | empurra vértices para a espessura nominal e deixa 13% por fora; a espessura passa a ser nominal, não medida | "Interna x externa: espessura..." (não usar sem avisar) |
| `deslocar_interna.py` | deslocamento pela espessura nominal de 2,8 mm, medida em outro scan; substituído pelo alinhamento por scan-ponte | "Interna alinhada por scan-PONTE (substitui o metodo de espessura nominal)" |
| `fechar_furos_com_dado.py` | fechou aberturas reais da grade (v15/v15b/v16 descartadas pelo usuário) | "CORRECAO: fechar furos automaticamente FOI ERRADO" |
| `buracos.py` | furos por contorno não servem para casca aberta; o útil é `faltando.py` | seção 1, linha de `avaliar_fusao.py`, `faltando.py`, `buracos.py` |
| `poisson_de_pedacos.py` | Poisson sobre as peças da construção por cobertura amplifica os degraus de emenda (528 µm) | "DESCOBERTA CENTRAL... Poisson sobre as pecas NAO ajuda" |
| `exp_pesos.py` | experimento pontual dos pesos de odometria do grafo de segmentos; o alinhamento válido é `alinhar_direto.py` | seção 3 ("O refino por pares fica longe do otimo; usar `alinhar_direto`") |

Mantidos em `pipeline/`, embora sejam citados como ferramenta não padrão:
- `remover_remendos_discordantes.py`: faz parte da regra da v19 (`fechar_furos_pequenos.py` seguido de
  remover remendos discordantes), que levou à v20 aprovada pelo usuário.
- `componentes.py`: serve como etapa de lixo solto antes do consenso (veredito de 2026-11-08).
