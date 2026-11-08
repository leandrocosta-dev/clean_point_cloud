# Corte de apoio por scan: avaliação de 0,5 e 1,0 mm

O usuário confirmou que retirar uma faixa junto à mesa é aceitável para remover resíduos encostados na peça, aproveitando outras posições para recuperar a superfície real. Isso não autoriza supor que toda região cortada está coberta em outra vista.

Foi acrescentado `pipeline/avaliar_corte_apoio.py`: estima quatro planos candidatos nos previews originais, escolhe o de maior área, orienta a normal pela seleção limpa e verifica se ele atravessa essa seleção. Para esta avaliação: RANSAC com faixa de 0,4 mm; dispersão MAD dos planos escolhidos de 0,130 a 0,149 mm. A identidade física do plano ainda requer conferência geométrica.

Dados: `D:/Projects/Project20261106_210152`, base limpa `pipeline/output/Project20261106_210152_acesso_corrigido`, poses já usadas na união em `Project20261106_210152_UNIDO_FINAL/alinhamento.json`. Amostra de um a cada quatro quadros e pixels, agregada em centroides de 0,15 mm: 241.325, 235.287 e 233.629 centroides por scan. As coordenadas são transformadas para o referencial comum somente para medir cobertura.

Para cada altura, os pontos da faixa cortada são comparados aos OUTROS scans depois de aplicar a mesma altura nos respectivos planos. Assim uma observação que também seria cortada na outra vista não conta como recuperação.

| Scan | Corte | Centroides retirados da seleção atual | Com observação em outra vista a até 0,3 mm | A até 0,5 mm |
|---|---:|---:|---:|---:|
| 210152 | 0,5 mm | 67 | 100,00% | 100,00% |
| 210152 | 1,0 mm | 1.883 | 100,00% | 100,00% |
| 210505 | 0,5 mm | 2.530 | 99,92% | 100,00% |
| 210505 | 1,0 mm | 6.382 | 99,89% | 100,00% |
| 210900 | 0,5 mm | 1.871 | 84,18% | 87,39% |
| 210900 | 1,0 mm | 3.893 | 92,29% | 93,89% |

No scan 210900, a faixa ADICIONAL entre 0,5 e 1,0 mm tem 2.022 centroides: 99,80% apoiados a 0,3 mm e 99,90% a 0,5 mm. A maior parte da ausência de confirmação está, portanto, na faixa mais próxima do apoio. Isso é compatível tanto com mesa residual quanto com uma região da peça sem cobertura; não distingue as duas hipóteses sozinho.

Conclusão operacional: o corte de apoio é um candidato mais localizado que excluir a camada laranja inteira. Os dois primeiros scans mostram forte possibilidade de recuperar as regiões afetadas. A faixa junto ao apoio no terceiro exige inspeção por região. Cobertura amostrada e tolerâncias de 0,3/0,5 mm não demonstram exatidão absoluta nem conservação de todo detalhe menor que essa escala.

Nenhum frame foi alterado nesta avaliação. Ainda não foram aplicados estes cortes, refeitas fusões ou aprovadas máscaras de mesa. Originais e entregas anteriores preservados. Resultados integrais, planos e amostras: `pipeline/output/avaliacao_apoio_05_10_20261108/avaliacao.json` e os três arquivos NPZ ao lado. Dois testes de recuperação/falta de cobertura e `--help` aprovados.

```powershell
.\.venv\Scripts\python.exe pipeline/avaliar_corte_apoio.py D:/Projects/Project20261106_210152 pipeline/output/Project20261106_210152_acesso_corrigido --poses pipeline/output/Project20261106_210152_UNIDO_FINAL/alinhamento.json --saida pipeline/output/avaliacao_apoio_nova
```
