"""Busca uma regra (limiar numa caracteristica da cena) que escolha entre 'plano' e 'sem base' por scan.

Ganho = quanto o plano limpa a mais (mantido_sem - mantido_plano) quando e seguro;
penalidade = perda de recall do plano em relacao ao sem base (peso alto: perder peca estraga a malha).
Validacao: leave-one-project-out (o limiar e escolhido sem ver o projeto avaliado).
"""
import json, sys, itertools
import numpy as np
c = {(x['proj'], x['scan']): x for x in json.load(open('aprendizado/caracteristicas.json'))}
r = json.load(open('aprendizado/aval_02.json'))
P = r['plano,selecao=hibrido,dominancia=0.5']; S = r['sem_base,selecao=hibrido']
nomes = ['frac_plano','cruza','acima','abaixo','razao_cel','plano_cobre','h01','h50','h90','h99','ext_plano_max','ext_plano_min','achatamento','mira_h_med','mira_h_dp','esp']
rows = [(c[(p['proj'], p['scan'])], p, s) for p, s in zip(P, S)]
PEN = 20.0
def valor(usa_plano, p, s):
    if not usa_plano: return 0.0
    ganho = s['mantido'] - p['mantido']
    perda = max(0.0, s['recall'] - p['recall'])
    return ganho - PEN * perda
def melhor_regra(idx):
    best = (0.0, None)  # (valor, regra) ; regra None = nunca usar plano
    for n in nomes:
        v = np.array([rows[i][0][n] for i in idx])
        for q in np.unique(np.percentile(v, np.arange(5, 100, 5))):
            for sinal in (1, -1):
                tot = sum(valor((sinal * (rows[i][0][n] - q)) > 0, rows[i][1], rows[i][2]) for i in idx)
                if tot > best[0]: best = (tot, (n, q, sinal))
    return best
todos = list(range(len(rows)))
tot, regra = melhor_regra(todos)
print("regra no conjunto todo:", regra, "valor", round(tot, 3), "| usar plano sempre:", round(sum(valor(True, rows[i][1], rows[i][2]) for i in todos), 3))
projs = sorted(set(rows[i][0]['proj'] for i in todos))
ganho_cv = 0; usos = 0; perdas = 0
for pr in projs:
    tr = [i for i in todos if rows[i][0]['proj'] != pr]; te = [i for i in todos if rows[i][0]['proj'] == pr]
    _, rg = melhor_regra(tr)
    for i in te:
        usa = rg is not None and (rg[2] * (rows[i][0][rg[0]] - rg[1])) > 0
        ganho_cv += valor(usa, rows[i][1], rows[i][2]); usos += usa
        if usa: perdas += max(0, rows[i][2]['recall'] - rows[i][1]['recall'])
print(f"leave-one-project-out: valor {ganho_cv:.3f} | scans que usariam o plano: {usos} de {len(todos)} | perda de recall acumulada {perdas:.3f}")
ganho_pos = [rows[i][2]['mantido'] - rows[i][1]['mantido'] for i in todos if rows[i][1]['recall'] >= rows[i][2]['recall'] - 0.02]
print("quando o plano e inofensivo, limpa a mais em media:", round(np.mean(ganho_pos) * 100, 2), "% dos pontos | max", round(np.max(ganho_pos) * 100, 1), "%")
