"""Inspecao final por LLM (OPCIONAL, uma unica chamada).

A LLM nao escolhe parametros. Ela recebe o render das 3 vistas + metricas e
responde em JSON se aprova e QUAL problema ve. Cada tipo de problema mapeia
para um ajuste deterministico (AJUSTES abaixo). No maximo 1 nova rodada.
"""
import base64
import json
import os

TIPOS = ["base_residual", "ruido_residual", "aresta_cortada", "superficie_duplicada",
         "trecho_deslocado", "outro"]

# Problema apontado -> ajuste de parametro (delta)
AJUSTES = {
    "base_residual": {"base_dist_mult": +0.8, "base_margin_mult": +0.8},
    "ruido_residual": {"sor_std": -0.4, "dbscan_eps_mult": -0.5},
    "aresta_cortada": {"sor_std": +0.5, "base_margin_mult": -0.5},
}

PROMPT = """Voce e um inspetor de qualidade de nuvens de pontos de escaneamento 3D.
A imagem mostra a nuvem ja limpa em 3 vistas ortogonais (topo, frente, lateral).
Cinza = peca. Verde = trecho que foi realinhado apos perda de rastreamento.
Metricas calculadas: {metricas}

Avalie se a nuvem esta pronta para gerar malha. Procure:
- base_residual: restos da mesa/base plana sob a peca
- ruido_residual: pontos soltos ou nuvens de pontos flutuando
- aresta_cortada: bordas ou detalhes finos da peca que parecem ter sido removidos
- superficie_duplicada: mesma superficie aparecendo em duas camadas paralelas
- trecho_deslocado: parte da peca visivelmente fora de posicao
Responda SOMENTE com JSON valido, sem texto fora dele:
{{"aprovado": true|false, "problemas": [{{"tipo": "<um de: {tipos}>", "descricao": "<curta>"}}]}}"""


def inspect(image_path, metrics, model=None):
    """Retorna dict {"aprovado": bool, "problemas": [...]} ou None se indisponivel."""
    try:
        import anthropic
    except ImportError:
        return None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    model = model or os.environ.get("MOVIAI_LLM_MODEL", "claude-sonnet-5-5")
    with open(image_path, "rb") as f:
        img = base64.standard_b64encode(f.read()).decode()
    m_publicas = {k: v for k, v in metrics.items() if not k.startswith("_")}
    msg = anthropic.Anthropic().messages.create(
        model=model, max_tokens=600,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img}},
            {"type": "text", "text": PROMPT.format(metricas=json.dumps(m_publicas, ensure_ascii=False),
                                                   tipos=", ".join(TIPOS))},
        ]}])
    txt = msg.content[0].text.strip()
    txt = txt[txt.find("{"): txt.rfind("}") + 1]
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"aprovado": False, "problemas": [{"tipo": "outro", "descricao": "resposta invalida da LLM"}]}


def apply_feedback(params, verdict):
    """Aplica os ajustes mapeados. Retorna (novos_params, mudou?)."""
    from dataclasses import replace
    deltas = {}
    for p in verdict.get("problemas", []):
        for k, dv in AJUSTES.get(p.get("tipo"), {}).items():
            deltas[k] = deltas.get(k, 0) + dv
    if not deltas:
        return params, False
    new = {k: max(0.3, getattr(params, k) + dv) for k, dv in deltas.items()}
    return replace(params, **new), True
