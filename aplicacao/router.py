"""
Roteador LLM — direciona perguntas para a coleção correta no pgvector.
=============================================================================

Fluxo:
  pergunta → LLM classifica → nome da coleção
                ↓ (confiança baixa ou coleção desconhecida)
           busca em TODAS as coleções → melhores K resultados (fallback)

Threshold de confiança: 0.6
  Abaixo disso considera-se que o LLM não conseguiu inferir a coleção
  e o sistema cai no fallback multi-coleção.
"""
import json
import re
from typing import Optional

from langchain_community.vectorstores import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import config
from providers import get_llm, get_embeddings

# ---------------------------------------------------------------------------
# Descrições derivadas dinamicamente de config.COLLECTION_DESCRIPTIONS
# Para adicionar um novo tipo, basta atualizar config.py — o roteador
# detecta as novas coleções sem precisar de alteração aqui.
# ---------------------------------------------------------------------------

def _build_collections_text() -> str:
    """Constrói o bloco de texto de coleções usado no prompt do roteador."""
    return "\n".join(
        f'- "{name}": {desc}'
        for name, desc in config.COLLECTION_DESCRIPTIONS.items()
    )

_COLLECTIONS_TEXT = _build_collections_text()

_ROUTER_PROMPT = ChatPromptTemplate.from_template(
    "Você é um roteador de consultas para o portal institucional da UFPel. "
    "Dada a pergunta abaixo, escolha UMA coleção de conhecimento.\n\n"
    "Coleções disponíveis:\n{collections}\n\n"
    "Pergunta: {query}\n\n"
    "Responda SOMENTE com JSON válido, sem texto adicional:\n"
    '  {{"collection": "nome_da_colecao", "confidence": 0.9}}\n\n'
    "Use confidence >= 0.7 apenas quando a pergunta claramente pertence "
    "a uma categoria. Se incerto, retorne:\n"
    '  {{"collection": null, "confidence": 0.0}}'
)

_CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Roteamento por regras (antes do LLM) — sem custo de API
# ---------------------------------------------------------------------------

_RULE_ROUTES: list[tuple[re.Pattern, str]] = [
    # Perguntas que mencionam "professor/docente" + área/atividade → servidores
    (re.compile(
        r"\b(?:professor[a]?|prof\.?|docente|servidor[a]?|pesquisador[a]?)\b.{0,80}"
        r"\b(?:atua|trabalha|pesquisa|estuda|área|especialidade|lattes|titulação|cargo|lotação)\b",
        re.IGNORECASE,
    ), "ufpel_servidores"),
    (re.compile(
        r"\b(?:atua|trabalha)\b.{0,60}"
        r"\b(?:professor[a]?|prof\.?|docente|pesquisador[a]?)\b",
        re.IGNORECASE,
    ), "ufpel_servidores"),
    # "qual/quais professor/docente/pesquisador ..." — busca de pessoa
    (re.compile(
        r"\bqual(?:is)?\b.{0,50}\b(?:professor[a]?|prof\.?|docente|pesquisador[a]?)\b",
        re.IGNORECASE,
    ), "ufpel_servidores"),
    # "quem pesquisa/atua/trabalha com X" → servidores (pessoa, não projeto)
    (re.compile(
        r"\bquem\b.{0,30}\b(?:pesquisa|atua|trabalha|estuda)\b",
        re.IGNORECASE,
    ), "ufpel_servidores"),
    # Projetos de pesquisa (somente se não menciona professor)
    (re.compile(r"\b(?:projeto[s]?|extensão|TCC|dissertação|tese)\b", re.IGNORECASE),
     "ufpel_projetos"),
    # Unidades / centros
    (re.compile(r"\b(?:unidade[s]?|centro[s]?|instituto[s]?|faculdade[s]?|câmpus|endereço)\b", re.IGNORECASE),
     "ufpel_unidades"),
]


def _rule_route(query: str) -> Optional[str]:
    """Roteamento rápido por regex sem custo de LLM. Retorna None se não aplicável."""
    for pattern, collection in _RULE_ROUTES:
        if pattern.search(query):
            print(f"[Router:regra] → {collection}")
            return collection
    return None


# ---------------------------------------------------------------------------
# Roteamento via LLM
# ---------------------------------------------------------------------------

def route_query(query: str) -> Optional[str]:
    """
    Usa o LLM para mapear a pergunta à coleção mais adequada.

    Retorna o nome da coleção (str) ou None quando a confiança for baixa
    — sinal para o chamador acionar o fallback multi-coleção.
    """
    # Tenta regras rápidas antes de chamar o LLM (sem custo de API)
    rule_result = _rule_route(query)
    if rule_result:
        return rule_result

    llm = get_llm()
    chain = _ROUTER_PROMPT | llm | StrOutputParser()
    raw = chain.invoke({"query": query, "collections": _COLLECTIONS_TEXT})

    try:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            return None
        data = json.loads(match.group())
        collection = data.get("collection")
        confidence = float(data.get("confidence", 0.0))
        if collection and confidence >= _CONFIDENCE_THRESHOLD:
            if collection in config.ALL_COLLECTIONS:
                print(f"[Router] → {collection}  (confidence={confidence:.2f})")
                return collection
            print(f"[Router] Coleção desconhecida '{collection}' — usando fallback")
    except Exception as exc:
        print(f"[Router] Erro ao parsear resposta: {exc} | raw={raw[:120]}")

    print("[Router] Confiança insuficiente — fallback multi-coleção")
    return None


# ---------------------------------------------------------------------------
# Fallback: busca em todas as coleções
# ---------------------------------------------------------------------------

def search_all_collections(
    query: str,
    top_k: int = 4,
    embeddings=None,
) -> list[tuple]:
    """
    Busca em todas as coleções UFPel e retorna os `top_k` resultados
    mais relevantes globalmente.

    Cada coleção contribui com até 2 candidatos; o ranking final é por
    score de relevância (cosseno) decrescente.
    """
    emb = embeddings or get_embeddings()
    all_results: list[tuple] = []

    for collection_name in config.ALL_COLLECTIONS:
        try:
            store = PGVector(
                collection_name=collection_name,
                connection_string=config.CONNECTION_STRING,
                embedding_function=emb,
                use_jsonb=True,
            )
            results = store.similarity_search_with_relevance_scores(query, k=2)
            all_results.extend(results)
        except Exception as exc:
            print(f"[Router] Erro em '{collection_name}': {exc}")

    all_results.sort(key=lambda x: x[1], reverse=True)
    top = all_results[:top_k]
    if top:
        best = top[0][1]
        cols = {d.metadata.get("categoria", d.metadata.get("tipo", "?")) for d, _ in top}
        print(f"[Router] Fallback: {len(top)} docs de {cols}  (melhor score={best:.3f})")
    return top


# ---------------------------------------------------------------------------
# Execução direta — teste rápido do roteador
# python router.py "qual a ementa de cálculo?"
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Quais disciplinas têm ementa de IA?"
    print(f"\nPergunta: {q}")
    col = route_query(q)
    print(f"Coleção selecionada: {col or '(fallback)'}")
    if col is None:
        docs = search_all_collections(q, top_k=3)
        for doc, score in docs:
            print(f"  score={score:.3f} | {doc.page_content[:80]}...")
