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
# Roteamento via LLM
# ---------------------------------------------------------------------------

def route_query(query: str) -> Optional[str]:
    """
    Usa o LLM para mapear a pergunta à coleção mais adequada.

    Retorna o nome da coleção (str) ou None quando a confiança for baixa
    — sinal para o chamador acionar o fallback multi-coleção.
    """
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
