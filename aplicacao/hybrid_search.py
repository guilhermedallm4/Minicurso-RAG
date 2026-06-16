"""
Busca Híbrida: BM25 + Semântica
=============================================================================
Conceito — Módulo Avançado

Retrieval híbrido combina dois paradigmas complementares:

  BM25 (esparso / léxico):
    - Baseado em frequência de termos (TF-IDF melhorado)
    - Excelente para keywords exatas, siglas e termos técnicos
    - Ex.: "PPGC", "RU", "matrícula 2024.2" → BM25 vence

  Semântico (denso):
    - Baseado em embeddings de significado
    - Excelente para paráfrases e sinônimos
    - Ex.: "quando posso me inscrever?" ≈ "prazo de matrícula" → semântico vence

  Fusão via RRF (Reciprocal Rank Fusion):
    score_rrf(doc) = Σ  1 / (k + rank_i)   onde k=60
    Combina os rankings de cada retriever sem normalizar escalas diferentes.

  Parâmetro alpha:
    alpha=1.0 → puramente semântico
    alpha=0.5 → peso igual (recomendado como ponto de partida)
    alpha=0.0 → puramente BM25
"""
import json
import psycopg2
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever

import config
from store import get_vector_store


def load_all_docs() -> List[Document]:
    """Carrega todos os documentos da coleção atual do pgvector."""
    conn = psycopg2.connect(**config.DB_CONFIG)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """, (config.COLLECTION_NAME,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    docs = []
    for text, meta_raw in rows:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def build_hybrid_retriever(
    store: PGVector = None,
    alpha: float = 0.5,
    top_k: int = 5,
) -> EnsembleRetriever:
    """
    Constrói o EnsembleRetriever combinando BM25 e busca semântica.

    Args:
        store  : PGVector store (criado automaticamente se None)
        alpha  : peso da busca semântica (1-alpha vai para BM25)
        top_k  : documentos retornados por cada retriever antes da fusão
    """
    store = store or get_vector_store()
    docs = load_all_docs()

    if not docs:
        raise RuntimeError(
            "Nenhum documento encontrado. Execute 'python store.py' primeiro."
        )

    bm25_retriever = BM25Retriever.from_documents(docs, k=top_k)
    semantic_retriever = store.as_retriever(search_kwargs={"k": top_k})

    return EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[round(1 - alpha, 2), round(alpha, 2)],
    )


def hybrid_search(
    query: str,
    store: PGVector = None,
    alpha: float = 0.5,
    top_k: int = 5,
) -> List[Document]:
    """Executa busca híbrida e retorna documentos fundidos via RRF."""
    retriever = build_hybrid_retriever(store, alpha=alpha, top_k=top_k)
    return retriever.invoke(query)


def demo_hibrido(query: str = "prazo de matrícula"):
    """Compara BM25, semântico e híbrido para a mesma query."""
    print("=" * 62)
    print("  MÓDULO AVANÇADO — Busca Híbrida (BM25 + Semântica)")
    print("=" * 62)
    print(f"\n  Query: '{query}'\n")

    store = get_vector_store()
    all_docs = load_all_docs()

    # --- BM25 puro ---
    bm25 = BM25Retriever.from_documents(all_docs, k=3)
    bm25_results = bm25.invoke(query)
    print("--- BM25 (léxico) ---")
    for i, d in enumerate(bm25_results, 1):
        src = d.metadata.get("source", "?")
        print(f"  [{i}] [{src}] {d.page_content[:70]}...")

    # --- Semântico puro ---
    sem_results = store.similarity_search(query, k=3)
    print("\n--- Semântico (denso) ---")
    for i, d in enumerate(sem_results, 1):
        src = d.metadata.get("source", "?")
        print(f"  [{i}] [{src}] {d.page_content[:70]}...")

    # --- Híbrido RRF ---
    hyb_results = hybrid_search(query, store, alpha=0.5, top_k=3)
    print("\n--- Híbrido RRF (alpha=0.5) ---")
    for i, d in enumerate(hyb_results, 1):
        src = d.metadata.get("source", "?")
        print(f"  [{i}] [{src}] {d.page_content[:70]}...")


# =============================================================================
# Execução direta
# python hybrid_search.py
# python hybrid_search.py "horário da biblioteca"
# Pré-requisito: python store.py
# =============================================================================
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "prazo de matrícula"
    demo_hibrido(query)
