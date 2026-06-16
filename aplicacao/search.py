"""
Busca Semântica e Métricas de Distância
=============================================================================
Conceito — SEÇÃO 5 do curso

Após converter a query em embedding, buscamos os K vetores mais próximos.
"Mais próximo" depende da métrica de distância escolhida:

  COSSENO  (<=>)
    Mede o ângulo entre dois vetores — ignora magnitude, foca direção.
    Ideal para texto: "banco de dados" ≈ "database" mesmo com comprimentos
    diferentes. Valor: 0 (idêntico) → 2 (oposto). PADRÃO DA INDÚSTRIA.

  EUCLIDIANA / L2  (<->)
    Distância "em linha reta" no espaço n-dimensional (regra de Pitágoras).
    Sensível à magnitude do vetor. Valor: 0 → ∞.

  MANHATTAN / L1  (<+>)
    Soma das diferenças absolutas por dimensão (distância de bloco).
    Menos sensível a outliers que L2. Requer pgvector ≥ 0.7.

  PRODUTO INTERNO NEGATIVO  (<#>)
    -dot(a, b). Em vetores normalizados equivale ao cosseno.
    Computacionalmente mais eficiente em índices HNSW com norma fixa.
"""
import psycopg2
from typing import List, Tuple

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document

import config
from providers import get_embeddings
from store import get_vector_store


DISTANCE_METRICS = {
    "cosine":    ("<=>", "Distância Cosseno          (padrão NLP)"),
    "euclidean": ("<->", "Distância Euclidiana (L2)"),
    "manhattan": ("<+>", "Distância Manhattan  (L1)  [pgvector ≥ 0.7]"),
    "inner":     ("<#>", "Produto Interno Negativo"),
}


def semantic_search(query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """
    Busca semântica por cosseno via LangChain.
    Retorna pares (documento, score_de_relevância) em ordem decrescente.
    score = 1 - distância_cosseno  →  1.0 = idêntico, 0.0 = sem relação.
    """
    store = get_vector_store()
    return store.similarity_search_with_relevance_scores(query, k=top_k)


def semantic_search_demo(query: str, top_k: int = 3):
    """
    Demonstração didática: compara as 4 métricas para a mesma query.
    Cada métrica usa uma conexão independente para evitar que um erro
    de SQL aborte a transação e contamine as consultas seguintes.
    """
    embeddings = get_embeddings()
    query_emb = embeddings.embed_query(query)
    vec_str = str(query_emb)

    print(f"\n{'='*62}")
    print(f"  Query: '{query}'")
    print(f"{'='*62}")

    for metric, (op, label) in DISTANCE_METRICS.items():
        conn = psycopg2.connect(**config.DB_CONFIG)
        cur = conn.cursor()
        try:
            cur.execute(f"""
                SELECT e.cmetadata->>'titulo' AS titulo,
                       e."document",
                       e.embedding {op} %s::vector AS dist
                FROM langchain_pg_embedding e
                JOIN langchain_pg_collection c ON e.collection_id = c.uuid
                WHERE c.name = %s
                ORDER BY e.embedding {op} %s::vector
                LIMIT %s;
            """, (vec_str, config.COLLECTION_NAME, vec_str, top_k))
            rows = cur.fetchall()
            print(f"\n--- {label} ---")
            for i, (titulo, doc, dist) in enumerate(rows, 1):
                preview = (doc[:60] + "...") if len(doc) > 60 else doc
                titulo_str = f"[{titulo}] " if titulo else ""
                print(f"  [{i}] dist={dist:+.4f} | {titulo_str}{preview}")
        except Exception as e:
            first_line = str(e).split("\n")[0]
            print(f"\n--- {label} ---")
            print(f"  (não disponível nesta versão do pgvector: {first_line})")
        finally:
            cur.close()
            conn.close()


def demo_busca(query: str = "prazo para entrega da dissertação de mestrado"):
    """Executa a demonstração comparativa de métricas."""
    print("=" * 62)
    print("  SEÇÃO 5 — Busca Semântica e Métricas de Distância")
    print("=" * 62)
    semantic_search_demo(query)

    print("\n--- Score de relevância LangChain (cosseno) ---")
    results = semantic_search(query, top_k=3)
    for doc, score in results:
        titulo = doc.metadata.get("titulo", "")
        source = doc.metadata.get("source", "?")
        titulo_str = f"[{titulo}] " if titulo else ""
        print(f"  score={score:.3f} | {titulo_str}[{source}] {doc.page_content[:60]}...")


# =============================================================================
# Execução direta: demonstra busca semântica com comparação de métricas
# python search.py
# python search.py "horário da biblioteca"
# Pré-requisito: python store.py (para popular o banco)
# =============================================================================
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 \
        else "prazo para entrega da dissertação de mestrado"
    demo_busca(query)
