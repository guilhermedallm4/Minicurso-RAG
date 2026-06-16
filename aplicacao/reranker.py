"""
Reranking de Documentos
=============================================================================
Conceito — Módulo Avançado

Problema: bi-encoders (embedding) codificam query e documento de forma
INDEPENDENTE — rápido, mas perde nuances da interação entre os dois.

Solução — two-stage pipeline:
  Etapa 1 — Retrieval (bi-encoder)  : recupera K candidatos rapidamente
  Etapa 2 — Reranking (cross-encoder): avalia cada par (query, doc) em
                                        conjunto, produzindo score mais preciso

Cross-encoder:
  Concatena [query; separador; doc] e passa pelo transformer completo.
  Score = relevância real do documento para aquela query específica.
  Mais lento que bi-encoder (O(K) inferências), mas só roda sobre os K
  candidatos já filtrados — custo gerenciável.

Opções de reranker:
  NVIDIA NIM  : llama-3.2-nv-rerankqa-1b-v2 (via API, sem GPU local)
  HuggingFace : cross-encoder/ms-marco-MiniLM-L-6-v2 (~22 MB, local)
"""
import os
from typing import List

from langchain_core.documents import Document

import aplicacao.config as config
from aplicacao.providers import NVIDIA_AVAILABLE

# Importações condicionais para evitar falha se pacote não estiver instalado
try:
    from langchain_nvidia_ai_endpoints import NVIDIARerank
    _nvidia_rerank_ok = NVIDIA_AVAILABLE
except ImportError:
    _nvidia_rerank_ok = False

try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
    _crossencoder_ok = True
except ImportError:
    _crossencoder_ok = False

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_NVIDIA_RERANK_MODEL  = "nvidia/llama-3.2-nv-rerankqa-1b-v2"

# Cache do cross-encoder local para não recarregar a cada chamada
_cross_encoder_cache = None


def _get_cross_encoder():
    global _cross_encoder_cache
    if _cross_encoder_cache is None:
        print(f"[Reranker] Carregando cross-encoder '{_CROSS_ENCODER_MODEL}'...")
        _cross_encoder_cache = _CrossEncoder(_CROSS_ENCODER_MODEL)
    return _cross_encoder_cache


def rerank_documents(
    query: str,
    docs: List[Document],
    top_k: int = 3,
) -> List[Document]:
    """
    Reordena documentos pelo score de relevância para a query.

    Usa NVIDIA NIM (sem GPU) como opção primária.
    Cai para cross-encoder local como fallback.

    Args:
        query : pergunta do usuário
        docs  : lista de candidatos do retriever inicial
        top_k : quantos documentos retornar após reranking
    """
    if not docs:
        return []

    if _nvidia_rerank_ok:
        print(f"[Reranker] NVIDIA NIM: {_NVIDIA_RERANK_MODEL}")
        try:
            reranker = NVIDIARerank(
                model=_NVIDIA_RERANK_MODEL,
                api_key=os.getenv("NVIDIA_API_KEY"),
                top_n=top_k,
            )
            return reranker.compress_documents(docs, query)
        except Exception as e:
            print(f"[Reranker] NVIDIA indisponível ({e!s:.80}), usando cross-encoder local.")

    if _crossencoder_ok:
        print(f"[Reranker] CrossEncoder local: {_CROSS_ENCODER_MODEL}")
        model = _get_cross_encoder()
        pairs = [[query, d.page_content] for d in docs]
        scores = model.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        for doc, score in ranked:
            doc.metadata["rerank_score"] = round(float(score), 4)
        return [doc for doc, _ in ranked[:top_k]]

    print("[Reranker] Nenhum reranker disponível — retornando ordem original.")
    return docs[:top_k]


def demo_reranker(query: str = "Qual o prazo para entregar a dissertação?"):
    """Demonstra o impacto do reranking comparando antes e depois."""
    from aplicacao.store import get_vector_store

    print("=" * 62)
    print("  MÓDULO AVANÇADO — Reranking de Documentos")
    print("=" * 62)
    print(f"\n  Query: '{query}'\n")

    store = get_vector_store()
    candidates = store.similarity_search(query, k=6)

    print("--- Antes do reranking (ordem do retriever semântico) ---")
    for i, d in enumerate(candidates, 1):
        src = d.metadata.get("source", "?")
        print(f"  [{i}] [{src}] {d.page_content[:70]}...")

    reranked = rerank_documents(query, candidates, top_k=3)

    print("\n--- Após reranking (top-3 selecionados pelo cross-encoder) ---")
    for i, d in enumerate(reranked, 1):
        score = d.metadata.get("rerank_score", "n/a")
        src = d.metadata.get("source", "?")
        print(f"  [{i}] score={score} | [{src}] {d.page_content[:70]}...")


# =============================================================================
# Execução direta
# python reranker.py
# python reranker.py "horário da biblioteca"
# Pré-requisito: python store.py
# =============================================================================
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 \
        else "Qual o prazo para entregar a dissertação?"
    demo_reranker(query)
