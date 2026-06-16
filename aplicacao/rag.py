"""
Ponto de entrada legado — mantido para compatibilidade.
O projeto foi modularizado. Use main.py para execução granular:

  python main.py                   # menu interativo
  python main.py --etapa busca     # só busca semântica
  python main.py --etapa chatbot   # só o chatbot
  python main.py --etapa tudo      # pipeline completo

Ou execute cada módulo diretamente:
  python providers.py              # testa embeddings e LLM
  python chunking.py               # demonstra segmentação
  python store.py                  # ingesta documentos de exemplo
  python search.py                 # busca semântica comparativa
  python pipeline.py               # pipeline RAG (uma pergunta)
  python chatbot.py                # chatbot interativo
"""

# Re-exports públicos — declara a API pública deste módulo de compatibilidade
from config import DB_CONFIG, CONNECTION_STRING, COLLECTION_NAME, RELEVANCE_THRESHOLD
from providers import get_embeddings, get_llm
from chunking import chunk_documents, chunk_texts
from store import get_vector_store, ingest_documents, ingest_pdf
from search import semantic_search, semantic_search_demo
from pipeline import RAGConfig, build_rag_chain, validate_and_answer
from hybrid_search import hybrid_search, build_hybrid_retriever
from reranker import rerank_documents
from guardrails import InputGuard, OutputGuard, GuardResult
from evaluation import mrr_at_k, bertscore_lite, bertscore_full, llm_judge, rouge_l_score
from chatbot import run_chatbot

__all__ = [
    # config
    "DB_CONFIG", "CONNECTION_STRING", "COLLECTION_NAME", "RELEVANCE_THRESHOLD",
    # providers
    "get_embeddings", "get_llm",
    # chunking
    "chunk_documents", "chunk_texts",
    # store
    "get_vector_store", "ingest_documents", "ingest_pdf",
    # search
    "semantic_search", "semantic_search_demo",
    # pipeline
    "RAGConfig", "build_rag_chain", "validate_and_answer",
    # avançados
    "hybrid_search", "build_hybrid_retriever",
    "rerank_documents",
    "InputGuard", "OutputGuard", "GuardResult",
    "mrr_at_k", "bertscore_lite", "bertscore_full", "llm_judge", "rouge_l_score",
    # interface
    "run_chatbot",
]

if __name__ == "__main__":
    from main import main
    main()
