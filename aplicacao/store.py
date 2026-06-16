"""
Banco Vetorial — pgvector
=============================================================================
Conceito — SEÇÃO 4 do curso

pgvector é uma extensão do PostgreSQL que adiciona:
  - Tipo de dado VECTOR(n) para armazenar embeddings
  - Índices HNSW e IVFFlat para busca aproximada eficiente
  - Operadores de distância: <=>, <->, <#>, <+>

O LangChain PGVector gerencia automaticamente as tabelas:
  langchain_pg_collection  : registro das coleções (namespaces)
  langchain_pg_embedding   : vetores + texto + metadados

Pipeline de ingestão (processo OFFLINE, executado uma vez):
  texto → chunking → embedding → inserção no pgvector
"""
from typing import List, Optional

from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

import config
from providers import get_embeddings
from chunking import chunk_documents, chunk_texts


def get_vector_store(
    embeddings=None,
    collection_name: Optional[str] = None,
) -> PGVector:
    """Conecta (ou cria) a coleção vetorial no pgvector."""
    return PGVector(
        collection_name=collection_name or config.COLLECTION_NAME,
        connection_string=config.CONNECTION_STRING,
        embedding_function=embeddings or get_embeddings(),
        use_jsonb=True,
    )


def ingest_documents(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    reset: bool = False,
) -> PGVector:
    """
    Ingesta textos na base vetorial.

    Args:
        texts     : lista de textos a indexar
        metadatas : metadados opcionais por documento (source, categoria, etc.)
        reset     : se True, apaga a coleção antes de inserir
    """
    embeddings = get_embeddings()
    chunks = chunk_texts(texts, metadatas)

    print(f"[Ingestão] Gerando embeddings e inserindo {len(chunks)} chunks...")
    store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection_string=config.CONNECTION_STRING,
        pre_delete_collection=reset,
        use_jsonb=True,
    )
    print("[Ingestão] Concluída!")
    return store


def ingest_pdf(
    path: str,
    source_name: Optional[str] = None,
    titulo: Optional[str] = None,
) -> PGVector:
    """Carrega um PDF e o ingesta na base vetorial."""
    loader = PyPDFLoader(path)
    documents = loader.load()
    for d in documents:
        if source_name:
            d.metadata["source"] = source_name
        if titulo:
            d.metadata["titulo"] = titulo
    chunks = chunk_documents(documents)
    embeddings = get_embeddings()
    print(f"[PDF] Ingerindo '{path}' → {len(chunks)} chunks...")
    store = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.COLLECTION_NAME,
        connection_string=config.CONNECTION_STRING,
        pre_delete_collection=False,
        use_jsonb=True,
    )
    print("[PDF] Concluído!")
    return store


# Documentos institucionais de exemplo usados nas demos
SAMPLE_DOCS = [
    "A matrícula para o semestre 2024.2 ocorre entre 01 e 10 de julho. "
    "Alunos com débitos na biblioteca não poderão realizar matrícula.",

    "O Programa de Pós-Graduação em Computação (PPGC) oferece mestrado e "
    "doutorado em Inteligência Artificial, Sistemas Distribuídos e Segurança.",

    "O prazo para entrega da dissertação de mestrado é de 24 meses após o "
    "ingresso. Prorrogações de até 6 meses devem ser solicitadas ao colegiado.",

    "A biblioteca universitária funciona de segunda a sexta, das 8h às 22h, "
    "e aos sábados das 9h às 17h. O empréstimo exige a carteira de estudante.",

    "O Restaurante Universitário (RU) oferece refeições subsidiadas para "
    "alunos regularmente matriculados. Cadastro na secretaria acadêmica.",
]

SAMPLE_METADATAS = [
    {"source": "manual_matricula_2024.pdf", "categoria": "matricula",     "titulo": "Manual de Matrícula 2024"},
    {"source": "regulamento_ppgc.pdf",      "categoria": "pos_graduacao", "titulo": "Regulamento do PPGC"},
    {"source": "regulamento_ppgc.pdf",      "categoria": "pos_graduacao", "titulo": "Regulamento do PPGC"},
    {"source": "guia_biblioteca.pdf",        "categoria": "servicos",      "titulo": "Guia da Biblioteca Universitária"},
    {"source": "guia_ru.pdf",               "categoria": "servicos",      "titulo": "Guia do Restaurante Universitário"},
]


def demo_ingestao(reset: bool = False):
    """Ingesta os documentos de exemplo e confirma os registros no banco."""
    print("=" * 60)
    print("  SEÇÃO 4 — Banco Vetorial (pgvector)")
    print("=" * 60)
    ingest_documents(SAMPLE_DOCS, SAMPLE_METADATAS, reset=reset)


# =============================================================================
# Execução direta: ingesta os documentos de exemplo
# python store.py
# python store.py --reset   (recria a coleção do zero)
# =============================================================================
if __name__ == "__main__":
    import sys
    reset = "--reset" in sys.argv
    demo_ingestao(reset=reset)
