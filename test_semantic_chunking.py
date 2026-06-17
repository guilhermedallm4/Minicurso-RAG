#!/usr/bin/env python3
"""
Script de teste para validar o chunking semântico.
Compara chunking semântico vs. recursivo em dados reais.
"""
import json
import sys
from pathlib import Path

# Imports do projeto
_APP_DIR = Path(__file__).resolve().parent / "aplicacao"
sys.path.insert(0, str(_APP_DIR))

import config
from chunking import chunk_documents
from chunking_semantico import chunk_documents_semantic
from langchain_core.documents import Document


def load_sample_documents(json_path: str, limit: int = 5) -> list[Document]:
    """Carrega documentos de amostra do JSON crawleado."""
    with open(json_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    docs = []
    for page in pages[:limit]:
        text = (page.get("text") or "").strip()
        if not text or len(text) < 100:
            continue

        metadata = {
            "source": page["url"],
            "title": page.get("title", "Sem título"),
            "tipo": page.get("tipo", "portal_geral"),
            "depth": page.get("depth", 0),
        }
        docs.append(Document(page_content=text, metadata=metadata))

    return docs


def compare_chunking(documents: list[Document]) -> None:
    """Compara os dois métodos de chunking."""
    print("\n" + "=" * 80)
    print("  COMPARAÇÃO: Chunking Semântico vs. Recursivo")
    print("=" * 80)

    for doc_idx, doc in enumerate(documents, 1):
        print(f"\n{'─' * 80}")
        print(f"Documento {doc_idx}: {doc.metadata.get('tipo', 'geral').upper()}")
        print(f"  Título: {doc.metadata['title'][:70]}")
        print(f"  Tamanho: {len(doc.page_content)} chars")
        print(f"{'─' * 80}")

        # Chunking recursivo
        recursive_chunks = chunk_documents([doc])
        print(f"\n  CHUNKING RECURSIVO → {len(recursive_chunks)} chunks:")
        for i, chunk in enumerate(recursive_chunks[:3], 1):
            preview = chunk.page_content[:70].replace("\n", " ")
            print(f"    [{i}] {len(chunk.page_content)} chars | {preview}...")

        # Chunking semântico
        semantic_chunks = chunk_documents_semantic([doc])
        print(f"\n  CHUNKING SEMÂNTICO → {len(semantic_chunks)} chunks:")
        for i, chunk in enumerate(semantic_chunks[:3], 1):
            has_structure = chunk.metadata.get("has_semantic_structure", False)
            headers = chunk.metadata.get("section_headers", [])
            preview = chunk.page_content[:70].replace("\n", " ")
            print(f"    [{i}] {len(chunk.page_content)} chars | has_structure={has_structure}")
            if headers:
                print(f"         headers={headers}")
            print(f"         {preview}...")


def print_chunk_details(chunks: list[Document], method_name: str) -> None:
    """Imprime detalhes estruturados dos chunks."""
    print(f"\n{method_name}:")
    total_size = sum(len(c.page_content) for c in chunks)
    print(f"  Total chunks: {len(chunks)}")
    print(f"  Total chars: {total_size}")
    print(f"  Avg. size: {total_size // len(chunks) if chunks else 0}")

    for i, chunk in enumerate(chunks[:5], 1):
        lines = chunk.page_content.split("\n")
        print(f"  [{i}] {len(chunk.page_content):4d} chars, {len(lines):2d} lines")


def main() -> None:
    json_path = Path(__file__).parent / "crawler" / "dados_ufpel.json"

    if not json_path.exists():
        print(f"❌ Arquivo não encontrado: {json_path}")
        sys.exit(1)

    print(f"Carregando amostra de dados: {json_path}")
    documents = load_sample_documents(str(json_path), limit=10)

    if not documents:
        print("❌ Nenhum documento válido carregado.")
        sys.exit(1)

    print(f"✓ {len(documents)} documentos carregados")

    compare_chunking(documents)

    # Comparação global
    print("\n" + "=" * 80)
    print("  RESUMO ESTATÍSTICO")
    print("=" * 80)

    all_recursive = chunk_documents(documents)
    all_semantic = chunk_documents_semantic(documents)

    print_chunk_details(all_recursive, "RECURSIVO")
    print_chunk_details(all_semantic, "SEMÂNTICO")

    # Análise de qualidade semântica
    print(f"\nQualidade Semântica:")
    semantic_with_structure = sum(
        1 for c in all_semantic
        if c.metadata.get("has_semantic_structure", False)
    )
    print(f"  Chunks com estrutura reconhecida: {semantic_with_structure}/{len(all_semantic)}")
    print(f"  Taxa de estrutura: {100 * semantic_with_structure / len(all_semantic):.1f}%")

    print("\n✓ Teste concluído com sucesso!")


if __name__ == "__main__":
    main()
