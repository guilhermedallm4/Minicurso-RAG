"""
Segmentação de Documentos (Chunking)
=============================================================================
Conceito — SEÇÃO 3 do curso

LLMs têm janela de contexto limitada (~4k–128k tokens).
Documentos institucionais longos (regulamentos, editais) precisam ser
divididos em partes menores que caibam no prompt.

RecursiveCharacterTextSplitter divide recursivamente por separadores:
  "\n\n" (parágrafo) → "\n" (linha) → ". " (frase) → " " (palavra)
garantindo que o texto não seja cortado no meio de uma ideia.

Parâmetros:
  chunk_size    : máximo de caracteres por chunk
  chunk_overlap : sobreposição entre chunks adjacentes
                  (preserva contexto nas fronteiras entre chunks)
"""
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import config


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Divide Documents do LangChain em chunks respeitando estrutura semântica."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[Chunking] {len(documents)} doc(s) → {len(chunks)} chunks "
          f"(size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")
    return chunks


def chunk_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
) -> List[Document]:
    """Converte strings brutas em Documents e aplica chunking."""
    metadatas = metadatas or [{}] * len(texts)
    docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    return chunk_documents(docs)


# =============================================================================
# Execução direta: demonstra o efeito do chunking com parâmetros variados
# python chunking.py
# =============================================================================
if __name__ == "__main__":
    TEXTO_EXEMPLO = """
O Programa de Pós-Graduação em Computação (PPGC) da Universidade Federal
oferece cursos de mestrado e doutorado nas áreas de Inteligência Artificial,
Sistemas Distribuídos e Segurança da Informação.

O prazo para entrega da dissertação de mestrado é de 24 meses após o ingresso.
Prorrogações de até 6 meses devem ser solicitadas formalmente ao colegiado do
programa, mediante justificativa e aprovação do orientador.

A matrícula para o semestre 2024.2 ocorre entre os dias 01 e 10 de julho.
Alunos com débitos na biblioteca não poderão realizar a matrícula no período
regular, devendo regularizar a situação previamente junto ao setor competente.
""".strip()

    print("=" * 60)
    print("  SEÇÃO 3 — Chunking de Documentos")
    print("=" * 60)

    print("\n--- Texto original ---")
    print(f"  {len(TEXTO_EXEMPLO)} caracteres\n")

    print("--- Comparação de parâmetros de chunking ---\n")
    for size, overlap in [(200, 20), (500, 50), (800, 80)]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(TEXTO_EXEMPLO)
        print(f"  chunk_size={size}, overlap={overlap} → {len(chunks)} chunks")
        for i, c in enumerate(chunks, 1):
            print(f"    [{i}] {len(c)} chars | {c[:60].replace(chr(10), ' ')}...")

    print()
    print("--- Com metadados (uso real) ---")
    docs = chunk_texts(
        [TEXTO_EXEMPLO],
        [{"source": "regulamento_ppgc.pdf", "categoria": "pos_graduacao"}],
    )
    for i, d in enumerate(docs, 1):
        print(f"  Chunk {i}: {d.metadata} | {d.page_content[:60]}...")
