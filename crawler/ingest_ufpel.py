"""
Ingestão dos Dados Crawleados no pgvector
==========================================
Lê o JSON gerado por crawl_ufpel.py, divide os textos em chunks
e insere os embeddings no banco PostgreSQL + pgvector.

Uso:
    # Inserção padrão (acumula na coleção existente)
    python ingest_ufpel.py

    # Especificar arquivo JSON de entrada
    python ingest_ufpel.py --input dados_ufpel.json

    # Apagar a coleção e recriar do zero antes de inserir
    python ingest_ufpel.py --reset

    # Combinado
    python ingest_ufpel.py --input ufpel_completo.json --reset

Pré-requisitos:
    1. PostgreSQL rodando com a extensão pgvector habilitada
    2. Variáveis de ambiente configuradas em aplicacao/.env
    3. Arquivo JSON gerado por crawl_ufpel.py
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# Permite importar config, providers e chunking da pasta aplicacao/
_APP_DIR = Path(__file__).resolve().parent.parent / "aplicacao"
sys.path.insert(0, str(_APP_DIR))

import config                                    # noqa: E402 — precisa antes dos imports LangChain
from providers import get_embeddings             # noqa: E402
from chunking import chunk_documents             # noqa: E402
from chunking_semantico import chunk_documents_semantic  # noqa: E402
from langchain_community.vectorstores import PGVector   # noqa: E402
from langchain_core.documents import Document   # noqa: E402

DEFAULT_INPUT     = "dados_ufpel.json"
# Free tier do Google: 100 embed_content calls/min.
# batch_size interno do langchain_google_genai = 100 textos/call.
# Com INGEST_BATCH_SIZE=100 → 1 call por lote → ~85 lotes/min com INGEST_DELAY=0.7s.
INGEST_BATCH_SIZE = 100
INGEST_DELAY      = 0.7   # segundos entre lotes (~85 chamadas/min — abaixo do limite de 100)
INGEST_MAX_RETRY  = 3     # tentativas por lote em erros transitórios (502, 503)

# Mapeamento tipo (campo 'tipo' no JSON) → tabela física PostgreSQL
# Mantido em sincronia com config.COLLECTION_MAP e setup_ambiente.sh
TIPO_TO_TABLE: dict[str, str] = {
    "disciplina":   "disciplinas",
    "projeto":      "projetos",
    "servidor":     "servidores",
    "unidade":      "unidades",
    "curso":        "cursos",
    "portal_geral": "portal_geral",
}


# =============================================================================
# Carregamento e conversão dos dados
# =============================================================================

def load_crawled_data(input_path: str) -> list[dict]:
    with open(input_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    print(f"[Ingestão] {len(data)} página(s) carregada(s) de '{input_path}'")
    return data


def _deduplicate_servidores(pages: list[dict]) -> list[dict]:
    """
    Agrupa registros de servidor pelo nome e mantém apenas o mais informativo
    (o de texto mais longo, que geralmente contém o currículo/Lattes).

    O portal UFPel cria IDs separados por vínculo (ativo, substituto, etc.),
    fazendo a mesma pessoa aparecer como múltiplas URLs. Sem isso, o RAG
    mostraria várias fontes para o mesmo servidor.
    """
    import re
    from collections import defaultdict

    servidor_pages = [p for p in pages if p.get("tipo") == "servidor"]
    other_pages    = [p for p in pages if p.get("tipo") != "servidor"]

    by_name: dict[str, list[dict]] = defaultdict(list)
    for page in servidor_pages:
        m = re.search(r"Nome do Servidor:\s*(.+)", page.get("text", ""))
        name = m.group(1).strip() if m else page["url"]
        by_name[name].append(page)

    deduped: list[dict] = []
    removed = 0
    for group in by_name.values():
        # Mantém o registro com texto mais longo (mais completo/com Lattes)
        primary = max(group, key=lambda p: len(p.get("text", "")))
        deduped.append(primary)
        removed += len(group) - 1

    if removed:
        print(f"[Dedup] {removed} registro(s) duplicado(s) de servidor removido(s) "
              f"({len(deduped)} servidores únicos mantidos).")

    return other_pages + deduped


def pages_to_documents(pages: list[dict]) -> list[Document]:
    """
    Converte os dicionários do JSON em LangChain Documents.

    Metadados preservados por chunk:
      source      → URL da página (rastreabilidade)
      title       → título HTML da página
      depth       → profundidade no grafo de links
      crawled_at  → timestamp ISO 8601 da coleta
      categoria   → rótulo fixo para facilitar filtragem
      tipo        → 'projeto', 'disciplina' ou ausente (portal geral)
    """
    pages = _deduplicate_servidores(pages)
    docs = []
    skipped = 0
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            skipped += 1
            continue
        raw_title = page.get("title", "Sem título")
        titulo = raw_title.split(" | ")[0].strip()
        metadata = {
            "source":     page["url"],
            "title":      raw_title,
            "titulo":     titulo,
            "depth":      page.get("depth", 0),
            "crawled_at": page.get("crawled_at", ""),
            "categoria":  "portal_institucional",
        }
        if page.get("tipo"):
            metadata["tipo"] = page["tipo"]
        docs.append(Document(page_content=text, metadata=metadata))

    print(f"[Ingestão] {len(docs)} documento(s) válidos "
          f"({skipped} ignorados por texto vazio)")
    return docs


# =============================================================================
# Criação das tabelas físicas por tipo
# =============================================================================

def _get_table_vector_dim(cur, table: str) -> int | None:
    """
    Retorna a dimensão atual da coluna `embedding` de `table`,
    ou None se a tabela não existir.
    Usa pg_attribute + format_type para extrair o número do tipo vector(N).
    """
    import re
    cur.execute("""
        SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
        FROM pg_catalog.pg_attribute a
        JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
        WHERE c.relname = %s AND a.attname = 'embedding' AND a.attnum > 0
    """, (table,))
    row = cur.fetchone()
    if row is None:
        return None
    match = re.search(r"vector\((\d+)\)", row[0])
    return int(match.group(1)) if match else None


def ensure_tables_exist() -> None:
    """
    Garante que as tabelas por tipo existam com a dimensão correta.

    - Se a tabela não existe: cria com VECTOR({EMBEDDING_DIMS}).
    - Se existe com dimensão errada: dropa e recria (sem dados = seguro).
    - Se existe com dimensão correta: não faz nada (idempotente).

    Tabelas gerenciadas: disciplinas, projetos, servidores, unidades, cursos, portal_geral
    Cada tabela:  id BIGSERIAL PK | conteudo TEXT | embedding VECTOR({EMBEDDING_DIMS})
    """
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        for table in TIPO_TO_TABLE.values():
            existing_dim = _get_table_vector_dim(cur, table)

            if existing_dim is not None and existing_dim != config.EMBEDDING_DIMS:
                print(f"[Tabelas] '{table}': dimensão {existing_dim} → {config.EMBEDDING_DIMS}. Recriando...")
                cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                existing_dim = None

            if existing_dim is None:
                cur.execute(f"""
                    CREATE TABLE {table} (
                        id        BIGSERIAL PRIMARY KEY,
                        conteudo  TEXT       NOT NULL,
                        embedding VECTOR({config.EMBEDDING_DIMS})
                    )
                """)

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
                    ON {table} USING hnsw (embedding vector_cosine_ops)
            """)

        print(f"[Tabelas] {len(TIPO_TO_TABLE)} tabelas por tipo verificadas/criadas "
              f"(dims={config.EMBEDDING_DIMS}).")
    finally:
        cur.close()
        conn.close()


# =============================================================================
# Inserção no pgvector
# =============================================================================

def _add_batch_with_retry(store: PGVector, batch: list[Document]) -> None:
    """Insere um lote com até INGEST_MAX_RETRY tentativas e backoff exponencial."""
    for attempt in range(INGEST_MAX_RETRY):
        try:
            store.add_documents(batch)
            return
        except Exception as exc:
            if attempt < INGEST_MAX_RETRY - 1:
                wait = (2 ** attempt) + random.random()
                print(f"  [Retry {attempt + 1}/{INGEST_MAX_RETRY}] "
                      f"Aguardando {wait:.1f}s após erro: {exc}")
                time.sleep(wait)
            else:
                raise


def _insert_raw_batch(
    table: str,
    texts: list[str],
    vectors: list[list[float]],
) -> None:
    """
    Insere um lote de (texto, vetor) diretamente na tabela física `table`.
    Usa execute_values para eficiência e o cast `::vector` para compatibilidade
    com pgvector sem depender do adaptador psycopg2 do pgvector.
    """
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {table} (conteudo, embedding) VALUES %s",
                    [
                        (text, f"[{','.join(map(str, vec))}]")
                        for text, vec in zip(texts, vectors)
                    ],
                    template="(%s, %s::vector)",
                )
    finally:
        conn.close()


def _embed_with_retry(
    embeddings_model,
    texts: list[str],
    max_retries: int = 10,
) -> list[list[float]]:
    """
    Gera embeddings com retry automático para erros 429 (rate limit).

    Extrai o `retryDelay` sugerido pelo servidor quando disponível;
    caso contrário usa backoff exponencial com jitter.
    """
    import re
    for attempt in range(max_retries):
        try:
            return embeddings_model.embed_documents(texts)
        except Exception as exc:
            msg = str(exc)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                delay_match = re.search(r"retryDelay.*?(\d+(?:\.\d+)?)", msg)
                base = float(delay_match.group(1)) if delay_match else min(60.0, 10.0 * (2 ** attempt))
                wait = base + random.random() * 2
                print(f"  [Rate limit] Aguardando {wait:.1f}s "
                      f"(tentativa {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Falha após {max_retries} tentativas de embedding (rate limit persistente)."
    )


def ingest_to_pgvector(
    documents: list[Document],
    reset: bool = False,
    batch_size: int = INGEST_BATCH_SIZE,
    collection_name: str | None = None,
) -> PGVector:
    """
    Divide os documentos em chunks e insere no pgvector em lotes.

    Fluxo:
        Document (texto completo)
            → RecursiveCharacterTextSplitter (chunk_size/overlap de config.py)
            → embed_documents() em lotes de `batch_size`  ← evita 502 na NIM
            → INSERT INTO langchain_pg_embedding

    Args:
        documents       : lista de LangChain Documents já construídos
        reset           : se True, DROP + CREATE na coleção antes de inserir
        batch_size      : chunks por lote de embedding (padrão INGEST_BATCH_SIZE)
        collection_name : coleção alvo (padrão: config.COLLECTION_NAME)
    """
    col = collection_name or config.COLLECTION_NAME
    embeddings = get_embeddings()
    chunks = chunk_documents(documents)
    total = len(chunks)

    print()
    print("[Ingestão] ─── Parâmetros ───────────────────────────────")
    print(f"           Coleção    : {col}")
    print(f"           Banco      : {config.DB_CONFIG['dbname']} "
          f"@ {config.DB_CONFIG['host']}:{config.DB_CONFIG['port']}")
    print(f"           Chunks     : {total}")
    print(f"           Chunk size : {config.CHUNK_SIZE} chars "
          f"(overlap={config.CHUNK_OVERLAP})")
    print(f"           Lote       : {batch_size} chunks  "
          f"(~{batch_size // 50} chamadas NIM/lote, delay={INGEST_DELAY}s)")
    print(f"           Reset      : {reset}")
    print("[Ingestão] ─────────────────────────────────────────────")

    # Primeiro lote: cria (ou recria) a coleção
    first = chunks[:batch_size]
    print(f"[Ingestão] Lote 1 — {len(first)} chunks ...")
    store = PGVector.from_documents(
        documents=first,
        embedding=embeddings,
        collection_name=col,
        connection_string=config.CONNECTION_STRING,
        pre_delete_collection=reset,
        use_jsonb=True,
    )
    ingested = len(first)

    # Lotes restantes com pausa entre eles
    n_batches = (total - batch_size + batch_size - 1) // batch_size
    for idx, start in enumerate(range(batch_size, total, batch_size), start=2):
        batch = chunks[start:start + batch_size]
        print(f"[Ingestão] Lote {idx}/{idx + n_batches - 1} — "
              f"{ingested}/{total} chunks inseridos ...")
        time.sleep(INGEST_DELAY)
        _add_batch_with_retry(store, batch)
        ingested += len(batch)

    print(f"[Ingestão] Concluída! {ingested} chunks inseridos em '{col}'.")
    return store


def ingest_segmented(
    documents: list[Document],
    reset: bool = False,
    batch_size: int = INGEST_BATCH_SIZE,
    delay: float = INGEST_DELAY,
    max_per_tipo: int | None = None,
    use_semantic_chunking: bool = True,
) -> dict[str, int]:
    """
    Pipeline segmentada por tipo:

    1. Cria as tabelas físicas por tipo se não existirem (idempotente).
    2. Agrupa documentos pelo campo `tipo` dos metadados.
    3. Para cada grupo (limitado a `max_per_tipo` docs), processa em lotes:
         a. Faz chunking (semântico ou recursivo)
         b. Gera embeddings UMA vez (Google gemini-embedding-001).
         c. Insere na tabela física da categoria (SQL direto).
         d. Insere na coleção PGVector do LangChain (para o pipeline RAG).

    Mapeamento tipo → tabela / coleção:
      disciplina  → disciplinas  / ufpel_disciplinas
      projeto     → projetos     / ufpel_projetos
      servidor    → servidores   / ufpel_servidores
      unidade     → unidades     / ufpel_unidades
      curso       → cursos       / ufpel_cursos
      (outros)    → portal_geral / ufpel_portal_geral

    Chunking Semântico:
      Se use_semantic_chunking=True, respeita estrutura de seções (Resumo:, Objetivos:, etc.)
      mantendo conteúdo semanticamente coerente e adicionando metadados de seção.
      Fallback para RecursiveCharacterTextSplitter se não houver estrutura clara.

    Limites free tier Google (gemini-embedding-001):
      ~1 chamada/min efetiva → use --max-por-tipo 200 --delay 62
      Com 200 docs/tipo: ~2 000 chunks → ~20 lotes → ~20 min por tipo.

    Args:
        documents           : LangChain Documents com metadado `tipo` preenchido
        reset               : se True, trunca tabelas e recria coleções antes de inserir
        batch_size          : chunks por lote de embedding (padrão INGEST_BATCH_SIZE)
        delay               : segundos entre lotes (padrão INGEST_DELAY)
        max_per_tipo        : limite de documentos por tipo (None = sem limite)
        use_semantic_chunking : se True, usa chunking semântico; senão, usa recursivo

    Returns:
        Dict {collection_name: número_de_documentos_inseridos}
    """
    from collections import defaultdict

    # 1. Garante existência das tabelas físicas
    ensure_tables_exist()

    # 2. Agrupa por tipo
    by_tipo: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        tipo = doc.metadata.get("tipo", "portal_geral")
        by_tipo[tipo].append(doc)

    embeddings_model = get_embeddings()
    summary: dict[str, int] = {}

    for tipo, docs in sorted(by_tipo.items()):
        col   = config.collection_for_tipo(tipo)
        table = TIPO_TO_TABLE.get(tipo, "portal_geral")

        # Aplica limite por tipo (útil para free tier)
        if max_per_tipo is not None and len(docs) > max_per_tipo:
            print(f"[Limite] '{tipo}': {len(docs)} docs → truncando para {max_per_tipo}")
            docs = docs[:max_per_tipo]

        print()
        print("=" * 62)
        print(f"  Tipo      : {tipo}")
        print(f"  Tabela    : {table}  |  Coleção: {col}")
        print(f"  Documentos: {len(docs)}"
              + (f" (de {len(by_tipo[tipo])} total)" if max_per_tipo else ""))
        print(f"  Chunking  : {'semântico' if use_semantic_chunking else 'recursivo'}")
        print("=" * 62)

        chunks = (
            chunk_documents_semantic(docs) if use_semantic_chunking
            else chunk_documents(docs)
        )
        total     = len(chunks)
        n_batches = max(1, (total + batch_size - 1) // batch_size)
        eta_min   = round(n_batches * delay / 60, 1)
        print(f"  Chunks    : {total}  |  Lotes: {n_batches}  |  ETA ≈ {eta_min} min")

        # Trunca tabela física antes do primeiro lote quando reset=True
        if reset:
            conn = psycopg2.connect(**config.DB_CONFIG)
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
            finally:
                conn.close()
            print(f"[Reset] Tabela '{table}' truncada.")

        store: PGVector | None = None
        ingested = 0

        for batch_idx, start in enumerate(range(0, total, batch_size)):
            batch     = chunks[start : start + batch_size]
            batch_num = batch_idx + 1
            print(f"[Ingestão] Lote {batch_num}/{n_batches} — {len(batch)} chunks ...")

            texts     = [c.page_content for c in batch]
            metadatas = [c.metadata for c in batch]

            # Gera embeddings UMA vez para uso em ambas as inserções (com retry em 429)
            vectors = _embed_with_retry(embeddings_model, texts)

            # a) Tabela física por tipo (SQL direto — para demos e SQL puro)
            _insert_raw_batch(table, texts, vectors)

            # b) Coleção PGVector do LangChain (para o pipeline RAG / router)
            text_embeddings = list(zip(texts, vectors))
            if store is None:
                store = PGVector.from_embeddings(
                    text_embeddings=text_embeddings,
                    embedding=embeddings_model,
                    metadatas=metadatas,
                    collection_name=col,
                    connection_string=config.CONNECTION_STRING,
                    pre_delete_collection=reset,
                    use_jsonb=True,
                )
            else:
                store.add_embeddings(
                    texts=texts,
                    embeddings=vectors,
                    metadatas=metadatas,
                )

            ingested += len(batch)
            if batch_num < n_batches:
                time.sleep(delay)

        print(f"[OK] '{tipo}': {ingested} chunks → '{table}' + coleção '{col}'.")
        summary[col] = len(docs)

    print()
    print("[Ingestão Segmentada] Resumo:")
    for col, qtd in sorted(summary.items()):
        tipo_label = next(
            (t for t, c in config.COLLECTION_MAP.items() if c == col), col
        )
        print(f"  {tipo_label:<15} → {col:<30}: {qtd} docs")
    return summary


# =============================================================================
# Entrypoint CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Insere dados crawleados da UFPel no pgvector.\n\n"
            "RATE LIMITS — free tier Google gemini-embedding-001:\n"
            "  ~1 req/min efetivo. Recomendado para free tier:\n"
            "    python ingest_ufpel.py --max-por-tipo 200 --delay 62\n"
            "  (≈ 20 min por tipo, viável para demonstração)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", default=DEFAULT_INPUT,
                   metavar="FILE", help="Arquivo JSON com dados crawleados")
    p.add_argument("--reset", action="store_true",
                   help="Apaga cada coleção/tabela no banco antes de inserir")
    p.add_argument("--max-por-tipo", type=int, default=None, metavar="N",
                   help=(
                       "Limita a N documentos por tipo (free tier: use 200). "
                       "Sem este flag, ingesta o dataset completo."
                   ))
    p.add_argument("--delay", type=float, default=INGEST_DELAY, metavar="SECS",
                   help=f"Segundos entre lotes de embedding (padrão: {INGEST_DELAY}; "
                        "free tier: use 62)")
    p.add_argument("--all-in-one", action="store_true",
                   help="Insere tudo em uma única coleção (sem segmentação por tipo)")
    p.add_argument("--chunking", choices=["semantico", "recursivo"], default="semantico",
                   help="Estratégia de chunking (padrão: semantico)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    pages = load_crawled_data(args.input)
    documents = pages_to_documents(pages)

    if not documents:
        print("[Aviso] Nenhum documento válido. Verifique o arquivo JSON.")
        sys.exit(1)

    use_semantic = args.chunking == "semantico"

    if args.all_in_one:
        ingest_to_pgvector(documents, reset=args.reset)
    else:
        ingest_segmented(
            documents,
            reset=args.reset,
            delay=args.delay,
            max_per_tipo=args.max_por_tipo,
            use_semantic_chunking=use_semantic,
        )


if __name__ == "__main__":
    main()
