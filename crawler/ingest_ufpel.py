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
INGEST_BATCH_SIZE = 500
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
    "gestao":       "gestao",
    "sobre":        "sobre",
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
    Agrupa registros de servidor pelo nome e mantém apenas o mais informativo.

    O portal UFPel cria IDs separados por vínculo (ativo, substituto, etc.),
    fazendo a mesma pessoa aparecer como múltiplas URLs. Sem isso, o RAG
    mostraria várias fontes para o mesmo servidor.

    Suporta o novo schema (campo 'titulo') e o legado (campo 'text').
    """
    import re
    from collections import defaultdict

    servidor_pages = [p for p in pages if p.get("tipo") == "servidor"]
    other_pages    = [p for p in pages if p.get("tipo") != "servidor"]

    by_name: dict[str, list[dict]] = defaultdict(list)
    for page in servidor_pages:
        # Novo schema: nome em 'titulo'
        if "titulo" in page:
            name = page["titulo"] or page.get("metadata", {}).get("url", "")
        else:
            # Legado: extrai "Nome do Servidor:" do campo text
            m = re.search(r"Nome do Servidor:\s*(.+)", page.get("text", ""))
            name = m.group(1).strip() if m else page.get("url", "")
        by_name[name].append(page)

    deduped: list[dict] = []
    removed = 0
    for group in by_name.values():
        # Mantém o registro com conteúdo mais longo (mais completo/com Lattes)
        def _content_len(p: dict) -> int:
            return len(p.get("embedding_text") or p.get("text") or "")
        primary = max(group, key=_content_len)
        deduped.append(primary)
        removed += len(group) - 1

    if removed:
        print(f"[Dedup] {removed} registro(s) duplicado(s) de servidor removido(s) "
              f"({len(deduped)} servidores únicos mantidos).")

    return other_pages + deduped


def _enrich_curso_professores(pages: list[dict]) -> list[dict]:
    """
    Enriquece os professores de cada curso com a URL do Lattes e a lotação
    correta, cruzando a URL do servidor (/servidores/id/XXX) com os dados
    dos próprios documentos de servidor já presentes na lista de páginas.

    Isso permite que o RAG retorne "Professor X — Lattes: https://..." ao
    responder "Quais professores do curso de Medicina?".
    """
    EMPTY = "Não há informações disponíveis"

    # Índice rápido: URL do servidor → dados_completos
    url_to_srv: dict[str, dict] = {}
    for p in pages:
        if p.get("tipo") == "servidor":
            srv_url = p.get("metadata", {}).get("url", "")
            if srv_url:
                url_to_srv[srv_url] = p.get("dados_completos", {})

    enriched = 0
    for page in pages:
        if page.get("tipo") != "curso":
            continue
        dados = page.get("dados_completos", {})
        profs = dados.get("professores", [])
        changed = False
        for prof in profs:
            srv_url = prof.get("url", "")
            if srv_url and srv_url in url_to_srv:
                srv = url_to_srv[srv_url]
                if not prof.get("lattes") and srv.get("lattes_url") and srv["lattes_url"] != EMPTY:
                    prof["lattes"] = srv["lattes_url"]
                    changed = True
                # Atualiza unidade com dados reais do servidor
                if (not prof.get("unidade") or prof["unidade"] == EMPTY) and srv.get("lotacao"):
                    prof["unidade"] = srv["lotacao"]
                    changed = True

        if changed:
            enriched += 1
            # Reconstrói embedding_text para incluir Lattes
            profs_texto = "\n".join(
                f"  - {p['nome']}"
                + (f" ({p.get('unidade', '')})" if p.get("unidade") and p["unidade"] != EMPTY else "")
                + (f" | Lattes: {p['lattes']}" if p.get("lattes") else "")
                for p in profs
                if p.get("nome") and p["nome"] != EMPTY
            )
            # Substitui apenas a seção de professores no embedding_text existente
            import re as _re
            old_text = page.get("embedding_text", "")
            new_section = f"Professores e docentes do curso:\n{profs_texto}"
            updated = _re.sub(
                r"Professores e docentes do curso:.*?(?=\n\n[A-Z]|\Z)",
                new_section,
                old_text,
                flags=_re.DOTALL,
            )
            page["embedding_text"] = updated if updated != old_text else old_text

    if enriched:
        print(f"[Enrich] {enriched} curso(s) com professores enriquecidos com Lattes.")
    return pages


def pages_to_documents(pages: list[dict]) -> list[Document]:
    """
    Converte os dicionários do JSON em LangChain Documents.

    Suporta dois schemas:
      - Novo (crawl_ufpel.py async): campos id, tipo, titulo, embedding_text,
        metadata.url, metadata.crawled_at, dados_completos
      - Legado (crawl BFS antigo): campos url, title, text, tipo, depth, crawled_at

    Metadados preservados por chunk:
      source      → URL da página
      title       → título do registro
      doc_id      → UUID do documento (novo schema) para busca por dados_completos
      crawled_at  → timestamp ISO 8601
      categoria   → rótulo fixo para filtragem
      tipo        → seção (curso, disciplina, projeto, etc.)
    """
    pages = _deduplicate_servidores(pages)
    pages = _enrich_curso_professores(pages)
    docs    = []
    skipped = 0

    for page in pages:
        # ── Detecta schema ────────────────────────────────────────────────────
        is_new_schema = "embedding_text" in page

        if is_new_schema:
            text       = (page.get("embedding_text") or "").strip()
            raw_title  = page.get("titulo", "Sem título")
            url        = page.get("metadata", {}).get("url", "")
            crawled_at = page.get("metadata", {}).get("crawled_at", "")
            doc_id     = page.get("id", "")
            tipo       = page.get("tipo", "portal_geral")
            # metadados extras vindos do novo schema (filtráveis por coleção)
            extra_meta = {k: v for k, v in page.get("metadata", {}).items()
                          if k not in ("url", "crawled_at")}
        else:
            text       = (page.get("text") or "").strip()
            raw_title  = page.get("title", "Sem título")
            url        = page.get("url", "")
            crawled_at = page.get("crawled_at", "")
            doc_id     = ""
            tipo       = page.get("tipo", "portal_geral")
            extra_meta = {}

        if not text:
            skipped += 1
            continue

        titulo = raw_title.split(" | ")[0].strip()

        metadata: dict = {
            "source":     url,
            "title":      raw_title,
            "titulo":     titulo,
            "crawled_at": crawled_at,
            "categoria":  "portal_institucional",
            "tipo":       tipo,
            **extra_meta,
        }
        if doc_id:
            metadata["doc_id"] = doc_id

        docs.append(Document(page_content=text, metadata=metadata))

    print(f"[Ingestão] {len(docs)} documento(s) válidos "
          f"({skipped} ignorados por texto vazio)")
    return docs


# =============================================================================
# Tabela de documentos completos (dados_completos para contexto LLM)
# =============================================================================

def ensure_dados_completos_table() -> None:
    """Cria a tabela doc_completos se não existir — armazena dados_completos por doc_id."""
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS doc_completos (
                doc_id  TEXT PRIMARY KEY,
                tipo    TEXT,
                titulo  TEXT,
                dados   JSONB
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS doc_completos_tipo_idx ON doc_completos (tipo)")
        # Índice GIN para busca rápida por título (trigram)
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS doc_completos_titulo_trgm
                ON doc_completos USING gin (titulo gin_trgm_ops)
        """)
    finally:
        cur.close()
        conn.close()


def store_dados_completos(pages: list[dict]) -> int:
    """
    Insere/atualiza dados_completos de cada página na tabela doc_completos.
    Retorna o número de registros inseridos/atualizados.
    """
    import json as _json

    rows = []
    for p in pages:
        if not p.get("id") or not p.get("dados_completos"):
            continue
        dc = dict(p["dados_completos"])
        # Inclui URL de origem para permitir links nas respostas do chatbot
        meta_url = (p.get("metadata") or {}).get("url", "")
        if meta_url and not dc.get("url"):
            dc["url"] = meta_url
        rows.append((
            p.get("id", ""),
            p.get("tipo", "portal_geral"),
            p.get("titulo", ""),
            _json.dumps(dc, ensure_ascii=False),
        ))
    if not rows:
        return 0

    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO doc_completos (doc_id, tipo, titulo, dados)
                    VALUES %s
                    ON CONFLICT (doc_id) DO UPDATE
                        SET tipo   = EXCLUDED.tipo,
                            titulo = EXCLUDED.titulo,
                            dados  = EXCLUDED.dados
                    """,
                    rows,
                )
    finally:
        conn.close()
    return len(rows)


def fetch_dados_completos(doc_ids: list[str]) -> dict[str, dict]:
    """
    Retorna {doc_id: dados_completos} para os IDs solicitados.
    Usado pelo pipeline RAG para enriquecer o contexto do LLM.
    """
    if not doc_ids:
        return {}
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT doc_id, dados FROM doc_completos WHERE doc_id = ANY(%s)",
                (doc_ids,),
            )
            return {row[0]: row[1] for row in cur.fetchall()}
    finally:
        conn.close()


def lookup_by_title(
    query: str,
    tipo: str | None = None,
    top_k: int = 3,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """
    Busca direta por similaridade de título em doc_completos (pg_trgm).

    Usado como shortcut quando a query menciona um nome específico — evita
    embedding + busca vetorial para perguntas como 'O que o professor X faz?'
    ou 'Qual a ementa da disciplina Y?'.

    Retorna lista de {'doc_id', 'tipo', 'titulo', 'dados', 'similarity'}.
    """
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn.cursor() as cur:
            # Usa o operador % (trigrama) em vez de similarity() > threshold,
            # permitindo que o planner use o índice GIN (169ms → ~25ms).
            cur.execute(f"SET pg_trgm.similarity_threshold = {similarity_threshold}")
            if tipo:
                cur.execute(
                    """
                    SELECT doc_id, tipo, titulo, dados,
                           similarity(titulo, %s) AS sim
                    FROM doc_completos
                    WHERE tipo = %s
                      AND titulo %% %s
                    ORDER BY sim DESC
                    LIMIT %s
                    """,
                    (query, tipo, query, top_k),
                )
            else:
                cur.execute(
                    """
                    SELECT doc_id, tipo, titulo, dados,
                           similarity(titulo, %s) AS sim
                    FROM doc_completos
                    WHERE titulo %% %s
                    ORDER BY sim DESC
                    LIMIT %s
                    """,
                    (query, query, top_k),
                )
            rows = cur.fetchall()
            return [
                {"doc_id": r[0], "tipo": r[1], "titulo": r[2],
                 "dados": r[3], "similarity": float(r[4])}
                for r in rows
            ]
    finally:
        conn.close()


# =============================================================================
# Tabelas estruturadas por tipo (*_info) — colunas reais de dados_completos
# =============================================================================
# Permitem responder com UM SQL a perguntas como:
#   SELECT * FROM disciplinas_info WHERE nome % 'Aprendizado de Máquina';
# e juntar com os chunks vetoriais via doc_id.

INFO_TABLES: dict[str, dict] = {
    "disciplina": {
        "table": "disciplinas_info",
        "name_col": "nome",
        "columns": {
            "codigo": "TEXT", "nome": "TEXT", "tipo_atividade": "TEXT",
            "periodicidade": "TEXT", "creditos": "TEXT", "carga_horaria": "TEXT",
            "ch_teorica": "TEXT", "ch_pratica": "TEXT", "ch_obrigatoria": "TEXT",
            "freq_aprovacao": "TEXT", "unidade_responsavel": "TEXT",
            "ementa": "TEXT", "objetivos": "TEXT", "conteudo_programatico": "TEXT",
            "bibliografia": "TEXT", "turmas_ofertadas": "JSONB",
            "cursos_relacionados": "JSONB",
        },
    },
    "projeto": {
        "table": "projetos_info",
        "name_col": "titulo",
        "columns": {
            "titulo": "TEXT", "resumo": "TEXT", "enfase": "TEXT",
            "data_inicio": "TEXT", "data_fim": "TEXT", "situacao": "TEXT",
            "coordenador": "TEXT", "unidade_origem": "TEXT", "area_cnpq": "TEXT",
            "eixo_tematico": "TEXT", "linha_extensao": "TEXT",
            "informacoes": "JSONB", "equipe_vigente": "JSONB",
            "financeiro": "JSONB", "professores_relacionados": "JSONB",
        },
    },
    "servidor": {
        "table": "servidores_info",
        "name_col": "nome",
        "columns": {
            "nome": "TEXT", "matricula_siape": "TEXT", "categoria": "TEXT",
            "cargo": "TEXT", "classe_nivel": "TEXT", "titulacao": "TEXT",
            "lotacao": "TEXT", "regime_jornada": "TEXT", "situacao": "TEXT",
            "data_ingresso_servico": "TEXT", "data_ingresso_ufpel": "TEXT",
            "data_ingresso_cargo": "TEXT", "email": "TEXT",
            "curriculo_resumo": "TEXT", "formacao_academica": "JSONB",
            "areas_atuacao": "JSONB", "lattes_url": "TEXT",
            "projetos_ativos": "JSONB", "disciplinas_ministradas": "JSONB",
            "cursos_relacionados": "JSONB",
        },
    },
    "curso": {
        "table": "cursos_info",
        "name_col": "nome",
        "columns": {
            "nome": "TEXT", "codigo_ufpel": "TEXT", "nivel": "TEXT",
            "grau": "TEXT", "modalidade": "TEXT", "turno": "TEXT",
            "codigo_emec": "TEXT", "codigo_capes": "TEXT", "unidade": "TEXT",
            "programa": "TEXT", "coordenador": "TEXT",
            "criacao_reconhecimento": "TEXT", "informacoes": "JSONB",
            "matriz_curricular": "JSONB", "professores": "JSONB",
            "turmas_ofertadas": "JSONB", "conceitos_curso": "JSONB",
            "formas_ingresso": "JSONB", "vagas_por_ingresso": "JSONB",
        },
    },
}


def ensure_info_tables() -> None:
    """Cria as tabelas estruturadas *_info (idempotente) com índice trigram no nome."""
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        for spec in INFO_TABLES.values():
            table = spec["table"]
            cols_sql = ",\n                ".join(
                f"{col} {sqltype}" for col, sqltype in spec["columns"].items()
            )
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    doc_id TEXT PRIMARY KEY,
                    {cols_sql},
                    url    TEXT
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_{spec['name_col']}_trgm
                    ON {table} USING gin ({spec['name_col']} gin_trgm_ops)
            """)
        print(f"[Tabelas] {len(INFO_TABLES)} tabelas estruturadas (*_info) verificadas/criadas.")
    finally:
        cur.close()
        conn.close()


def store_info_tables(pages: list[dict]) -> int:
    """
    Popula as tabelas *_info a partir de dados_completos (upsert por doc_id).
    Colunas JSONB recebem listas/dicts serializados; TEXT recebe o valor cru.
    Retorna o total de registros inseridos/atualizados.
    """
    import json as _json

    by_tipo: dict[str, list[tuple]] = {}
    for p in pages:
        tipo = p.get("tipo", "")
        spec = INFO_TABLES.get(tipo)
        if not spec or not p.get("id") or not p.get("dados_completos"):
            continue
        dc  = p["dados_completos"]
        url = (p.get("metadata") or {}).get("url", "")
        row = [p["id"]]
        for col, sqltype in spec["columns"].items():
            val = dc.get(col)
            if sqltype == "JSONB":
                row.append(_json.dumps(val, ensure_ascii=False) if val is not None else None)
            else:
                row.append(val)
        row.append(url)
        by_tipo.setdefault(tipo, []).append(tuple(row))

    if not by_tipo:
        return 0

    total = 0
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                for tipo, rows in by_tipo.items():
                    spec = INFO_TABLES[tipo]
                    cols = ["doc_id"] + list(spec["columns"].keys()) + ["url"]
                    updates = ", ".join(
                        f"{c} = EXCLUDED.{c}" for c in cols if c != "doc_id"
                    )
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {spec['table']} ({', '.join(cols)})
                        VALUES %s
                        ON CONFLICT (doc_id) DO UPDATE SET {updates}
                        """,
                        rows,
                    )
                    total += len(rows)
    finally:
        conn.close()
    return total


def lookup_registro(
    tipo: str,
    nome: str,
    top_k: int = 3,
    similarity_threshold: float = 0.3,
) -> list[dict]:
    """
    Busca registros estruturados na tabela *_info do tipo pelo nome (pg_trgm).

    Exemplo: lookup_registro('disciplina', 'Aprendizado de Máquina') retorna
    todas as colunas estruturadas da disciplina (ementa, créditos, turmas, etc.)
    prontas para compor o contexto do LLM.
    """
    spec = INFO_TABLES.get(tipo)
    if spec is None:
        return []
    table, name_col = spec["table"], spec["name_col"]
    cols = ["doc_id"] + list(spec["columns"].keys()) + ["url"]

    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute(f"SET pg_trgm.similarity_threshold = {similarity_threshold}")
            cur.execute(
                f"""
                SELECT {', '.join(cols)}, similarity({name_col}, %s) AS sim
                FROM {table}
                WHERE {name_col} %% %s
                ORDER BY sim DESC
                LIMIT %s
                """,
                (nome, nome, top_k),
            )
            rows = cur.fetchall()
            return [
                {**dict(zip(cols, r[:-1])), "similarity": float(r[-1])}
                for r in rows
            ]
    finally:
        conn.close()


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
    Cada tabela:  id BIGSERIAL PK | doc_id TEXT | titulo TEXT | conteudo TEXT
                  | embedding VECTOR({EMBEDDING_DIMS})
    doc_id liga cada chunk ao registro estruturado (*_info / doc_completos).
    """
    conn = psycopg2.connect(**config.DB_CONFIG)
    conn.autocommit = True
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

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
                        doc_id    TEXT,
                        titulo    TEXT,
                        conteudo  TEXT       NOT NULL,
                        embedding VECTOR({config.EMBEDDING_DIMS})
                    )
                """)
            else:
                # Tabela pré-existente da versão antiga: adiciona colunas de ligação
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS doc_id TEXT")
                cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS titulo TEXT")

            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_embedding_hnsw
                    ON {table} USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_doc_id_idx ON {table} (doc_id)")
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table}_titulo_trgm
                    ON {table} USING gin (titulo gin_trgm_ops)
            """)

        print(f"[Tabelas] {len(TIPO_TO_TABLE)} tabelas verificadas/criadas "
              f"(dims={config.EMBEDDING_DIMS}).")
    finally:
        cur.close()
        conn.close()


# =============================================================================
# Chunking por documento (1 doc = 1 chunk com o embedding_text inteiro)
# =============================================================================

# nv-embedqa-e5-v5 aceita no máximo 512 tokens (~2000 chars em português).
# Textos maiores são truncados pela API (truncate="END") — o embedding
# representa apenas o início do documento.
NVEMBED_SAFE_CHARS = 2000


def chunk_documents_whole(documents: list[Document]) -> list[Document]:
    """
    Modo 'documento': cada documento vira UM único chunk com o texto completo
    (embedding_text inteiro), sem divisão por tamanho. Mantém cada registro
    (disciplina, projeto, servidor, curso) íntegro no índice vetorial.
    """
    longos = sum(1 for d in documents if len(d.page_content) > NVEMBED_SAFE_CHARS)
    print(f"[Chunking Documento] {len(documents)} doc(s) → {len(documents)} chunks "
          f"(1 doc = 1 chunk, texto integral)")
    if longos:
        print(f"  [Aviso] {longos} doc(s) excedem ~{NVEMBED_SAFE_CHARS} chars "
              f"(512 tokens do nv-embedqa-e5-v5) — o embedding será truncado no fim.")
    return documents


CHUNKING_STRATEGIES = {
    "documento": chunk_documents_whole,
    "semantico": chunk_documents_semantic,
    "recursivo": chunk_documents,
}


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
    metadatas: list[dict] | None = None,
) -> None:
    """
    Insere um lote de (texto, vetor) diretamente na tabela física `table`,
    com doc_id e titulo dos metadados para ligação com as tabelas *_info.
    Usa execute_values para eficiência e o cast `::vector` para compatibilidade
    com pgvector sem depender do adaptador psycopg2 do pgvector.
    """
    metadatas = metadatas or [{}] * len(texts)
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        with conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {table} (doc_id, titulo, conteudo, embedding) VALUES %s",
                    [
                        (
                            meta.get("doc_id"),
                            meta.get("titulo") or meta.get("title"),
                            text,
                            f"[{','.join(map(str, vec))}]",
                        )
                        for text, vec, meta in zip(texts, vectors, metadatas)
                    ],
                    template="(%s, %s, %s, %s::vector)",
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
    chunking: str = "documento",
) -> dict[str, int]:
    """
    Pipeline segmentada por tipo:

    1. Cria as tabelas físicas por tipo se não existirem (idempotente).
    2. Agrupa documentos pelo campo `tipo` dos metadados.
    3. Para cada grupo (limitado a `max_per_tipo` docs), processa em lotes:
         a. Faz chunking ('documento' = texto integral, 'semantico' ou 'recursivo')
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

    Estratégias de chunking (param `chunking`):
      'documento' (padrão) : 1 doc = 1 chunk com o embedding_text inteiro —
                             cada registro fica íntegro no índice vetorial.
      'semantico'          : respeita seções (Resumo:, Objetivos:, etc.) até CHUNK_SIZE.
      'recursivo'          : RecursiveCharacterTextSplitter clássico.

    Limites free tier Google (gemini-embedding-001):
      ~1 chamada/min efetiva → use --max-por-tipo 200 --delay 62
      Com 200 docs/tipo: ~2 000 chunks → ~20 lotes → ~20 min por tipo.

    Args:
        documents           : LangChain Documents com metadado `tipo` preenchido
        reset               : se True, trunca tabelas e recria coleções antes de inserir
        batch_size          : chunks por lote de embedding (padrão INGEST_BATCH_SIZE)
        delay               : segundos entre lotes (padrão INGEST_DELAY)
        max_per_tipo        : limite de documentos por tipo (None = sem limite)
        chunking            : 'documento' | 'semantico' | 'recursivo'

    Returns:
        Dict {collection_name: número_de_documentos_inseridos}
    """
    from collections import defaultdict

    chunk_fn = CHUNKING_STRATEGIES.get(chunking)
    if chunk_fn is None:
        raise ValueError(
            f"Chunking '{chunking}' inválido. Use: {', '.join(CHUNKING_STRATEGIES)}"
        )

    # 1. Garante existência das tabelas físicas e da tabela de dados completos
    ensure_tables_exist()
    ensure_dados_completos_table()
    ensure_info_tables()

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
        print(f"  Chunking  : {chunking}")
        print("=" * 62)

        chunks = chunk_fn(docs)
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
            _insert_raw_batch(table, texts, vectors, metadatas)

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
    p.add_argument("--chunking", choices=["documento", "semantico", "recursivo"],
                   default="documento",
                   help="Estratégia de chunking (padrão: documento — "
                        "1 doc = 1 chunk com o embedding_text inteiro)")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    pages = load_crawled_data(args.input)

    # Persiste dados_completos no banco (novo schema) para enriquecer contexto RAG
    ensure_dados_completos_table()
    n_stored = store_dados_completos(pages)
    if n_stored:
        print(f"[Ingestão] {n_stored} dados_completos armazenados na tabela doc_completos.")

    # Persiste as tabelas estruturadas por tipo (*_info) para consultas SQL diretas
    ensure_info_tables()
    n_info = store_info_tables(pages)
    if n_info:
        print(f"[Ingestão] {n_info} registros estruturados armazenados nas tabelas *_info.")

    documents = pages_to_documents(pages)

    if not documents:
        print("[Aviso] Nenhum documento válido. Verifique o arquivo JSON.")
        sys.exit(1)

    if args.all_in_one:
        ingest_to_pgvector(documents, reset=args.reset)
    else:
        ingest_segmented(
            documents,
            reset=args.reset,
            delay=args.delay,
            max_per_tipo=args.max_por_tipo,
            chunking=args.chunking,
        )


if __name__ == "__main__":
    main()
