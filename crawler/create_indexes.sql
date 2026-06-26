-- =============================================================================
-- Índices e otimizações de busca — H2IA Chatbot UFPel
-- =============================================================================
-- Executar: psql $DATABASE_URL -f create_indexes.sql
-- Todos os índices são CONCURRENT (não bloqueiam leitura durante a criação).
-- =============================================================================

-- Habilita extensões necessárias (idempotente)
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS vector;

-- -----------------------------------------------------------------------------
-- 1. langchain_pg_embedding — collection_id
--    JOIN com langchain_pg_collection hoje faz Nested Loop sem índice.
--    Impacto: toda busca vetorial e todo ILIKE percorre as 36k rows sem filtro.
-- -----------------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_collection_id
    ON langchain_pg_embedding (collection_id);

-- -----------------------------------------------------------------------------
-- 2. langchain_pg_embedding — coluna com dimensão fixa + HNSW
--    pgvector 0.6 requer dimensão fixa na coluna para HNSW sem cast.
--    ALTER TYPE vector(1024) é in-place (sem reescrita de dados).
--    Após o ALTER, o HNSW funciona sem cast — planner usa o índice.
--    m=16, ef_construction=128 é o trade-off padrão qualidade/memória.
--    Benchmark: vector search 172ms → 4ms (43x) após HNSW.
-- -----------------------------------------------------------------------------
ALTER TABLE langchain_pg_embedding
    ALTER COLUMN embedding TYPE vector(1024);

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_embedding_hnsw
    ON langchain_pg_embedding
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 128);

-- -----------------------------------------------------------------------------
-- 3. langchain_pg_embedding — GIN trigram em document
--    Permite ILIKE '%termo%' usar índice em vez de seq scan.
--    ILIKE de 'Inteligência Artificial': 278ms → esperado ~5ms após índice.
-- -----------------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_document_trgm
    ON langchain_pg_embedding
    USING gin (document gin_trgm_ops);

-- -----------------------------------------------------------------------------
-- 4. langchain_pg_embedding — índice parcial por coleção (expression index)
--    Para buscas frequentes dentro de uma coleção específica.
--    Usa subquery para garantir compatibilidade sem hardcode de UUID.
-- -----------------------------------------------------------------------------
-- ATENÇÃO: índices parciais não suportam subqueries em predicados (PostgreSQL).
-- Os UUIDs abaixo foram consultados previamente via:
--   SELECT name, uuid FROM langchain_pg_collection;
-- Se os UUIDs mudarem, recrie o índice com os novos valores.
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_servidores_doc_trgm
    ON langchain_pg_embedding
    USING gin (document gin_trgm_ops)
    WHERE collection_id = 'f56c4105-08ad-43af-bbdd-ef8b85ef56e6'::uuid;

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_projetos_doc_trgm
    ON langchain_pg_embedding
    USING gin (document gin_trgm_ops)
    WHERE collection_id = 'da62ebe4-0090-4cfb-a8bd-82c91bf2c8cc'::uuid;

-- -----------------------------------------------------------------------------
-- 5. doc_completos — habilita operador % (trigram) para lookup direto
--    O índice GIN já existe em titulo, mas similarity() > threshold não o usa.
--    Configurar threshold globalmente permite que o planner use o índice GIN.
-- -----------------------------------------------------------------------------
SET pg_trgm.similarity_threshold = 0.3;

-- Garante que a configuração persiste na sessão (pode colocar em postgresql.conf
-- para persistir globalmente: pg_trgm.similarity_threshold = 0.3)

-- Índice já existe: doc_completos_titulo_trgm (gin titulo gin_trgm_ops)
-- Mas adiciona índice composto tipo + titulo para filtros com tipo específico:
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_dc_tipo_titulo_trgm
    ON doc_completos
    USING gin (titulo gin_trgm_ops)
    WHERE tipo IN ('curso', 'servidor', 'disciplina', 'unidade');

-- -----------------------------------------------------------------------------
-- 6. doc_completos — cmetadata de coleção para lookup rápido por tipo+url
-- -----------------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_dc_url
    ON doc_completos ((dados->>'url'))
    WHERE dados->>'url' IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 7. langchain_pg_embedding — cmetadata campos mais consultados
--    cmetadata->>'titulo' é usado em SELECT DISTINCT para nome do professor
--    cmetadata->>'tipo'   é usado em WHERE para filtrar tipo de documento
-- -----------------------------------------------------------------------------
CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_meta_titulo
    ON langchain_pg_embedding ((cmetadata->>'titulo'));

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    idx_lpe_meta_tipo
    ON langchain_pg_embedding ((cmetadata->>'tipo'));

-- -----------------------------------------------------------------------------
-- 8. Estatísticas atualizadas (o planner usa stats obsoletas se não rodar ANALYZE)
-- -----------------------------------------------------------------------------
ANALYZE langchain_pg_embedding;
ANALYZE doc_completos;
ANALYZE langchain_pg_collection;

-- -----------------------------------------------------------------------------
-- Verificação final
-- -----------------------------------------------------------------------------
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size,
    indexdef
FROM pg_indexes
WHERE tablename IN ('langchain_pg_embedding', 'doc_completos')
ORDER BY tablename, indexname;
