#!/usr/bin/env bash
# =============================================================================
# setup_ambiente.sh — Configuração do ambiente para o Minicurso de RAG
# Pós-Graduação em Computação
#
# Sistema operacional testado: Ubuntu 24.04 LTS
# Execute com: bash setup_ambiente.sh
# =============================================================================

set -e  # interrompe se qualquer comando falhar

PROJETO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJETO_DIR/.venv"
DB_NAME="semanticdb"
DB_USER="gdlima"
DB_PASS="12345"

echo "=============================================="
echo "  Minicurso RAG — Configuração do Ambiente"
echo "=============================================="
echo ""

# ==============================================================================
# PASSO 1: PostgreSQL — Instalação
# ==============================================================================
echo "[1/7] Instalando PostgreSQL e dependências do sistema..."

sudo apt-get update -q

# postgresql-contrib: módulos extras (uuid-ossp, pg_stat_statements, etc.)
sudo apt-get install -y \
    postgresql \
    postgresql-contrib \
    postgresql-client \
    postgresql-16-pgvector \
    libpq-dev \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    git \
    curl \
    lsb-release \
    gnupg

echo "      OK — $(psql --version)"
echo "      OK — Python $(python3 --version)"

# Garante que o PostgreSQL está ativo e inicia com o sistema
sudo systemctl enable --now postgresql

# ==============================================================================
# PASSO 2: pgAdmin 4 — Instalação (interface web)
# Repositório oficial: https://www.pgadmin.org/download/pgadmin-4-apt/
# ==============================================================================
echo ""
echo "[2/7] Instalando pgAdmin 4 (interface web)..."

# Adiciona a chave GPG do repositório pgAdmin
curl -fsS https://www.pgadmin.org/static/packages_pgadmin_org.pub \
    | sudo gpg --dearmor -o /usr/share/keyrings/packages-pgadmin-org.gpg

# Adiciona o repositório ao apt
echo "deb [signed-by=/usr/share/keyrings/packages-pgadmin-org.gpg] \
https://ftp.postgresql.org/pub/pgadmin/pgadmin4/apt/$(lsb_release -cs) pgadmin4 main" \
    | sudo tee /etc/apt/sources.list.d/pgadmin4.list > /dev/null

sudo apt-get update -q
sudo apt-get install -y pgadmin4-web

echo "      OK — pgAdmin 4 instalado"
echo ""
echo "  *** ACAO NECESSARIA ***"
echo "  Execute o assistente de configuracao do pgAdmin (pedira e-mail e senha):"
echo ""
echo "      sudo /usr/pgadmin4/bin/setup-web.sh"
echo ""
echo "  Guarde as credenciais — serao usadas para acessar http://127.0.0.1/pgadmin4"
echo ""
read -rp "  Pressione ENTER apos concluir o setup-web.sh para continuar..."

# Verifica se o Apache (servidor do pgAdmin) está rodando
if ! sudo systemctl is-active --quiet apache2; then
    echo "  Apache parado — iniciando..."
    sudo systemctl start apache2
fi
sudo systemctl enable apache2
echo "      OK — Apache rodando. pgAdmin disponivel em http://127.0.0.1/pgadmin4"

# ==============================================================================
# PASSO 3: PostgreSQL — Configuração do banco, usuário e pgvector
# ==============================================================================
echo ""
echo "[3/7] Configurando PostgreSQL..."

# --- 3.1: Criar usuário de aplicação ---
# Verifica se o usuário já existe antes de criar
sudo -u postgres psql -v ON_ERROR_STOP=1 <<SQL
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '$DB_USER') THEN
    CREATE ROLE $DB_USER;
  END IF;
END
\$\$;

-- Define senha e permissões administrativas para uso no curso
ALTER USER $DB_USER WITH PASSWORD '$DB_PASS';
ALTER ROLE $DB_USER WITH SUPERUSER CREATEDB CREATEROLE LOGIN;
SQL

echo "      OK — usuario '$DB_USER' criado com permissoes de superusuario"

# Para testar a conexao pelo terminal (apos o script):
#   psql -h localhost -U $DB_USER -d postgres
#   Informe a senha: $DB_PASS

# --- 3.2: Criar banco de dados ---
sudo -u postgres psql -v ON_ERROR_STOP=0 \
    -c "CREATE DATABASE $DB_NAME OWNER $DB_USER;" 2>/dev/null || \
    echo "      (banco '$DB_NAME' ja existe — mantido)"

echo "      OK — banco '$DB_NAME' pronto"

# --- 3.3: Ativar extensão pgvector ---
sudo -u postgres psql -d "$DB_NAME" \
    -c "CREATE EXTENSION IF NOT EXISTS vector;"

echo "      OK — extensao pgvector ativada"

# --- 3.4: Criar tabelas de referência e índices vetoriais (uso didático) ---
# Estas tabelas são para demonstrar SQL puro do pgvector em aula.
# O LangChain PGVector cria suas próprias tabelas automaticamente
# (langchain_pg_collection + langchain_pg_embedding) com a dimensão correta.
#
# Modelo de embeddings adotado: NVIDIA NIM nvidia/nv-embedqa-e5-v5 → 1024 dims
# pgvector 0.6 suporta HNSW apenas até 2000 dims — 1024 é compatível.
# Para outros modelos / dimensões, ajuste EMBEDDING_DIMS em config.py e recrie as tabelas:
#   NVIDIA nv-embedqa-e5-v5                                 : 1024 dims  ← padrão deste projeto
#   Google gemini-embedding-001 (output_dimensionality=768) : 768 dims
#   Google gemini-embedding-001 nativo                      : 3072 dims (requer pgvector >= 0.7 / halfvec)
#   OpenAI text-embedding-3-small                           : 1536 dims
sudo -u postgres psql -d "$DB_NAME" <<SQL

-- ── Tabela genérica de referência ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documentos (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Tabela por tipo — disciplinas ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS disciplinas (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Tabela por tipo — projetos ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS projetos (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Tabela por tipo — servidores ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS servidores (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Tabela por tipo — unidades ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS unidades (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Tabela por tipo — cursos ─────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS cursos (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Tabela por tipo — portal_geral ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS portal_geral (
    id        BIGSERIAL PRIMARY KEY,
    conteudo  TEXT,
    embedding VECTOR(1024)
);

-- ── Índices HNSW por tabela ──────────────────────────────────────────────────
-- HNSW (Hierarchical Navigable Small World): busca aproximada muito eficiente
-- vector_cosine_ops → otimizado para distância por cosseno (padrão NLP)
CREATE INDEX IF NOT EXISTS documentos_embedding_hnsw
    ON documentos  USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS disciplinas_embedding_hnsw
    ON disciplinas USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS projetos_embedding_hnsw
    ON projetos    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS servidores_embedding_hnsw
    ON servidores  USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS unidades_embedding_hnsw
    ON unidades    USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS cursos_embedding_hnsw
    ON cursos      USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS portal_geral_embedding_hnsw
    ON portal_geral USING hnsw (embedding vector_cosine_ops);
SQL

echo "      OK — tabelas por tipo (disciplinas, projetos, servidores, unidades, cursos, portal_geral) e indices HNSW criados"

# ==============================================================================
# PASSO 4: pgAdmin — Registrar servidor (instruções manuais)
# ==============================================================================
echo ""
echo "[4/7] Instrucoes para registrar o servidor no pgAdmin:"
echo ""
echo "  1. Acesse: http://127.0.0.1/pgadmin4"
echo "  2. Clique em 'Add New Server'"
echo ""
echo "  Aba General:"
echo "    Name: PostgreSQL Local"
echo ""
echo "  Aba Connection:"
echo "    Host name/address : localhost"
echo "    Port              : 5432"
echo "    Maintenance DB    : postgres"
echo "    Username          : $DB_USER"
echo "    Password          : $DB_PASS"
echo ""
echo "  Clique em Save."
echo ""
read -rp "  Pressione ENTER para continuar com a instalacao Python..."

# ==============================================================================
# PASSO 5: Python — Ambiente virtual
# ==============================================================================
echo ""
echo "[5/7] Criando ambiente virtual Python em $VENV_DIR ..."

python3 -m venv "$VENV_DIR"

# Ativa o venv para os passos seguintes deste script
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Atualiza ferramentas base
pip install --quiet --upgrade pip setuptools wheel

echo "      OK — venv criado com $(python --version)"

# ==============================================================================
# PASSO 6: Python — Instalar dependências do projeto
# ==============================================================================
echo ""
echo "[6/7] Instalando pacotes Python (requirements.txt)..."

pip install --quiet -r "$PROJETO_DIR/requirements.txt"

echo "      OK — pacotes instalados"

# ==============================================================================
# PASSO 7: Criar arquivo .env com chaves de API
# ==============================================================================
echo ""
echo "[7/7] Configurando arquivo .env ..."

ENV_FILE="$PROJETO_DIR/.env"

if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<EOF
# NVIDIA NIM (OBRIGATORIO — embeddings; LLM de fallback)
# Obtenha em: https://build.nvidia.com
NVIDIA_API_KEY=sua_chave_aqui

# Google Gemini (OBRIGATORIO — LLM principal)
# Obtenha em: https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=sua_chave_aqui
EOF
    echo "      Arquivo .env criado. EDITE com as chaves de API antes de executar."
else
    echo "      .env ja existe — mantido sem alteracao."
fi

# ==============================================================================
# VERIFICAÇÃO FINAL
# ==============================================================================
echo ""
echo "=============================================="
echo "  Verificacao do ambiente"
echo "=============================================="

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python - <<'PYCHECK'
import sys
print(f"Python        : {sys.version.split()[0]}")
print()

packages = [
    ("langchain",                     "LangChain"),
    ("langchain_community",           "LangChain Community"),
    ("langchain_google_genai",        "LangChain Google Gemini"),
    ("langchain_nvidia_ai_endpoints", "LangChain NVIDIA NIM"),
    ("psycopg2",                      "psycopg2"),
    ("pgvector",                      "pgvector"),
    ("pypdf",                         "pypdf"),
    ("dotenv",                        "python-dotenv"),
    ("sqlalchemy",                    "SQLAlchemy"),
]

ok = True
for module, label in packages:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "ok")
        print(f"  [OK]    {label:<35} {ver}")
    except ImportError:
        print(f"  [FALTA] {label}")
        ok = False

if not ok:
    print("\nALERTA: alguns pacotes nao foram instalados corretamente.")
    sys.exit(1)
PYCHECK

# Testa conexão com o banco e valida extensão e índice
python - <<PGCHECK
import psycopg2, sys

try:
    conn = psycopg2.connect(
        dbname="$DB_NAME", user="$DB_USER",
        password="$DB_PASS", host="localhost", port=5432
    )
    cur = conn.cursor()

    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
    row = cur.fetchone()
    print(f"  [OK]    PostgreSQL + pgvector {row[0] if row else '(ativo)'}")

    tabelas = ['documentos', 'disciplinas', 'projetos', 'servidores',
               'unidades', 'cursos', 'portal_geral']
    cur.execute("""
        SELECT tablename FROM pg_tables
        WHERE schemaname='public' AND tablename = ANY(%s);
    """, (tabelas,))
    encontradas = {r[0] for r in cur.fetchall()}
    for t in tabelas:
        status = "OK" if t in encontradas else "FALTA"
        print(f"  [{status}]    Tabela '{t}'")

    conn.close()
except Exception as e:
    print(f"  [FALTA] PostgreSQL: {e}")
    sys.exit(1)
PGCHECK

echo ""
echo "=============================================="
echo "  Ambiente configurado com sucesso!"
echo ""
echo "  Verificar busca semantica (SQL de referencia):"
echo "    psql -h localhost -U $DB_USER -d $DB_NAME"
echo ""
echo "    -- Busca por cosseno:"
echo "    SELECT id, conteudo,"
echo "           embedding <=> '[0.1,0.2,...]' AS distancia"
echo "    FROM documentos"
echo "    ORDER BY embedding <=> '[0.1,0.2,...]'"
echo "    LIMIT 5;"
echo ""
echo "    -- Operadores disponiveis:"
echo "    <=>  distancia por cosseno (padrao NLP)"
echo "    <->  distancia euclidiana (L2)"
echo "    <#>  produto interno negativo"
echo ""
echo "  Para ativar o venv nas proximas sessoes:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  IMPORTANTE: edite o .env com as chaves de API:"
echo "    nano $ENV_FILE"
echo "=============================================="
