"""
Configurações centralizadas do projeto RAG.
Único ponto de verdade para credenciais, conexão e constantes.
"""
import os
import warnings
from dotenv import load_dotenv

# Deve ser o primeiro módulo importado por qualquer outro arquivo do projeto,
# garantindo que os valores do .env estejam disponíveis antes de qualquer
# os.getenv() fora deste arquivo.
load_dotenv()

warnings.filterwarnings("ignore", message=".*langchain-community.*")

# =============================================================================
# BANCO DE DADOS
# =============================================================================

DB_CONFIG = {
    "host":     os.getenv("DB_HOST",   "localhost"),
    "port":     int(os.getenv("DB_PORT", "5432")),
    "dbname":   os.getenv("DB_NAME",   "semanticdb"),
    "user":     os.getenv("DB_USER",   "gdlima"),
    "password": os.getenv("DB_PASS",   "12345"),
}

CONNECTION_STRING = (
    f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# =============================================================================
# VECTOR STORE
# =============================================================================

# Coleção legado usada pelos exemplos didáticos de store.py
COLLECTION_NAME = "documentos_institucionais"

# Limiar mínimo de relevância: score LangChain = 1 − distância_cosseno
# 1.0 = idêntico ao query  |  0.0 = sem relação semântica
RELEVANCE_THRESHOLD = 0.2

# Dimensão dos vetores — NVIDIA nv-embedqa-e5-v5 usa 1024 dims
# pgvector 0.6 suporta HNSW até 2000 dims; 1024 é compatível.
EMBEDDING_DIMS = 1024

# =============================================================================
# COLLECTIONS POR TIPO — Portal UFPel
# =============================================================================
# Para adicionar um novo tipo: inclua uma entrada em COLLECTION_MAP e
# uma descrição correspondente em COLLECTION_DESCRIPTIONS.
# O roteador e a ingestão segmentada detectam automaticamente os novos tipos.

# Mapeamento tipo (campo 'tipo' no JSON crawleado) → nome da coleção pgvector
COLLECTION_MAP: dict[str, str] = {
    "disciplina":   "ufpel_disciplinas",
    "projeto":      "ufpel_projetos",
    "servidor":     "ufpel_servidores",
    "unidade":      "ufpel_unidades",
    "curso":        "ufpel_cursos",
    "portal_geral": "ufpel_portal_geral",
}

# Descrições exibidas ao LLM no prompt de roteamento.
# Chave = nome da coleção (valor de COLLECTION_MAP).
COLLECTION_DESCRIPTIONS: dict[str, str] = {
    "ufpel_disciplinas": (
        "disciplinas acadêmicas: ementa, objetivos, conteúdo programático, "
        "carga horária, departamento responsável"
    ),
    "ufpel_projetos": (
        "projetos de pesquisa e extensão: título, resumo, coordenador, "
        "área CNPq, ênfase, unidade de origem"
    ),
    "ufpel_servidores": (
        "professores e servidores técnico-administrativos: nome, cargo, "
        "unidade de lotação, titulação, regime de trabalho"
    ),
    "ufpel_unidades": (
        "unidades, centros, institutos e departamentos: endereço, "
        "direção/chefia, contato, estrutura organizacional"
    ),
    "ufpel_cursos": (
        "cursos de graduação e pós-graduação: duração, turno, vagas, "
        "grade curricular, coordenação"
    ),
    "ufpel_portal_geral": (
        "informações institucionais gerais que não se encaixam nas "
        "categorias acima (editais, notícias, eventos, etc.)"
    ),
}

ALL_COLLECTIONS: list[str] = list(COLLECTION_MAP.values())


def collection_for_tipo(tipo: str) -> str:
    """Retorna o nome da coleção pgvector para um dado tipo de página."""
    return COLLECTION_MAP.get(tipo, COLLECTION_MAP["portal_geral"])

# =============================================================================
# CHUNKING
# =============================================================================

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
