"""
Fábricas de Embeddings e LLM
=============================================================================
Conceito — SEÇÃO 1 e 2 do curso

Centraliza a seleção de provedor sem que os outros módulos precisem
conhecer detalhes de API key ou nome de modelo.

Embeddings : NVIDIA NIM — nvidia/nv-embedqa-e5-v5 (1024 dims)
LLM        : OpenRouter — deepseek/deepseek-v4-flash
             Interface OpenAI-compatível (openrouter.ai/api/v1)

Como ambas as opções usam APIs externas, NÃO é necessária GPU local.

Por que NVIDIA para embeddings:
  - Sem restrição de rate limit agressiva
  - Otimizado para QA/retrieval em português e inglês
  - 1024 dims: compatível com HNSW do pgvector 0.6 (limite 2000 dims)

Por que OpenRouter para LLM:
  - Acesso unificado a dezenas de modelos via API OpenAI-compatível
  - Modelo padrão: deepseek/deepseek-v4-flash (rápido e econômico)
"""
import os
import config  # garante load_dotenv() antes de qualquer os.getenv()

# ── Embeddings — NVIDIA NIM ───────────────────────────────────────────────────
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    _raw_nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_AVAILABLE = bool(_raw_nvidia_key) and _raw_nvidia_key != "sua_chave_aqui"
except ImportError:
    NVIDIA_AVAILABLE = False

# ── LLM — OpenRouter ─────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

_raw_openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_AVAILABLE = bool(_raw_openrouter_key) and _raw_openrouter_key != "sua_chave_aqui"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "deepseek/deepseek-v4-flash"

# Mantido por compatibilidade com código legado que importa essas constantes
GOOGLE_AVAILABLE = False


def get_embeddings():
    """Retorna embeddings NVIDIA NIM: nvidia/nv-embedqa-e5-v5 (1024 dims)."""
    if not NVIDIA_AVAILABLE:
        raise EnvironmentError(
            "NVIDIA_API_KEY não configurada ou inválida. "
            "Defina NVIDIA_API_KEY no arquivo .env — obrigatório para embeddings."
        )
    print("[Embeddings] NVIDIA NIM: nvidia/nv-embedqa-e5-v5 (1024 dims)")
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=os.getenv("NVIDIA_API_KEY"),
        truncate="END",
    )


def get_llm(temperature: float = 0.2):
    """
    Retorna o LLM via OpenRouter (deepseek/deepseek-v4-flash).

    Usa a interface OpenAI-compatível do OpenRouter, sem necessidade de
    biblioteca proprietária — apenas langchain-openai com base_url customizada.
    """
    if not OPENROUTER_AVAILABLE:
        raise EnvironmentError(
            "OPENROUTER_API_KEY não configurada ou inválida. "
            "Defina OPENROUTER_API_KEY no arquivo .env."
        )

    print(f"[LLM] OpenRouter: {OPENROUTER_MODEL}")
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base=OPENROUTER_BASE_URL,
        temperature=temperature,
    )


# =============================================================================
# Execução direta: testa os provedores disponíveis
# python providers.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SEÇÃO 1 — Provedores de Embeddings e LLM")
    print("=" * 60)
    print(f"\n  NVIDIA disponível    : {NVIDIA_AVAILABLE}  (embeddings)")
    print(f"  OpenRouter disponível: {OPENROUTER_AVAILABLE}  (LLM)")
    print(f"  Modelo LLM           : {OPENROUTER_MODEL}")

    emb = get_embeddings()
    vetor = emb.embed_query("teste de embedding em português")
    print(f"\n  Embedding gerado: {len(vetor)} dims | primeiros 5: {[round(v, 4) for v in vetor[:5]]}")

    print()
    llm = get_llm()
    resposta = llm.invoke("Responda em uma frase: o que é um embedding?")
    print(f"\n  LLM respondeu: {resposta.content}")
