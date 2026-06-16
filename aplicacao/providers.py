"""
Fábricas de Embeddings e LLM
=============================================================================
Conceito — SEÇÃO 1 e 2 do curso

Centraliza a seleção de provedor sem que os outros módulos precisem
conhecer detalhes de API key ou nome de modelo.

Embeddings : NVIDIA NIM — nvidia/nv-embedqa-e5-v5 (1024 dims)
LLM        : Google Gemini 2.5 Flash → NVIDIA NIM (llama-3.3-70b)

Como ambas as opções usam APIs externas, NÃO é necessária GPU local.

Por que NVIDIA para embeddings:
  - Sem restrição de rate limit agressiva (free tier Google: ~1 req/min)
  - Otimizado para QA/retrieval em português e inglês
  - 1024 dims: compatível com HNSW do pgvector 0.6 (limite 2000 dims)
"""
import os
import config  # garante load_dotenv() antes de qualquer os.getenv()

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
    _raw_nvidia_key = os.getenv("NVIDIA_API_KEY", "")
    NVIDIA_AVAILABLE = bool(_raw_nvidia_key) and _raw_nvidia_key != "sua_chave_aqui"
except ImportError:
    NVIDIA_AVAILABLE = False

_raw_google_key = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_AVAILABLE = bool(_raw_google_key) and _raw_google_key != "sua_chave_aqui"

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI


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


def get_llm():
    """Retorna o LLM disponível: Gemini 2.5 Flash → NVIDIA NIM (llama-3.3-70b)."""
    if GOOGLE_AVAILABLE:
        print("[LLM] Google Gemini 2.5 Flash")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
        )
    if NVIDIA_AVAILABLE:
        print("[LLM] NVIDIA NIM: meta/llama-3.3-70b-instruct")
        return ChatNVIDIA(
            model="meta/llama-3.3-70b-instruct",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.2,
        )
    raise EnvironmentError(
        "Nenhuma chave de LLM configurada. "
        "Defina GOOGLE_API_KEY ou NVIDIA_API_KEY no arquivo .env"
    )


# =============================================================================
# Execução direta: testa os provedores disponíveis
# python providers.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("  SEÇÃO 1 — Provedores de Embeddings e LLM")
    print("=" * 50)
    print(f"\n  NVIDIA disponível : {NVIDIA_AVAILABLE}  (embeddings + LLM fallback)")
    print(f"  Google disponível : {GOOGLE_AVAILABLE}  (LLM principal)\n")

    emb = get_embeddings()
    vetor = emb.embed_query("teste de embedding em português")
    print(f"\n  Embedding gerado com sucesso!")
    print(f"  Dimensões : {len(vetor)}")
    print(f"  Primeiros 5 valores : {[round(v, 4) for v in vetor[:5]]}")

    print()
    llm = get_llm()
    resposta = llm.invoke("Responda em uma frase: o que é um embedding?")
    print(f"\n  LLM respondeu: {resposta.content}")
