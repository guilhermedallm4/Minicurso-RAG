"""
Fábricas de Embeddings e LLM
=============================================================================
Conceito — SEÇÃO 1 e 2 do curso

Centraliza a seleção de provedor sem que os outros módulos precisem
conhecer detalhes de API key ou nome de modelo.

Embeddings : NVIDIA NIM — nvidia/nv-embedqa-e5-v5 (1024 dims)

LLM — ordem de preferência (avaliada uma vez na inicialização):
  1. NVIDIA NIM — deepseek-ai/deepseek-v4-pro   (mais capaz)
  2. NVIDIA NIM — deepseek-ai/deepseek-v4-flash  (mais rápido)
  3. OpenRouter  — deepseek/deepseek-v4-flash    (fallback externo)

A checagem de disponibilidade na NVIDIA é feita via GET /v1/models
na primeira chamada a get_llm() e o resultado fica em cache por
processo — sem overhead nas chamadas subsequentes.

Em runtime, se um modelo NVIDIA falhar (timeout, 5xx, etc.), o
FallbackLLM avança automaticamente para o próximo na cadeia.
"""
import os
import requests
import threading
from typing import Any

import config  # garante load_dotenv() antes de qualquer os.getenv()

from reliability import record_fallback, record_info

# ── Chaves ────────────────────────────────────────────────────────────────────

_raw_nvidia_key      = os.getenv("NVIDIA_API_KEY", "")
_raw_openrouter_key  = os.getenv("OPENROUTER_API_KEY", "")

NVIDIA_AVAILABLE     = bool(_raw_nvidia_key)     and _raw_nvidia_key     != "sua_chave_aqui"
OPENROUTER_AVAILABLE = bool(_raw_openrouter_key) and _raw_openrouter_key != "sua_chave_aqui"

# Mantido por compatibilidade com código legado
GOOGLE_AVAILABLE = False

# ── Embeddings — NVIDIA NIM ───────────────────────────────────────────────────
try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
    _nvidia_embeddings_lib = True
except ImportError:
    _nvidia_embeddings_lib = False
    NVIDIA_AVAILABLE = False

# ── Constantes de modelos LLM ─────────────────────────────────────────────────

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL_PRO    = "deepseek-ai/deepseek-v4-pro"
NVIDIA_MODEL_FLASH  = "deepseek-ai/deepseek-v4-flash"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "deepseek/deepseek-v4-flash"

# ── Cache da lista de provedores disponíveis ──────────────────────────────────
# Lista de tuplas (base_url, model_id, api_key, label) em ordem de preferência.
# Preenchida na primeira chamada; estável pelo tempo de vida do processo.

_provider_chain: list[tuple[str, str, str, str]] | None = None
_provider_chain_lock = threading.Lock()


def _fetch_nvidia_models(timeout: float = 6.0) -> set[str]:
    """Retorna o conjunto de model IDs disponíveis na NVIDIA NIM, ou {} em erro."""
    if not NVIDIA_AVAILABLE:
        return set()
    try:
        resp = requests.get(
            f"{NVIDIA_NIM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {_raw_nvidia_key}"},
            timeout=timeout,
        )
        if resp.ok:
            return {m["id"] for m in resp.json().get("data", [])}
        print(f"[providers] NVIDIA /models retornou {resp.status_code}")
    except Exception as exc:
        print(f"[providers] Aviso: não foi possível consultar NVIDIA NIM /models — {exc}")
    return set()


def _build_provider_chain() -> list[tuple[str, str, str, str]]:
    """
    Constrói a lista de provedores em ordem de preferência:
      1. NVIDIA pro  (se disponível)
      2. NVIDIA flash (se disponível)
      3. OpenRouter flash (fallback)

    Cada entrada: (base_url, model_id, api_key, label)
    """
    chain: list[tuple[str, str, str, str]] = []

    if NVIDIA_AVAILABLE:
        available = _fetch_nvidia_models()
        if NVIDIA_MODEL_PRO in available:
            chain.append((
                NVIDIA_NIM_BASE_URL, NVIDIA_MODEL_PRO,
                _raw_nvidia_key, f"NVIDIA NIM: {NVIDIA_MODEL_PRO}",
            ))
        if NVIDIA_MODEL_FLASH in available:
            chain.append((
                NVIDIA_NIM_BASE_URL, NVIDIA_MODEL_FLASH,
                _raw_nvidia_key, f"NVIDIA NIM: {NVIDIA_MODEL_FLASH}",
            ))

    if OPENROUTER_AVAILABLE:
        chain.append((
            OPENROUTER_BASE_URL, OPENROUTER_MODEL,
            _raw_openrouter_key, f"OpenRouter: {OPENROUTER_MODEL}",
        ))

    return chain


def _get_provider_chain() -> list[tuple[str, str, str, str]]:
    global _provider_chain
    if _provider_chain is not None:
        return _provider_chain
    with _provider_chain_lock:
        if _provider_chain is None:
            _provider_chain = _build_provider_chain()
            labels = [lbl for *_, lbl in _provider_chain]
            print(f"[providers] Cadeia LLM: {' → '.join(labels) if labels else 'vazia'}")
        return _provider_chain


def _promote_provider(working_index: int) -> None:
    """
    Remove da cadeia global todos os provedores antes de `working_index`.
    Chamado quando um fallback ocorre em runtime — garante que as próximas
    requisições comecem diretamente no provedor que está funcionando,
    sem repetir timeouts desnecessários.
    """
    global _provider_chain
    if working_index <= 0 or _provider_chain is None:
        return
    with _provider_chain_lock:
        if _provider_chain is not None and working_index < len(_provider_chain):
            removed  = [lbl for *_, lbl in _provider_chain[:working_index]]
            _provider_chain = _provider_chain[working_index:]
            new_primary = _provider_chain[0][3] if _provider_chain else "nenhum"
            print(f"[providers] Promovido para primário: {new_primary} "
                  f"(removidos: {removed})")
            record_info("llm", f"Novo provedor primário após fallback: {new_primary}")


# ── FallbackLLM ───────────────────────────────────────────────────────────────

class FallbackLLM:
    """
    Wrapper que tenta cada provedor na cadeia em ordem.
    Se um falhar (timeout, erro 5xx, etc.) avança para o próximo.

    Expõe `.invoke()` e `.stream()` — a mesma interface do ChatOpenAI —
    para que o resto do código não precise saber sobre a cadeia.
    """

    def __init__(self, temperature: float = 0.3):
        self._temperature = temperature
        self._chain = _get_provider_chain()

    def _make_llm(self, base_url: str, model: str, api_key: str):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=self._temperature,
            request_timeout=30,
            max_retries=1,          # 1 retry interno; o FallbackLLM faz o fallback externo
        )

    def _try_chain(self, method: str, *args, **kwargs) -> Any:
        import openai

        # Lê a cadeia atual no momento da chamada (pode ter sido promovida)
        chain = _get_provider_chain()

        last_exc: Exception | None = None
        for idx, (base_url, model, api_key, label) in enumerate(chain):
            try:
                record_info("llm", label)
                llm = self._make_llm(base_url, model, api_key)
                result = getattr(llm, method)(*args, **kwargs)
                if idx > 0:
                    # Sucesso após fallback — promove este provedor para primário
                    # nas próximas requisições, evitando novos timeouts desnecessários.
                    record_fallback(
                        "llm",
                        f"{chain[idx-1][3]} falhou",
                        fallback_to=label,
                    )
                    _promote_provider(idx)
                return result
            except (openai.APITimeoutError, openai.APIStatusError, Exception) as exc:
                last_exc = exc
                if idx < len(chain) - 1:
                    next_label = chain[idx + 1][3]
                    print(f"[providers] {label} falhou ({type(exc).__name__}) — tentando {next_label}")
                    record_fallback("llm", f"{label}: {type(exc).__name__}", fallback_to=next_label)
                else:
                    print(f"[providers] {label} falhou ({type(exc).__name__}) — sem mais provedores")

        raise RuntimeError(
            f"Todos os provedores LLM falharam. Último erro: {last_exc}"
        ) from last_exc

    def invoke(self, *args, **kwargs) -> Any:
        return self._try_chain("invoke", *args, **kwargs)

    def stream(self, *args, **kwargs):
        # stream é lazy — não podemos fazer fallback após iniciar; usa invoke internamente
        return self._try_chain("stream", *args, **kwargs)

    # Suporte a chamada direta (compatibilidade com chain | llm)
    def __call__(self, *args, **kwargs) -> Any:
        return self._try_chain("invoke", *args, **kwargs)

    # Pipe operator para LangChain (chain | llm | parser)
    def __or__(self, other):
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda x: self.invoke(x)) | other

    def __ror__(self, other):
        from langchain_core.runnables import RunnableLambda
        return other | RunnableLambda(lambda x: self.invoke(x))


# ── API pública ───────────────────────────────────────────────────────────────

def get_embeddings():
    """Retorna embeddings NVIDIA NIM: nvidia/nv-embedqa-e5-v5 (1024 dims)."""
    if not NVIDIA_AVAILABLE:
        record_fallback("embeddings", "NVIDIA_API_KEY não configurada ou inválida")
        raise EnvironmentError(
            "NVIDIA_API_KEY não configurada ou inválida. "
            "Defina NVIDIA_API_KEY no arquivo .env — obrigatório para embeddings."
        )
    record_info("embeddings", "NVIDIA NIM: nvidia/nv-embedqa-e5-v5 (1024 dims)")
    return NVIDIAEmbeddings(
        model="nvidia/nv-embedqa-e5-v5",
        api_key=_raw_nvidia_key,
        truncate="END",
    )


def get_llm(temperature: float = 0.3) -> FallbackLLM:
    """
    Retorna um FallbackLLM que tenta os provedores nesta ordem:
      1. NVIDIA NIM deepseek-v4-pro
      2. NVIDIA NIM deepseek-v4-flash
      3. OpenRouter deepseek-v4-flash

    A disponibilidade na NVIDIA é checada via /v1/models na primeira chamada
    e fica em cache — sem overhead nas chamadas seguintes.
    Se um provedor falhar em runtime (timeout/5xx), avança para o próximo.
    """
    chain = _get_provider_chain()
    if not chain:
        raise EnvironmentError(
            "Nenhum provedor LLM disponível. Configure NVIDIA_API_KEY ou OPENROUTER_API_KEY no .env."
        )
    return FallbackLLM(temperature=temperature)


def get_llm_provider_info() -> dict:
    """Retorna informações sobre o provedor LLM primário ativo."""
    chain = _get_provider_chain()
    if not chain:
        return {"label": "nenhum", "model": "", "provider": "none"}
    base_url, model, _, label = chain[0]
    return {
        "label":    label,
        "model":    model,
        "provider": "nvidia" if "nvidia" in base_url else "openrouter",
        "chain":    [lbl for *_, lbl in chain],
    }


# =============================================================================
# Execução direta: testa os provedores disponíveis
# python providers.py
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  SEÇÃO 1 — Provedores de Embeddings e LLM")
    print("=" * 60)
    print(f"\n  NVIDIA disponível    : {NVIDIA_AVAILABLE}  (embeddings + LLM)")
    print(f"  OpenRouter disponível: {OPENROUTER_AVAILABLE}  (LLM fallback)")

    info = get_llm_provider_info()
    print(f"\n  Provedor primário : {info['label']}")
    print(f"  Cadeia completa   : {' → '.join(info['chain'])}")

    emb = get_embeddings()
    vetor = emb.embed_query("teste de embedding em português")
    print(f"\n  Embedding gerado: {len(vetor)} dims | primeiros 5: {[round(v,4) for v in vetor[:5]]}")

    print("\n  Testando LLM (com fallback automático)...")
    llm = get_llm()
    resposta = llm.invoke("Responda em uma frase: o que é um embedding?")
    print(f"\n  LLM respondeu: {resposta.content}")
