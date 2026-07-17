"""
Histórico de Resultados por Sessão — Paginação de Contexto
=============================================================================
Quando uma busca semântica retorna mais documentos relevantes do que o
TOP-K enviado ao LLM, o excedente NÃO é descartado: fica guardado aqui,
como um histórico em cache por sessão de aluno.

Fluxo:
  1ª pergunta  → busca traz TODOS os candidatos relevantes
               → 4 primeiros vão ao LLM, o restante fica neste cache
  "fale outras" → próxima página (4 itens) sai direto do cache,
                  sem nova busca vetorial, até esgotar a lista.

Cada sessão (aluno) tem seu próprio estado, identificado por session_id
(UUID do st.session_state no Streamlit; "cli" no chatbot de terminal).
TTL de 30 min — depois disso a lista expira e uma nova busca é necessária.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

STATE_TTL_SECONDS = 1800   # 30 min por lista pendente
MAX_SESSIONS      = 256    # sessões simultâneas antes de evictar as mais antigas


@dataclass
class _SessionState:
    query:      str            # pergunta original que gerou a lista
    collection: Optional[str]  # coleção usada na busca (para o prompt certo)
    items:      list           # [(Document, score)] ranqueados por relevância
    offset:     int = 0        # índice do próximo item ainda não apresentado
    created_at: float = field(default_factory=time.time)

    def is_expired(self, ttl: float) -> bool:
        return (time.time() - self.created_at) > ttl


class PaginationStore:
    """Cache thread-safe de listas de resultados pendentes, por sessão."""

    def __init__(self, ttl: float = STATE_TTL_SECONDS, max_sessions: int = MAX_SESSIONS):
        self._ttl = ttl
        self._max = max_sessions
        self._sessions: dict[str, _SessionState] = {}
        self._lock = threading.Lock()

    def save(
        self,
        session_id: str,
        query: str,
        collection: Optional[str],
        items: list,
        offset: int = 0,
    ) -> None:
        """Registra a lista completa de resultados e quantos já foram mostrados."""
        with self._lock:
            self._evict()
            self._sessions[session_id] = _SessionState(
                query=query, collection=collection, items=items, offset=offset,
            )

    def next_page(self, session_id: str, n: int) -> Optional[tuple]:
        """
        Avança a lista da sessão e retorna (page, remaining, query, collection).

        page      : próximos até `n` itens [(Document, score)] — [] se esgotou
        remaining : quantos ainda restam após esta página
        Retorna None se a sessão não tem lista pendente (ou expirou).
        """
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None:
                return None
            if state.is_expired(self._ttl):
                del self._sessions[session_id]
                return None
            page = state.items[state.offset: state.offset + n]
            state.offset += len(page)
            remaining = len(state.items) - state.offset
            return page, remaining, state.query, state.collection

    def remaining(self, session_id: str) -> int:
        """Quantos itens ainda não foram apresentados nesta sessão."""
        with self._lock:
            state = self._sessions.get(session_id)
            if state is None or state.is_expired(self._ttl):
                return 0
            return len(state.items) - state.offset

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def _evict(self) -> None:
        """Remove sessões expiradas; se ainda acima do limite, as mais antigas."""
        now = time.time()
        expired = [k for k, s in self._sessions.items()
                   if (now - s.created_at) > self._ttl]
        for k in expired:
            del self._sessions[k]
        if len(self._sessions) >= self._max:
            oldest = sorted(self._sessions.items(), key=lambda x: x[1].created_at)
            for k, _ in oldest[:len(self._sessions) - self._max + 1]:
                del self._sessions[k]


# Instância global — compartilhada por app.py (Streamlit) e chatbot.py (CLI)
pagination_store = PaginationStore()
