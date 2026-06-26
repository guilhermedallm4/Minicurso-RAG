"""
Cache de respostas em memória — H2IA Chatbot UFPel
=============================================================================
Evita re-executar o pipeline completo para perguntas idênticas ou muito
similares. Funciona em dois níveis:

  Nível 1 — Exact match (hash SHA-256 normalizado)
    Custo: O(1) dict lookup. Hit rate alto para perguntas repetidas.

  Nível 2 — Fuzzy match (trigrama)
    Se não há exact match, compara a query com as entradas em cache usando
    similaridade de trigrama. Threshold configurável (padrão 0.85).
    Útil para variações mínimas: "Qual o prof de CC?" vs "qual prof de cc?".

TTL configurável por entrada (padrão 1 hora). Cache limpo automaticamente
a cada acesso quando > MAX_SIZE entradas.

Uso:
    from cache import response_cache
    hit = response_cache.get(query)
    if hit:
        return hit
    resposta = pipeline(query)
    response_cache.set(query, resposta)
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# ── Configurações ─────────────────────────────────────────────────────────────

CACHE_TTL_SECONDS  = 3600   # 1 hora por entrada
CACHE_MAX_SIZE     = 512    # entradas máximas antes de evictar mais antigas
FUZZY_THRESHOLD    = 0.82   # similaridade mínima para fuzzy hit


# ── Trigrama simples (sem dependência externa) ────────────────────────────────

def _trigrams(text: str) -> set[str]:
    t = f"  {text}  "
    return {t[i:i+3] for i in range(len(t) - 2)}


def _trgm_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _normalize(query: str) -> str:
    """Normaliza a query para melhorar o hit rate."""
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?!.,;]$", "", q).strip()
    return q


# ── Entrada de cache ──────────────────────────────────────────────────────────

@dataclass
class _CacheEntry:
    query_norm: str
    response:   str
    created_at: float = field(default_factory=time.time)
    hits:       int   = 0

    def is_expired(self, ttl: float = CACHE_TTL_SECONDS) -> bool:
        return (time.time() - self.created_at) > ttl

    def touch(self) -> None:
        self.hits += 1


# ── Cache principal ───────────────────────────────────────────────────────────

class ResponseCache:
    def __init__(
        self,
        ttl: float = CACHE_TTL_SECONDS,
        max_size: int = CACHE_MAX_SIZE,
        fuzzy_threshold: float = FUZZY_THRESHOLD,
    ):
        self._ttl             = ttl
        self._max_size        = max_size
        self._fuzzy_threshold = fuzzy_threshold
        self._exact: dict[str, _CacheEntry] = {}   # sha256 → entry
        self._lock  = threading.Lock()
        self._total_hits   = 0
        self._total_misses = 0

    # ── chave ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _key(query_norm: str) -> str:
        return hashlib.sha256(query_norm.encode()).hexdigest()

    # ── eviction ──────────────────────────────────────────────────────────────

    def _evict(self) -> None:
        """Remove entradas expiradas; se ainda > max_size, remove as mais antigas."""
        now = time.time()
        expired = [k for k, e in self._exact.items() if (now - e.created_at) > self._ttl]
        for k in expired:
            del self._exact[k]
        if len(self._exact) > self._max_size:
            oldest = sorted(self._exact.items(), key=lambda x: x[1].created_at)
            for k, _ in oldest[:len(self._exact) - self._max_size]:
                del self._exact[k]

    # ── get ───────────────────────────────────────────────────────────────────

    def get(self, query: str) -> Optional[str]:
        norm = _normalize(query)
        key  = self._key(norm)

        with self._lock:
            # Nível 1: exact match
            entry = self._exact.get(key)
            if entry and not entry.is_expired(self._ttl):
                entry.touch()
                self._total_hits += 1
                print(f"[Cache] HIT exato (hits={entry.hits}, age={int(time.time()-entry.created_at)}s)")
                return entry.response

            # Nível 2: fuzzy match
            if self._fuzzy_threshold < 1.0:
                for e in self._exact.values():
                    if e.is_expired(self._ttl):
                        continue
                    sim = _trgm_similarity(norm, e.query_norm)
                    if sim >= self._fuzzy_threshold:
                        e.touch()
                        self._total_hits += 1
                        print(f"[Cache] HIT fuzzy (sim={sim:.2f}, hits={e.hits})")
                        return e.response

            self._total_misses += 1
            return None

    # ── set ───────────────────────────────────────────────────────────────────

    def set(self, query: str, response: str) -> None:
        norm = _normalize(query)
        key  = self._key(norm)
        with self._lock:
            if len(self._exact) >= self._max_size:
                self._evict()
            self._exact[key] = _CacheEntry(query_norm=norm, response=response)

    # ── invalidate ────────────────────────────────────────────────────────────

    def invalidate(self, query: str) -> None:
        norm = _normalize(query)
        key  = self._key(norm)
        with self._lock:
            self._exact.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._exact.clear()

    # ── stats ─────────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        with self._lock:
            total = self._total_hits + self._total_misses
            return {
                "entries":    len(self._exact),
                "hits":       self._total_hits,
                "misses":     self._total_misses,
                "hit_rate":   f"{self._total_hits/total:.1%}" if total else "N/A",
                "ttl_s":      self._ttl,
                "max_size":   self._max_size,
            }


# ── Instância global ──────────────────────────────────────────────────────────

response_cache = ResponseCache()
