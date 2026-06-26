"""
Métricas de Reliability e Availability — Chatbot UFPel
=============================================================================
Coleta em memória (sem dependência externa) das métricas operacionais:

  - MTBF   : Mean Time Between Failures (média de tempo entre falhas)
  - MTTR   : Mean Time To Recovery (tempo médio para recuperar)
  - Taxa de falha por componente (LLM, Embeddings, SQL, RAG)
  - Disponibilidade = uptime / (uptime + downtime)
  - Fallbacks acionados por API indisponível

Uso:
    from reliability import record_request, record_failure, record_fallback, get_report
    with record_request("llm"):
        resposta = llm.invoke(...)

Os dados persistem em `logs/reliability.jsonl` para análise posterior.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

# ── Logger estruturado ────────────────────────────────────────────────────────

_LOG_DIR = Path(__file__).parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_logger = logging.getLogger("reliability")

# Handler que grava em arquivo plain-text rotacionado diariamente
_file_handler = logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8")
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
))
_logger.addHandler(_file_handler)

# ── Estrutura de métricas por componente ──────────────────────────────────────

COMPONENTS = ("llm", "embeddings", "sql", "rag_pipeline", "keyword_extractor")

@dataclass
class ComponentStats:
    total_requests: int = 0
    total_failures: int = 0
    total_fallbacks: int = 0
    total_latency_ms: float = 0.0
    # Para MTBF: timestamps de falhas
    failure_timestamps: list[float] = field(default_factory=list)
    # Para MTTR: pares (falha_ts, recuperação_ts)
    recovery_pairs: list[tuple[float, float]] = field(default_factory=list)
    # Rastreia se o componente está em falha agora
    _in_failure: bool = field(default=False, repr=False)
    _failure_start: Optional[float] = field(default=None, repr=False)

    # ── métricas derivadas ────────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.total_failures) / self.total_requests

    @property
    def failure_rate(self) -> float:
        return 1.0 - self.success_rate

    @property
    def avg_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def mtbf_seconds(self) -> Optional[float]:
        """Mean Time Between Failures em segundos."""
        ts = self.failure_timestamps
        if len(ts) < 2:
            return None
        intervals = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        return sum(intervals) / len(intervals)

    @property
    def mttr_seconds(self) -> Optional[float]:
        """Mean Time To Recovery em segundos."""
        if not self.recovery_pairs:
            return None
        durations = [r - f for f, r in self.recovery_pairs]
        return sum(durations) / len(durations)

    @property
    def availability(self) -> float:
        """Uptime / (Uptime + Downtime) — baseado nos pares de recuperação."""
        total_downtime = sum(r - f for f, r in self.recovery_pairs)
        # Se ainda está em falha, conta o downtime em aberto
        if self._in_failure and self._failure_start:
            total_downtime += time.time() - self._failure_start
        if total_downtime == 0:
            return 1.0
        # Janela de observação: do primeiro request até agora
        window = time.time() - (self.failure_timestamps[0] if self.failure_timestamps else time.time())
        if window <= 0:
            return 1.0
        return max(0.0, 1.0 - (total_downtime / window))


# ── Estado global (thread-safe) ───────────────────────────────────────────────

_lock = threading.Lock()
_stats: dict[str, ComponentStats] = {c: ComponentStats() for c in COMPONENTS}
_session_start = time.time()


def _get_or_create(component: str) -> ComponentStats:
    if component not in _stats:
        _stats[component] = ComponentStats()
    return _stats[component]


# ── API pública ───────────────────────────────────────────────────────────────

@contextmanager
def record_request(component: str, context: str = "") -> Iterator[None]:
    """
    Context manager que mede latência e captura falhas de um componente.

    Uso:
        with record_request("llm", context="keyword_extractor"):
            resultado = llm.invoke(prompt)
    """
    t0 = time.perf_counter()
    try:
        yield
        elapsed = (time.perf_counter() - t0) * 1000
        with _lock:
            s = _get_or_create(component)
            s.total_requests += 1
            s.total_latency_ms += elapsed
            # Recuperação: se estava em falha, registra o par
            if s._in_failure and s._failure_start:
                s.recovery_pairs.append((s._failure_start, time.time()))
                s._in_failure = False
                s._failure_start = None
                _logger.info(
                    "[%s] Recuperado após %.1fs de indisponibilidade. %s",
                    component, s.recovery_pairs[-1][1] - s.recovery_pairs[-1][0],
                    f"[ctx={context}]" if context else "",
                )
        _log_event("success", component, elapsed_ms=elapsed, context=context)

    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        with _lock:
            s = _get_or_create(component)
            s.total_requests += 1
            s.total_failures += 1
            s.total_latency_ms += elapsed
            now = time.time()
            s.failure_timestamps.append(now)
            if not s._in_failure:
                s._in_failure = True
                s._failure_start = now
        _log_event("failure", component, elapsed_ms=elapsed, context=context, error=str(exc))
        _logger.error(
            "[%s] FALHA — %s %s",
            component, exc,
            f"[ctx={context}]" if context else "",
        )
        raise


def record_fallback(component: str, reason: str, fallback_to: str = "") -> None:
    """Registra que um fallback foi acionado por indisponibilidade do componente."""
    with _lock:
        s = _get_or_create(component)
        s.total_fallbacks += 1
    msg = f"[{component}] FALLBACK acionado — {reason}"
    if fallback_to:
        msg += f" → usando {fallback_to}"
    _logger.warning(msg)
    _log_event("fallback", component, reason=reason, fallback_to=fallback_to)


def record_info(component: str, message: str) -> None:
    """Log informativo associado a um componente."""
    _logger.info("[%s] %s", component, message)
    _log_event("info", component, message=message)


def _log_event(event_type: str, component: str, **kwargs) -> None:
    """Grava evento em JSONL para análise posterior."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "component": component,
        **{k: v for k, v in kwargs.items() if v is not None and v != ""},
    }
    try:
        with open(_LOG_DIR / "reliability.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass  # nunca deixa o log derrubar a aplicação


# ── Relatório ─────────────────────────────────────────────────────────────────

def get_report() -> dict:
    """Retorna snapshot das métricas de todos os componentes."""
    uptime = time.time() - _session_start
    with _lock:
        components = {}
        for name, s in _stats.items():
            if s.total_requests == 0:
                continue
            components[name] = {
                "requests":       s.total_requests,
                "failures":       s.total_failures,
                "fallbacks":      s.total_fallbacks,
                "success_rate":   f"{s.success_rate:.1%}",
                "failure_rate":   f"{s.failure_rate:.1%}",
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "availability":   f"{s.availability:.2%}",
                "mtbf_s":         round(s.mtbf_seconds, 1) if s.mtbf_seconds else "N/A",
                "mttr_s":         round(s.mttr_seconds, 1) if s.mttr_seconds else "N/A",
            }
    return {
        "session_uptime_s": round(uptime, 1),
        "session_start":    datetime.fromtimestamp(_session_start).strftime("%Y-%m-%d %H:%M:%S"),
        "components":       components,
    }


def print_report() -> None:
    r = get_report()
    print(f"\n{'='*62}")
    print(f"  RELIABILITY REPORT — uptime {r['session_uptime_s']}s desde {r['session_start']}")
    print(f"{'='*62}")
    for name, m in r["components"].items():
        print(f"\n  [{name}]")
        print(f"    Requests    : {m['requests']}  |  Falhas: {m['failures']}  |  Fallbacks: {m['fallbacks']}")
        print(f"    Taxa sucesso: {m['success_rate']}  |  Latência média: {m['avg_latency_ms']} ms")
        print(f"    Disponib.   : {m['availability']}  |  MTBF: {m['mtbf_s']}s  |  MTTR: {m['mttr_s']}s")
    print()
