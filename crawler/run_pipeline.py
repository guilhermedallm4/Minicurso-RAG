"""
Pipeline Completo: Crawl → Ingestão Segmentada
================================================
Executa o deep-crawl dos cursos de Computação (crawl_ufpel.py) e a
ingestão segmentada (ingest_ufpel.py) em sequência.

Cursos alvo do crawl:
  3900 — Ciência da Computação    (Graduação / Bacharelado)
  8102 — Computação               (Pós-Graduação / Doutorado)
  7057 — Computação               (Pós-Graduação / Mestrado Acadêmico)
  3910 — Engenharia de Computação (Graduação / Bacharelado)

Cada tipo de página é inserido em sua própria coleção no pgvector:
  curso      → ufpel_cursos        (dados do curso)
  disciplina → ufpel_disciplinas   (ementa, objetivos, conteúdo, turmas)
  servidor   → ufpel_servidores    (professores: currículo, projetos, aulas)
  projeto    → ufpel_projetos      (projetos ativos: equipe, financeiro)

O embedding usa o RESUMO de cada registro (embedding_text); os dados
estruturados completos vão para a tabela doc_completos (JSONB), permitindo
consultas SQL para montar o contexto completo do LLM.

Uso:
    # Pipeline completo (crawl + ingestão)
    python run_pipeline.py --reset

    # Apenas alguns cursos
    python run_pipeline.py --cursos 3900 3910 --reset

    # Re-ingerir a partir de JSON existente (sem re-crawl)
    python run_pipeline.py --from-json dados_ufpel.json --reset

    # Chunking recursivo (legado)
    python run_pipeline.py --from-json dados_ufpel.json --chunking recursivo --reset
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone

from crawl_ufpel  import TARGET_CURSOS, UFPelCrawler, save_json
from ingest_ufpel import (
    pages_to_documents, ingest_segmented,
    ensure_dados_completos_table, store_dados_completos,
    ensure_info_tables, store_info_tables,
)

DEFAULT_JSON_OUTPUT = "dados_ufpel.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline completo: deep-crawl dos cursos de Computação "
                    "da UFPel + ingestão segmentada por coleção",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_crawl = p.add_argument_group("Crawl")
    g_crawl.add_argument("--cursos", nargs="+", default=None, metavar="COD",
                         choices=list(TARGET_CURSOS.keys()),
                         help="Subconjunto dos códigos de curso alvo "
                              f"(padrão: todos — {' '.join(TARGET_CURSOS)})")
    g_crawl.add_argument("--concurrency",  type=int, default=5, metavar="N",
                         help="Requisições simultâneas")
    g_crawl.add_argument("--crawl-delay",  type=float, default=1.0, metavar="SECS",
                         help="Delay (s) entre requests por worker")

    g_ingest = p.add_argument_group("Ingestão")
    g_ingest.add_argument("--output",       default=DEFAULT_JSON_OUTPUT, metavar="FILE",
                          help="Arquivo JSON onde os dados crawleados serão salvos")
    g_ingest.add_argument("--from-json",    default=None, metavar="FILE",
                          help="Pula o crawl e ingesta a partir de um JSON existente")
    g_ingest.add_argument("--reset",        action="store_true",
                          help="Recria cada coleção no banco antes de inserir")
    g_ingest.add_argument("--max-por-tipo", type=int, default=None, metavar="N",
                          help="Limita N documentos por tipo na ingestão (útil no free tier)")
    g_ingest.add_argument("--delay",        type=float, default=0.7, metavar="SECS",
                          help="Segundos entre lotes de embedding")
    g_ingest.add_argument("--chunking",     choices=["documento", "semantico", "recursivo"],
                          default="documento",
                          help="Estratégia de chunking (padrão: documento — "
                               "1 doc = 1 chunk com o embedding_text inteiro)")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    t0   = datetime.now(timezone.utc)

    # ── Passo 1: Crawl (ou leitura de JSON existente) ────────────────────────
    print("=" * 62)
    if args.from_json:
        print(f"  PASSO 1/2 — Carregando JSON: {args.from_json}")
        print("=" * 62)
        with open(args.from_json, encoding="utf-8") as fh:
            pages: list[dict] = json.load(fh)
        print(f"[Pipeline] {len(pages)} documentos carregados do JSON.")
    else:
        alvo = args.cursos or list(TARGET_CURSOS.keys())
        print("  PASSO 1/2 — Deep-crawl dos cursos de Computação:")
        for cod in alvo:
            info = TARGET_CURSOS[cod]
            print(f"    ↳ {cod} — {info['nome']} ({info['nivel']} / {info['grau']})")
        print("=" * 62)

        crawler = UFPelCrawler(
            concurrency=args.concurrency,
            delay=args.crawl_delay,
            cursos=args.cursos,
        )
        pages = asyncio.run(crawler.crawl())
        save_json(pages, args.output)

    # ── Passo 2: Ingestão segmentada ─────────────────────────────────────────
    print()
    print("=" * 62)
    print("  PASSO 2/2 — Ingestão Segmentada no pgvector")
    print("=" * 62)

    # Persiste dados_completos (JSONB) para contexto rico no LLM via SQL
    ensure_dados_completos_table()
    n_stored = store_dados_completos(pages)
    if n_stored:
        print(f"[Pipeline] {n_stored} dados_completos armazenados.")

    # Persiste as tabelas estruturadas por tipo (*_info) para consultas SQL diretas
    ensure_info_tables()
    n_info = store_info_tables(pages)
    if n_info:
        print(f"[Pipeline] {n_info} registros estruturados armazenados nas tabelas *_info.")

    documents = pages_to_documents(pages)
    if not documents:
        print("[Erro] Nenhum documento válido para ingerir.")
        sys.exit(1)

    summary = ingest_segmented(
        documents,
        reset=args.reset,
        delay=args.delay,
        max_per_tipo=args.max_por_tipo,
        chunking=args.chunking,
    )

    # ── Resumo final ─────────────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    by_tipo: dict[str, int] = {}
    for pg in pages:
        t = pg.get("tipo", "portal_geral")
        by_tipo[t] = by_tipo.get(t, 0) + 1

    print()
    print("=" * 62)
    print("  Pipeline concluído com sucesso!")
    print(f"  Tempo total         : {elapsed:.1f}s")
    print(f"  Documentos crawleados: {len(pages)}")
    for tipo, qtd in sorted(by_tipo.items()):
        print(f"    ↳ {tipo:<20}: {qtd}")
    print(f"  Documentos válidos  : {len(documents)}")
    if not args.from_json:
        print(f"  JSON salvo em       : {args.output}")
    print()
    print("  Coleções criadas/atualizadas:")
    for col, qtd in sorted(summary.items()):
        print(f"    ↳ {col:<30}: {qtd} docs")
    print("=" * 62)


if __name__ == "__main__":
    main()
