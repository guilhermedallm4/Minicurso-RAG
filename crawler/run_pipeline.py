"""
Pipeline Completo: Crawl → Ingestão Segmentada
================================================
Executa crawl_ufpel.py e ingest_ufpel.py em sequência.
Cada tipo de página é inserido em sua própria coleção no pgvector.

Coleções geradas:
  ufpel_cursos / ufpel_disciplinas / ufpel_projetos
  ufpel_servidores / ufpel_unidades / ufpel_gestao / ufpel_sobre

Uso:
    # Tudo de uma vez (recomendado — chunking semântico)
    python run_pipeline.py --all-types --reset

    # Seções individuais
    python run_pipeline.py --cursos-only --reset
    python run_pipeline.py --servidores-only --servidores-max 200

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
from pathlib import Path

from crawl_ufpel  import UFPelCrawler, save_json
from ingest_ufpel import (
    pages_to_documents, ingest_segmented,
    ensure_dados_completos_table, store_dados_completos,
)

DEFAULT_JSON_OUTPUT = "dados_ufpel.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline completo: crawl UFPel + ingestão segmentada por coleção",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g_crawl = p.add_argument_group("Seleção de seções para crawl")
    g_crawl.add_argument("--all-types",         action="store_true",
                         help="Crawlea todas as seções em sequência.")
    g_crawl.add_argument("--cursos-only",        action="store_true")
    g_crawl.add_argument("--disciplinas-only",   action="store_true")
    g_crawl.add_argument("--projetos-only",      action="store_true")
    g_crawl.add_argument("--servidores-only",    action="store_true")
    g_crawl.add_argument("--unidades-only",      action="store_true")
    g_crawl.add_argument("--gestao-only",        action="store_true")
    g_crawl.add_argument("--sobre-only",         action="store_true")

    g_lim = p.add_argument_group("Limites por seção (padrão: sem limite)")
    g_lim.add_argument("--cursos-max",           type=int, default=None, metavar="N")
    g_lim.add_argument("--disciplinas-max",      type=int, default=None, metavar="N")
    g_lim.add_argument("--projetos-max",         type=int, default=None, metavar="N")
    g_lim.add_argument("--servidores-max",       type=int, default=None, metavar="N")
    g_lim.add_argument("--unidades-max",         type=int, default=None, metavar="N")

    g_perf = p.add_argument_group("Performance do crawl")
    g_perf.add_argument("--concurrency",         type=int, default=5, metavar="N",
                        help="Requisições simultâneas por seção")
    g_perf.add_argument("--crawl-delay",         type=float, default=1.0, metavar="SECS",
                        help="Delay (s) entre requests por worker")

    g_ingest = p.add_argument_group("Ingestão")
    g_ingest.add_argument("--output",            default=DEFAULT_JSON_OUTPUT, metavar="FILE",
                          help="Arquivo JSON onde os dados crawleados serão salvos")
    g_ingest.add_argument("--from-json",         default=None, metavar="FILE",
                          help="Pula o crawl e ingesta a partir de um JSON existente")
    g_ingest.add_argument("--reset",             action="store_true",
                          help="Recria cada coleção no banco antes de inserir")
    g_ingest.add_argument("--max-por-tipo",      type=int, default=None, metavar="N",
                          help="Limita N documentos por tipo na ingestão (útil no free tier)")
    g_ingest.add_argument("--delay",             type=float, default=0.7, metavar="SECS",
                          help="Segundos entre lotes de embedding")
    g_ingest.add_argument("--chunking",          choices=["semantico", "recursivo"],
                          default="semantico",
                          help="Estratégia de chunking (padrão: semantico)")
    return p


async def _run_crawl(args: argparse.Namespace) -> list[dict]:
    """Executa o crawl assíncrono e retorna a lista de documentos."""
    crawler = UFPelCrawler(concurrency=args.concurrency, delay=args.crawl_delay)

    any_specific = any([
        args.all_types, args.cursos_only, args.disciplinas_only,
        args.projetos_only, args.servidores_only, args.unidades_only,
        args.gestao_only, args.sobre_only,
    ])

    if not any_specific or args.all_types:
        return await crawler.crawl_all(
            cursos_max=args.cursos_max,
            disciplinas_max=args.disciplinas_max,
            projetos_max=args.projetos_max,
            servidores_max=args.servidores_max,
            unidades_max=args.unidades_max,
        )

    pages: list[dict] = []
    if args.cursos_only:
        pages += await crawler.crawl_cursos(args.cursos_max)
    if args.disciplinas_only:
        pages += await crawler.crawl_disciplinas(args.disciplinas_max)
    if args.projetos_only:
        pages += await crawler.crawl_projetos(args.projetos_max)
    if args.servidores_only:
        pages += await crawler.crawl_servidores(args.servidores_max)
    if args.unidades_only:
        pages += await crawler.crawl_unidades(args.unidades_max)
    if args.gestao_only:
        pages += await crawler.crawl_gestao()
    if args.sobre_only:
        pages += await crawler.crawl_sobre()
    return pages


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
        active = []
        if args.all_types or args.cursos_only:      active.append("Cursos")
        if args.all_types or args.disciplinas_only: active.append("Disciplinas")
        if args.all_types or args.projetos_only:    active.append("Projetos")
        if args.all_types or args.servidores_only:  active.append("Servidores")
        if args.all_types or args.unidades_only:    active.append("Unidades")
        if args.all_types or args.gestao_only:      active.append("Gestao")
        if args.all_types or args.sobre_only:       active.append("Sobre")
        label = " + ".join(active) if active else "Todas as seções"
        print(f"  PASSO 1/2 — Crawling: {label}")
        print("=" * 62)

        pages = asyncio.run(_run_crawl(args))
        save_json(pages, args.output)

    # ── Passo 2: Ingestão segmentada ─────────────────────────────────────────
    print()
    print("=" * 62)
    print("  PASSO 2/2 — Ingestão Segmentada no pgvector")
    print("=" * 62)

    # Persiste dados_completos para contexto rico no LLM
    ensure_dados_completos_table()
    n_stored = store_dados_completos(pages)
    if n_stored:
        print(f"[Pipeline] {n_stored} dados_completos armazenados.")

    documents = pages_to_documents(pages)
    if not documents:
        print("[Erro] Nenhum documento válido para ingerir.")
        sys.exit(1)

    use_semantic = args.chunking == "semantico"
    summary = ingest_segmented(
        documents,
        reset=args.reset,
        delay=args.delay,
        max_per_tipo=args.max_por_tipo,
        use_semantic_chunking=use_semantic,
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
