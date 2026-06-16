"""
Pipeline Completo: Crawl → Ingestão Segmentada
================================================
Executa crawl_ufpel.py e ingest_ufpel.py em sequência.
Cada tipo de página (disciplina, projeto, servidor, unidade, curso)
é inserido em sua própria coleção no pgvector.

Uso:
    # Tudo de uma vez (recomendado para produção)
    python run_pipeline.py --all-types --reset

    # Só disciplinas e projetos
    python run_pipeline.py --disciplines-only --projects-only --reset

    # Só servidores (até 200)
    python run_pipeline.py --servidores-only --servidores-max 200 --reset

    # Crawl geral BFS (150 páginas, profundidade 3)
    python run_pipeline.py

    # Re-ingerir a partir de JSON já existente (sem re-crawl)
    python run_pipeline.py --from-json dados_ufpel.json --reset
"""

import argparse
import json
import sys
from pathlib import Path

from crawl_ufpel  import UFPelCrawler, BASE_URL, save_json
from ingest_ufpel import pages_to_documents, ingest_segmented

DEFAULT_JSON_OUTPUT = "dados_ufpel.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pipeline completo: crawl UFPel + ingestão segmentada por coleção",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── Modos de crawl ───────────────────────────────────────────────────────
    p.add_argument("--all-types",         action="store_true",
                   help="Crawlea todos os tipos em sequência.")
    p.add_argument("--projects-only",     action="store_true",
                   help="Crawlea projetos (/projetos/id/).")
    p.add_argument("--disciplines-only",  action="store_true",
                   help="Crawlea disciplinas (/disciplinas/id/).")
    p.add_argument("--servidores-only",   action="store_true",
                   help="Crawlea servidores/professores (/servidores/id/).")
    p.add_argument("--unidades-only",     action="store_true",
                   help="Crawlea unidades/centros (/unidades/id/).")
    p.add_argument("--cursos-only",       action="store_true",
                   help="Crawlea cursos (/cursos/id/).")
    # ── Limites por tipo ─────────────────────────────────────────────────────
    p.add_argument("--projects-max",      type=int, default=500,  metavar="N")
    p.add_argument("--disciplines-max",   type=int, default=None, metavar="N")
    p.add_argument("--servidores-max",    type=int, default=None, metavar="N")
    p.add_argument("--unidades-max",      type=int, default=None, metavar="N")
    p.add_argument("--cursos-max",        type=int, default=None, metavar="N")
    p.add_argument("--max-pages",         type=int, default=150,  metavar="N",
                   help="Máximo de páginas no crawl BFS geral")
    p.add_argument("--depth",             type=int, default=3,    metavar="D",
                   help="Profundidade máxima de navegação (BFS geral)")
    # ── Saída e banco ────────────────────────────────────────────────────────
    p.add_argument("--output",            default=DEFAULT_JSON_OUTPUT, metavar="FILE",
                   help="Arquivo JSON onde os dados crawleados serão salvos")
    p.add_argument("--from-json",         default=None, metavar="FILE",
                   help="Pula o crawl e ingesta diretamente a partir de um JSON existente")
    p.add_argument("--reset",             action="store_true",
                   help="Recria cada coleção no banco antes de inserir")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # ── Passo 1: Crawl (ou leitura de JSON existente) ─────────────────────────
    if args.from_json:
        print("=" * 62)
        print(f"  PASSO 1/2 — Carregando JSON: {args.from_json}")
        print("=" * 62)
        with open(args.from_json, encoding="utf-8") as fh:
            pages: list[dict] = json.load(fh)
        print(f"[Pipeline] {len(pages)} páginas carregadas do JSON.")
    else:
        active_modes = []
        if args.all_types or args.projects_only:
            active_modes.append(f"Projetos (max {args.projects_max})")
        if args.all_types or args.disciplines_only:
            active_modes.append("Disciplinas")
        if args.all_types or args.servidores_only:
            active_modes.append("Servidores")
        if args.all_types or args.unidades_only:
            active_modes.append("Unidades")
        if args.all_types or args.cursos_only:
            active_modes.append("Cursos")
        label = " + ".join(active_modes) if active_modes else "Portal Institucional UFPel (BFS)"

        print("=" * 62)
        print(f"  PASSO 1/2 — Crawling: {label}")
        print("=" * 62)

        crawler = UFPelCrawler(max_pages=args.max_pages, max_depth=args.depth)
        pages = []

        any_specific = any([
            args.all_types, args.projects_only, args.disciplines_only,
            args.servidores_only, args.unidades_only, args.cursos_only,
        ])

        if any_specific:
            if args.all_types or args.projects_only:
                pages += crawler.crawl_projects(limit=args.projects_max)
            if args.all_types or args.disciplines_only:
                pages += crawler.crawl_disciplines(limit=args.disciplines_max)
            if args.all_types or args.servidores_only:
                pages += crawler.crawl_servidores(limit=args.servidores_max)
            if args.all_types or args.unidades_only:
                pages += crawler.crawl_unidades(limit=args.unidades_max)
            if args.all_types or args.cursos_only:
                pages += crawler.crawl_cursos(limit=args.cursos_max)
        else:
            pages = crawler.crawl(start_url=BASE_URL)

        save_json(pages, args.output)

    # ── Passo 2: Ingestão segmentada ─────────────────────────────────────────
    print()
    print("=" * 62)
    print("  PASSO 2/2 — Ingestão Segmentada no pgvector")
    print("=" * 62)

    documents = pages_to_documents(pages)
    if not documents:
        print("[Erro] Nenhum documento válido para ingerir.")
        sys.exit(1)

    summary = ingest_segmented(documents, reset=args.reset)

    # ── Resumo ────────────────────────────────────────────────────────────────
    tipos: dict[str, int] = {}
    for pg in pages:
        t = pg.get("tipo", "portal_geral")
        tipos[t] = tipos.get(t, 0) + 1

    print()
    print("=" * 62)
    print("  Pipeline concluído com sucesso!")
    print(f"  Páginas crawleadas : {len(pages)}")
    for tipo, qtd in sorted(tipos.items()):
        print(f"    ↳ {tipo:<20}: {qtd}")
    print(f"  Documentos válidos : {len(documents)}")
    if not args.from_json:
        print(f"  JSON salvo em      : {args.output}")
    print()
    print("  Coleções criadas/atualizadas:")
    for col, qtd in sorted(summary.items()):
        print(f"    ↳ {col:<30}: {qtd} docs")
    print("=" * 62)


if __name__ == "__main__":
    main()
