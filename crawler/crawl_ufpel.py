"""
Crawler do Portal Institucional UFPel
=====================================
Percorre https://institucional.ufpel.edu.br/ em largura (BFS),
extrai o texto de cada página e salva o resultado em JSON.

Uso:
    # Crawl com parâmetros padrão (150 páginas, profundidade 3)
    python crawl_ufpel.py

    # Limitar número de páginas
    python crawl_ufpel.py --max-pages 300

    # Limitar profundidade de navegação
    python crawl_ufpel.py --depth 4

    # Alterar arquivo de saída
    python crawl_ufpel.py --output dados_ufpel.json

    # Todos os parâmetros juntos
    python crawl_ufpel.py --max-pages 500 --depth 5 --output ufpel_completo.json

Requisitos:
    pip install -r requirements_crawler.txt
"""

import argparse
import json
import re
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL          = "https://institucional.ufpel.edu.br/"
ALLOWED_DOMAINS   = {"institucional.ufpel.edu.br"}

DEFAULT_OUTPUT      = "crawled_data.json"
DEFAULT_MAX_PAGES   = 150
DEFAULT_DEPTH       = 3
PROJECTS_URL        = "https://institucional.ufpel.edu.br/projetos"
DISCIPLINES_URL     = "https://institucional.ufpel.edu.br/disciplinas"
SERVIDORES_URL      = "https://institucional.ufpel.edu.br/servidores"
UNIDADES_URL        = "https://institucional.ufpel.edu.br/unidades"
CURSOS_URL          = "https://institucional.ufpel.edu.br/cursos"
REQUEST_DELAY     = 1.0   # segundos entre requisições (polite crawling)
MIN_TEXT_LENGTH   = 80    # mínimo de chars para salvar uma página

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Extensões que não são páginas HTML
_SKIP_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".zip", ".rar", ".png", ".jpg", ".jpeg",
    ".gif", ".svg", ".mp4", ".mp3", ".ico",
}

# Tags HTML que não contêm conteúdo informacional
_STRIP_TAGS = {
    "script", "style", "nav", "header", "footer",
    "aside", "form", "button", "noscript", "iframe",
}


# =============================================================================
# Utilitários de URL
# =============================================================================

def _normalize_url(url: str) -> str:
    """Remove fragmentos e query string para evitar duplicatas."""
    p = urlparse(url)
    return p._replace(fragment="", query="").geturl()


def _is_crawlable(url: str) -> bool:
    p = urlparse(url)
    if p.scheme not in ("http", "https"):
        return False
    if p.netloc not in ALLOWED_DOMAINS:
        return False
    path_lower = p.path.lower()
    if any(path_lower.endswith(ext) for ext in _SKIP_EXTENSIONS):
        return False
    return True


# =============================================================================
# Extração de conteúdo HTML
# =============================================================================

def _extract_links(soup: BeautifulSoup, page_url: str) -> set[str]:
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = _normalize_url(urljoin(page_url, href))
        if _is_crawlable(full):
            links.add(full)
    return links


def _extract_title(soup: BeautifulSoup) -> str:
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    if soup.title:
        return soup.title.get_text(strip=True)
    h1 = soup.find("h1")
    return h1.get_text(strip=True) if h1 else "Sem título"


def _extract_equipe_members(soup: BeautifulSoup) -> str:
    """
    Extrai membros da equipe de um projeto.

    Busca por:
      1. Seção com ID/classe contendo "equipe" ou "membros"
      2. Tabelas com colunas de nome/papel
      3. Listas não-ordenadas com membros
    """
    members = []

    # Procura por seção de equipe/membros
    equipe_section = soup.find(
        re.compile(r"div|section"),
        {"id": re.compile(r"equipe|membros|team", re.I)}
    )
    if not equipe_section:
        equipe_section = soup.find(
            re.compile(r"div|section"),
            {"class": re.compile(r"equipe|membros|team", re.I)}
        )

    if equipe_section:
        # Tenta extrair de tabelas
        table = equipe_section.find("table")
        if table:
            for row in table.find_all("tr")[1:]:  # Skip header
                cells = row.find_all(["td", "th"])
                if len(cells) >= 2:
                    name = cells[0].get_text(strip=True)
                    role = cells[1].get_text(strip=True) if len(cells) > 1 else ""
                    if name:
                        members.append(f"{name}" + (f" ({role})" if role else ""))

        # Tenta extrair de listas
        ul = equipe_section.find("ul")
        if ul:
            for li in ul.find_all("li"):
                text = li.get_text(strip=True)
                if text:
                    members.append(text)

        # Tenta extrair de parágrafos estruturados
        if not members:
            for p in equipe_section.find_all("p"):
                text = p.get_text(strip=True)
                if text and len(text) > 10:
                    members.append(text)

    if members:
        return "Equipe:\n" + "\n".join(f"- {m}" for m in members)
    return ""


def _extract_ficha_fields(soup: BeautifulSoup) -> str:
    """
    Extrai campos estruturados de páginas com o padrão ficha-label/ficha-campo
    (projetos, disciplinas, servidores) mais as seções de acordeão em
    #informacoes (Ementa, Objetivos, Conteúdo Programático, etc.).

    Gera texto no formato 'Label: Valor' para maximizar a qualidade da
    busca semântica — projetos têm o Resumo em ficha-dados; disciplinas
    têm Ementa e Objetivos nas seções de acordeão.
    """
    ficha = soup.find(class_="ficha-dados")
    if ficha is None:
        return ""

    lines = []

    # ── Campos da ficha principal (ficha-label / ficha-campo) ─────────────────
    for label_el in ficha.find_all(class_="ficha-label"):
        label = label_el.get_text(strip=True)
        campo = label_el.find_next_sibling(class_="ficha-campo")
        if campo is None:
            continue
        value = campo.get_text(separator=" ", strip=True)
        value = re.sub(r"\s{2,}", " ", value)
        if value:
            lines.append(f"{label}: {value}")

    # ── Seções de acordeão em #informacoes (Ementa, Objetivos, etc.) ──────────
    informacoes = soup.find(id="informacoes")
    if informacoes:
        for accordion in informacoes.find_all(class_="accordion"):
            heading = accordion.find("h3")
            content = accordion.find(class_="accordion-content")
            if not heading or not content:
                continue
            label = heading.get_text(strip=True)
            value = content.get_text(separator=" ", strip=True)
            value = re.sub(r"\s{2,}", " ", value)
            if value:
                lines.append(f"{label}: {value}")

    # ── Extrai equipe de projetos ──────────────────────────────────────────────
    equipe_text = _extract_equipe_members(soup)
    if equipe_text:
        lines.append(equipe_text)

    return "\n".join(lines)


def _extract_text(soup: BeautifulSoup) -> str:
    """
    Remove tags decorativas e retorna texto limpo.

    Estratégia (em ordem de preferência):
      1. ficha-dados  — páginas de projeto/disciplina: extrai campos estruturados
                        incluindo o Resumo completo
      2. <main>
      3. <article>    — só se tiver texto suficiente (> 150 chars)
      4. div/section com id/class contendo 'content', 'main' ou 'conteudo'
      5. <body> completo
    """
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    # Páginas de ficha (projetos, disciplinas) têm estrutura própria
    ficha_text = _extract_ficha_fields(soup)
    if ficha_text:
        return ficha_text

    candidates = [
        soup.find("main"),
        soup.find("div", id=re.compile(r"conteudo|content|main", re.I)),
        soup.find("div", class_=re.compile(r"conteudo|content|main|wrapper", re.I)),
        soup.find("section", class_=re.compile(r"conteudo|content|main", re.I)),
    ]

    # <article> entra como candidato apenas se for substancial
    article = soup.find("article")
    if article:
        article_text = article.get_text(strip=True)
        if len(article_text) > 150:
            candidates.insert(0, article)

    # Usa o primeiro candidato não-None
    element = next((c for c in candidates if c is not None), None)
    if element is None:
        element = soup.find("body")
    if element is None:
        return ""

    text = element.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# =============================================================================
# Crawler BFS
# =============================================================================

def get_project_urls(session: requests.Session | None = None) -> list[str]:
    """
    Busca a página de listagem de projetos e retorna todas as URLs
    de projetos individuais (/projetos/id/XXXX) encontradas.
    """
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", HEADERS["User-Agent"])
    resp = sess.get(PROJECTS_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/projetos/id/" in href:
            full = _normalize_url(urljoin(PROJECTS_URL, href))
            if _is_crawlable(full):
                urls.add(full)
    return sorted(urls)


def get_discipline_urls(session: requests.Session | None = None) -> list[str]:
    """
    Busca a página de listagem de disciplinas e retorna todas as URLs
    de disciplinas individuais (/disciplinas/id/XXXX) encontradas.
    """
    sess = session or requests.Session()
    sess.headers.setdefault("User-Agent", HEADERS["User-Agent"])
    resp = sess.get(DISCIPLINES_URL, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "/disciplinas/id/" in href:
            full = _normalize_url(urljoin(DISCIPLINES_URL, href))
            if _is_crawlable(full):
                urls.add(full)
    return sorted(urls)


class UFPelCrawler:
    """
    Crawler BFS para o portal institucional da UFPel.

    Regra central: links são sempre enfileirados a partir de qualquer
    página HTML válida, mesmo que o conteúdo textual seja insuficiente
    para ser salvo. Isso garante que páginas-hub (como a homepage) não
    bloqueiem a descoberta das páginas filhas.
    """

    def __init__(
        self,
        max_pages: int = DEFAULT_MAX_PAGES,
        max_depth: int = DEFAULT_DEPTH,
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self._visited: set[str] = set()
        self._pages: list[dict] = []
        self._session = requests.Session()
        self._session.headers.update(HEADERS)

    @property
    def pages(self) -> list[dict]:
        return self._pages

    def crawl(self, start_url: str = BASE_URL) -> list[dict]:
        """Executa o crawl BFS a partir de start_url."""
        queue: deque[tuple[str, int]] = deque([
            (_normalize_url(start_url), 0)
        ])

        print(f"[Crawler] URL inicial : {start_url}")
        print(f"[Crawler] Max páginas : {self.max_pages}")
        print(f"[Crawler] Max depth   : {self.max_depth}")
        print("-" * 60)

        while queue and len(self._pages) < self.max_pages:
            url, depth = queue.popleft()

            if url in self._visited or depth > self.max_depth:
                continue
            self._visited.add(url)

            result = self._fetch(url, depth)
            if result is None:
                # Erro de rede — não há links para extrair
                continue

            text, links, title = result

            # Sempre enfileira links, independente do tamanho do texto.
            # Isso permite que páginas-hub (homepage, índices) alimentem
            # a fila mesmo sem conteúdo textual próprio significativo.
            if depth < self.max_depth:
                for link in links:
                    if link not in self._visited:
                        queue.append((link, depth + 1))

            # Salva apenas páginas com conteúdo suficiente
            if len(text) >= MIN_TEXT_LENGTH:
                self._pages.append({
                    "url":        url,
                    "title":      title,
                    "text":       text,
                    "depth":      depth,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "links":      list(links),
                })
                count = len(self._pages)
                print(f"  [{count:>4}/{self.max_pages}] depth={depth}  {url[:80]}")
            else:
                print(f"  [skip] depth={depth}  {url[:80]}  ({len(text)} chars)")

            time.sleep(REQUEST_DELAY)

        print(f"\n[Crawler] Finalizado: {len(self._pages)} páginas coletadas.")
        return self._pages

    def crawl_projects(self, limit: int | None = None) -> list[dict]:
        """
        Modo dedicado para projetos: busca todas as URLs da listagem de
        projetos e crawlea cada página individual em ordem, sem limite de
        profundidade. Produz texto estruturado com Resumo para busca semântica.

        O texto de cada projeto começa com 'Tipo: Projeto' para que consultas
        como "projetos de IA" encontrem apenas projetos, não disciplinas.

        Args:
            limit: máximo de projetos a coletar (padrão: self.max_pages).
        """
        print(f"[Projetos] Buscando URLs em {PROJECTS_URL} ...")
        project_urls = get_project_urls(self._session)
        total = len(project_urls)
        effective_limit = min(total, limit if limit is not None else self.max_pages)
        print(f"[Projetos] {total} projetos encontrados — serão coletados até {effective_limit}")
        print("-" * 60)

        for url in project_urls[:effective_limit]:
            if url in self._visited:
                continue
            self._visited.add(url)

            result = self._fetch(url, depth=0)
            if result is None:
                continue

            text, _, title = result
            if len(text) >= MIN_TEXT_LENGTH:
                self._pages.append({
                    "url":        url,
                    "title":      title,
                    "text":       f"Tipo: Projeto\n{text}",
                    "tipo":       "projeto",
                    "depth":      0,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "links":      [],
                })
                count = len(self._pages)
                print(f"  [{count:>4}/{effective_limit}] {url}")
            else:
                print(f"  [skip] {url}  ({len(text)} chars)")

            time.sleep(REQUEST_DELAY)

        print(f"\n[Projetos] Finalizado: {len(self._pages)} projetos coletados.")
        return self._pages

    def crawl_disciplines(self, limit: int | None = None) -> list[dict]:
        """
        Modo dedicado para disciplinas: busca todas as URLs da listagem de
        disciplinas e crawlea cada página individual. Extrai Ementa, Objetivos
        e Conteúdo Programático de cada disciplina via ficha-dados + #informacoes.

        O texto de cada disciplina começa com 'Tipo: Disciplina' para que consultas
        como "disciplina de IA" encontrem apenas disciplinas, não projetos.

        Args:
            limit: máximo de disciplinas a coletar (padrão: todas disponíveis).
        """
        print(f"[Disciplinas] Buscando URLs em {DISCIPLINES_URL} ...")
        discipline_urls = get_discipline_urls(self._session)
        total = len(discipline_urls)
        effective_limit = total if limit is None else min(total, limit)
        print(f"[Disciplinas] {total} disciplinas encontradas — serão coletadas até {effective_limit}")
        print("-" * 60)

        for url in discipline_urls[:effective_limit]:
            if url in self._visited:
                continue
            self._visited.add(url)

            result = self._fetch(url, depth=0)
            if result is None:
                continue

            text, _, title = result
            if len(text) >= MIN_TEXT_LENGTH:
                self._pages.append({
                    "url":        url,
                    "title":      title,
                    "text":       f"Tipo: Disciplina\n{text}",
                    "tipo":       "disciplina",
                    "depth":      0,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "links":      [],
                })
                count = len(self._pages)
                print(f"  [{count:>4}/{effective_limit}] {url}")
            else:
                print(f"  [skip] {url}  ({len(text)} chars)")

            time.sleep(REQUEST_DELAY)

        print(f"\n[Disciplinas] Finalizado: {len(self._pages)} disciplinas coletadas.")
        return self._pages

    # ------------------------------------------------------------------
    # Helpers genéricos para novas seções do portal
    # ------------------------------------------------------------------

    def _get_section_urls(
        self,
        listing_url: str,
        id_pattern: str,
    ) -> list[str]:
        """
        Busca a página de listagem e retorna URLs que contêm `id_pattern`.
        Exemplo: id_pattern="/servidores/id/" captura /servidores/id/12345.
        """
        try:
            resp = self._session.get(listing_url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")
            urls: set[str] = set()
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if id_pattern in href:
                    full = _normalize_url(urljoin(listing_url, href))
                    if _is_crawlable(full):
                        urls.add(full)
            return sorted(urls)
        except requests.RequestException as err:
            print(f"  [ERRO] {listing_url} → {err}")
            return []

    def _crawl_section(
        self,
        listing_url: str,
        id_pattern: str,
        tipo: str,
        tipo_label: str,
        limit: int | None = None,
    ) -> list[dict]:
        """
        Crawl genérico para qualquer seção do portal.

        Args:
            listing_url : URL da página de listagem (ex: /servidores)
            id_pattern  : fragmento de URL que identifica páginas individuais
            tipo        : valor do campo 'tipo' no JSON (ex: 'servidor')
            tipo_label  : nome legível para logs (ex: 'Servidor')
            limit       : máximo de itens a coletar (None = todos)
        """
        print(f"[{tipo_label}] Buscando URLs em {listing_url} ...")
        urls = self._get_section_urls(listing_url, id_pattern)
        total = len(urls)
        effective_limit = total if limit is None else min(total, limit)
        print(f"[{tipo_label}] {total} encontrados — coletando até {effective_limit}")
        print("-" * 60)

        collected = 0
        for url in urls[:effective_limit]:
            if url in self._visited:
                continue
            self._visited.add(url)

            result = self._fetch(url, depth=0)
            if result is None:
                continue

            text, _, title = result
            if len(text) >= MIN_TEXT_LENGTH:
                self._pages.append({
                    "url":        url,
                    "title":      title,
                    "text":       f"Tipo: {tipo_label}\n{text}",
                    "tipo":       tipo,
                    "depth":      0,
                    "crawled_at": datetime.now(timezone.utc).isoformat(),
                    "links":      [],
                })
                collected += 1
                print(f"  [{collected:>4}/{effective_limit}] {url}")
            else:
                print(f"  [skip] {url}  ({len(text)} chars)")

            time.sleep(REQUEST_DELAY)

        print(f"\n[{tipo_label}] Finalizado: {collected} coletados.")
        return self._pages

    def crawl_servidores(self, limit: int | None = None) -> list[dict]:
        """Coleta páginas individuais de servidores/professores."""
        return self._crawl_section(
            listing_url=SERVIDORES_URL,
            id_pattern="/servidores/id/",
            tipo="servidor",
            tipo_label="Servidor",
            limit=limit,
        )

    def crawl_unidades(self, limit: int | None = None) -> list[dict]:
        """Coleta páginas individuais de unidades/centros/departamentos."""
        return self._crawl_section(
            listing_url=UNIDADES_URL,
            id_pattern="/unidades/id/",
            tipo="unidade",
            tipo_label="Unidade",
            limit=limit,
        )

    def crawl_cursos(self, limit: int | None = None) -> list[dict]:
        """Coleta páginas individuais de cursos de graduação e pós-graduação."""
        return self._crawl_section(
            listing_url=CURSOS_URL,
            id_pattern="/cursos/id/",
            tipo="curso",
            tipo_label="Curso",
            limit=limit,
        )

    def _fetch(self, url: str, depth: int) -> tuple[str, set, str] | None:
        """
        Faz GET na URL e retorna (texto, links, título).
        Retorna None apenas em caso de erro de rede/HTTP.
        """
        try:
            resp = self._session.get(url, timeout=15, allow_redirects=True)
            resp.raise_for_status()

            if "text/html" not in resp.headers.get("Content-Type", ""):
                return None

            soup = BeautifulSoup(resp.text, "lxml")
            links = _extract_links(soup, url)
            text  = _extract_text(soup)
            title = _extract_title(soup)
            return text, links, title

        except requests.RequestException as err:
            print(f"  [ERRO] {url} → {err}")
            return None


# =============================================================================
# Persistência
# =============================================================================

def save_json(pages: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(pages, fh, ensure_ascii=False, indent=2)
    print(f"[Saída] {len(pages)} páginas salvas em '{output_path}'")


# =============================================================================
# Entrypoint CLI
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawler do Portal Institucional UFPel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--max-pages",          type=int, default=DEFAULT_MAX_PAGES,
                   metavar="N", help="Máximo de páginas no BFS geral")
    p.add_argument("--projects-max",       type=int, default=500,
                   metavar="N", help="Máximo de projetos a coletar (--projects-only)")
    p.add_argument("--disciplines-max",    type=int, default=None,
                   metavar="N", help="Máximo de disciplinas a coletar (padrão: todas)")
    p.add_argument("--depth",              type=int, default=DEFAULT_DEPTH,
                   metavar="D", help="Profundidade máxima de navegação (BFS geral)")
    p.add_argument("--output",             default=DEFAULT_OUTPUT,
                   metavar="FILE", help="Arquivo JSON de saída")
    p.add_argument("--url",                default=BASE_URL,
                   metavar="URL", help="URL de partida para o BFS geral")
    p.add_argument("--projects-only",      action="store_true",
                   help="Crawlea projetos (/projetos/id/). Limitado por --projects-max.")
    p.add_argument("--disciplines-only",   action="store_true",
                   help="Crawlea disciplinas (/disciplinas/id/). Padrão: todas.")
    p.add_argument("--servidores-only",    action="store_true",
                   help="Crawlea servidores/professores (/servidores/id/).")
    p.add_argument("--servidores-max",     type=int, default=None, metavar="N",
                   help="Máximo de servidores a coletar (padrão: todos)")
    p.add_argument("--unidades-only",      action="store_true",
                   help="Crawlea unidades/centros/departamentos (/unidades/id/).")
    p.add_argument("--unidades-max",       type=int, default=None, metavar="N",
                   help="Máximo de unidades a coletar (padrão: todas)")
    p.add_argument("--cursos-only",        action="store_true",
                   help="Crawlea cursos de graduação e pós-graduação (/cursos/id/).")
    p.add_argument("--cursos-max",         type=int, default=None, metavar="N",
                   help="Máximo de cursos a coletar (padrão: todos)")
    p.add_argument("--all-types",          action="store_true",
                   help="Crawlea todos os tipos: projetos, disciplinas, servidores, "
                        "unidades e cursos em sequência.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    crawler = UFPelCrawler(max_pages=args.max_pages, max_depth=args.depth)

    pages: list[dict] = []

    # --all-types ativa todos os modos específicos de uma vez
    if args.all_types:
        pages  = crawler.crawl_projects(limit=args.projects_max)
        pages += crawler.crawl_disciplines(limit=args.disciplines_max)
        pages += crawler.crawl_servidores(limit=args.servidores_max)
        pages += crawler.crawl_unidades(limit=args.unidades_max)
        pages += crawler.crawl_cursos(limit=args.cursos_max)
    elif any([args.projects_only, args.disciplines_only,
              args.servidores_only, args.unidades_only, args.cursos_only]):
        if args.projects_only:
            pages += crawler.crawl_projects(limit=args.projects_max)
        if args.disciplines_only:
            pages += crawler.crawl_disciplines(limit=args.disciplines_max)
        if args.servidores_only:
            pages += crawler.crawl_servidores(limit=args.servidores_max)
        if args.unidades_only:
            pages += crawler.crawl_unidades(limit=args.unidades_max)
        if args.cursos_only:
            pages += crawler.crawl_cursos(limit=args.cursos_max)
    else:
        pages = crawler.crawl(start_url=args.url)

    save_json(pages, args.output)


if __name__ == "__main__":
    main()
