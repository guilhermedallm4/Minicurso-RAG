"""
Crawler Async do Portal Institucional UFPel
===========================================
Extrai dados estruturados de todas as seções do portal:
  - cursos, disciplinas, projetos, servidores, unidades, gestao, sobre

Schema de saída (por documento):
  {
    "id":             <uuid>,
    "tipo":           <seção>,
    "titulo":         <nome/título do registro>,
    "embedding_text": <texto otimizado para embeddings>,
    "metadata":       { url, crawled_at, ...campos filtráveis },
    "dados_completos":{ todos os dados capturados }
  }

Campos vazios, nulos ou inexistentes são preenchidos com:
  "Não há informações disponíveis"

Uso:
    python crawl_ufpel.py --all-types --output dados_ufpel.json
    python crawl_ufpel.py --cursos-only --cursos-max 50
    python crawl_ufpel.py --projetos-only
    python crawl_ufpel.py --concurrency 5 --delay 1.0

Requisitos:
    pip install -r requirements_crawler.txt
"""

import argparse
import asyncio
import json
import logging
import random
import re
import unicodedata
import uuid
from datetime import datetime, date, timezone
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://institucional.ufpel.edu.br"

SECTION_URLS: dict[str, str] = {
    "cursos":      f"{BASE_URL}/cursos",
    "disciplinas": f"{BASE_URL}/disciplinas",
    "projetos":    f"{BASE_URL}/projetos",
    "servidores":  f"{BASE_URL}/servidores",
    "unidades":    f"{BASE_URL}/unidades",
    "gestao":      f"{BASE_URL}/gestao",
    "sobre":       f"{BASE_URL}/sobre",
}

URL_PATTERNS: dict[str, str] = {
    "cursos":      "/cursos/cod/",      # portal usa /cod/ não /id/ para cursos
    "disciplinas": "/disciplinas/id/",
    "projetos":    "/projetos/id/",
    "servidores":  "/servidores/id/",
    "unidades":    "/unidades/id/",
}

EMPTY = "Não há informações disponíveis"

HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Accept":          "text/html,application/xhtml+xml",
}

SKIP_HTTP_ERRORS: frozenset[int] = frozenset({400, 401, 403, 404, 500, 502, 503})

MAX_RETRY           = 4
RETRY_BASE          = 2.0      # backoff exponencial: 2, 4, 8, 16 s
DEFAULT_CONCURRENCY = 5
DEFAULT_DELAY       = 1.0      # segundos entre requests por worker
DEFAULT_OUTPUT      = "dados_ufpel.json"

SKIP_EXTENSIONS = frozenset({
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".zip", ".rar", ".png", ".jpg", ".jpeg",
    ".gif", ".svg", ".mp4", ".mp3", ".ico",
})

NOISE_TAGS = ["script", "style", "noscript", "iframe", "nav",
              "header", "footer", "aside", "form", "button"]

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ufpel_crawler")


# ─────────────────────────────────────────────────────────────────────────────
# Utilitários de texto
# ─────────────────────────────────────────────────────────────────────────────

def _clean(text: str | None) -> str:
    """Normaliza UTF-8, remove espaços e quebras de linha duplicados."""
    if not text:
        return EMPTY
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or EMPTY


def _soup_text(el: Any) -> str:
    return _clean(el.get_text(separator=" ", strip=True)) if el else EMPTY


def _strip_noise(soup: BeautifulSoup) -> None:
    for tag in soup(NOISE_TAGS):
        tag.decompose()


def _normalize_url(url: str) -> str:
    p = urlparse(url)
    return p._replace(fragment="").geturl()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_date(text: str | None) -> date | None:
    """Converte 'DD/MM/AAAA' em date; retorna None se inválido."""
    m = re.search(r"(\d{2})/(\d{2})/(\d{4})", text or "")
    if not m:
        return None
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def _first_valid(*values: str) -> str:
    """Retorna o primeiro valor que não seja EMPTY ou vazio."""
    for v in values:
        if v and v != EMPTY:
            return v
    return EMPTY


def _ficha_get(ficha: dict[str, str], *keys: str) -> str:
    """
    Lookup case-insensitive no dict de ficha-dados.
    Aceita múltiplas chaves alternativas e retorna a primeira que tiver valor.
    Normaliza: remove espaços extras, ignora case e caracteres de anotação como (*).
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*\)", "", s)).strip().lower()

    normalized = {_norm(k): v for k, v in ficha.items()}

    for key in keys:
        # Tentativa exata primeiro
        if key in ficha and ficha[key] != EMPTY:
            return ficha[key]
        # Tentativa case-insensitive + sem anotações
        nk = _norm(key)
        if nk in normalized and normalized[nk] != EMPTY:
            return normalized[nk]
    return EMPTY


# ─────────────────────────────────────────────────────────────────────────────
# Cliente HTTP assíncrono — rate limiting + retry exponencial
# ─────────────────────────────────────────────────────────────────────────────

class RateLimitedClient:
    """
    Gerencia sessão aiohttp com:
      - Semáforo de concorrência (N workers simultâneos)
      - Delay pós-request (polite crawling)
      - Retry exponencial com jitter para erros transitórios
      - Skip automático de erros HTTP listados em SKIP_HTTP_ERRORS
    """

    def __init__(self, concurrency: int = DEFAULT_CONCURRENCY,
                 delay: float = DEFAULT_DELAY):
        self._sem   = asyncio.Semaphore(concurrency)
        self._delay = delay
        self._session: aiohttp.ClientSession | None = None
        self.stats: dict[str, int] = {"ok": 0, "skip": 0, "error": 0, "total": 0}

    async def __aenter__(self) -> "RateLimitedClient":
        timeout   = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(limit=0, ssl=False)
        self._session = aiohttp.ClientSession(
            headers=HEADERS, timeout=timeout, connector=connector,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch(self, url: str) -> BeautifulSoup | None:
        """Faz GET com retry. Aplica delay após cada request."""
        async with self._sem:
            result = await self._fetch_with_retry(url)
            await asyncio.sleep(self._delay)
            return result

    async def _fetch_with_retry(self, url: str) -> BeautifulSoup | None:
        assert self._session is not None, "Use como context manager"
        self.stats["total"] += 1

        for attempt in range(MAX_RETRY):
            try:
                async with self._session.get(url, allow_redirects=True) as resp:
                    if resp.status in SKIP_HTTP_ERRORS:
                        log.warning("SKIP HTTP %s — %s", resp.status, url[:80])
                        self.stats["skip"] += 1
                        return None
                    resp.raise_for_status()
                    html = await resp.text(errors="replace")
                    self.stats["ok"] += 1
                    return BeautifulSoup(html, "lxml")

            except aiohttp.ClientResponseError as exc:
                if exc.status in SKIP_HTTP_ERRORS:
                    log.warning("SKIP HTTP %s — %s", exc.status, url[:80])
                    self.stats["skip"] += 1
                    return None
                if attempt == MAX_RETRY - 1:
                    break
                wait = RETRY_BASE ** (attempt + 1) + random.random()
                log.warning("Retry %d/%d (HTTP %s) %.1fs — %s",
                            attempt + 1, MAX_RETRY, exc.status, wait, url[:70])
                await asyncio.sleep(wait)

            except Exception as exc:
                if attempt == MAX_RETRY - 1:
                    break
                wait = RETRY_BASE ** (attempt + 1) + random.random()
                log.warning("Retry %d/%d (%s) %.1fs — %s",
                            attempt + 1, MAX_RETRY, type(exc).__name__, wait, url[:70])
                await asyncio.sleep(wait)

        log.error("FALHA definitiva — %s", url[:80])
        self.stats["error"] += 1
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Paginação automática
# ─────────────────────────────────────────────────────────────────────────────

async def get_listing_urls(
    client: RateLimitedClient,
    listing_url: str,
    pattern: str,
) -> list[str]:
    """
    Navega todas as páginas de uma listagem (paginação automática) e retorna
    URLs que contêm `pattern`, em ordem determinística.
    """
    collected: set[str] = set()
    visited:   set[str] = set()
    page_url:  str | None = listing_url

    while page_url and page_url not in visited:
        visited.add(page_url)
        soup = await client.fetch(page_url)
        if soup is None:
            break

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if pattern in href:
                path = urlparse(href).path.lower()
                if not any(path.endswith(e) for e in SKIP_EXTENSIONS):
                    collected.add(_normalize_url(urljoin(listing_url, href)))

        page_url = _find_next_page(soup, page_url)

    log.info("[Listagem] %s → %d URLs (padrão=%s)", listing_url, len(collected), pattern)
    return sorted(collected)


def _find_next_page(soup: BeautifulSoup, current: str) -> str | None:
    """Detecta link de próxima página por rel, texto ou classe CSS."""
    for a in soup.find_all("a", href=True):
        rel  = " ".join(a.get("rel", []))
        cls  = " ".join(a.get("class", []))
        text = a.get_text(strip=True).lower()
        is_next = (
            "next" in rel
            or any(t in text for t in ("próxima", "próximo", "next"))
            or text in ("»", ">")
            or any(c in cls.lower() for c in ("next", "proxima", "pagination-next"))
        )
        if is_next:
            full = _normalize_url(urljoin(current, a["href"].strip()))
            if full != current:
                return full
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Extratores compartilhados (padrões HTML do portal UFPel)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ficha(soup: BeautifulSoup) -> dict[str, str]:
    """Extrai pares label→valor da estrutura ficha-dados do portal."""
    ficha = soup.find(class_="ficha-dados")
    if not ficha:
        return {}
    result: dict[str, str] = {}
    for label_el in ficha.find_all(class_="ficha-label"):
        label = label_el.get_text(strip=True).rstrip(":")
        campo = label_el.find_next_sibling(class_="ficha-campo")
        if campo:
            result[label] = _clean(campo.get_text(separator=" ", strip=True))
    return result


def _extract_accordions(soup: BeautifulSoup,
                        container_id: str = "informacoes") -> dict[str, str]:
    """Extrai seções de acordeão em #informacoes (Ementa, Objetivos, etc.)."""
    container = soup.find(id=container_id)
    if not container:
        return {}
    result: dict[str, str] = {}
    for acc in container.find_all(class_=re.compile(r"accordion", re.I)):
        heading = acc.find("h3") or acc.find("h2") or acc.find("h4")
        content = acc.find(class_=re.compile(r"accordion.content|conteudo|body", re.I))
        if not content:
            divs = [d for d in acc.find_all("div", recursive=False)]
            content = divs[1] if len(divs) > 1 else None
        if heading and content:
            key = heading.get_text(strip=True)
            result[key] = _clean(content.get_text(separator=" ", strip=True))
    return result


def _extract_table_rows(table: Any) -> list[dict[str, str]]:
    """Converte tabela HTML em lista de dicts {cabeçalho: valor}."""
    if not table:
        return []
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    rows: list[dict[str, str]] = []
    for tr in table.find_all("tr")[1 if headers else 0:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        if headers:
            row = {headers[i]: _clean(c.get_text(strip=True))
                   for i, c in enumerate(cells) if i < len(headers)}
        else:
            row = {str(i): _clean(c.get_text(strip=True)) for i, c in enumerate(cells)}
        if any(v != EMPTY for v in row.values()):
            rows.append(row)
    return rows


def _extract_list_items(el: Any) -> list[str]:
    if not el:
        return []
    return [t for li in el.find_all("li")
            if (t := _clean(li.get_text(strip=True))) != EMPTY]


# ─────────────────────────────────────────────────────────────────────────────
# Extratores por seção
# ─────────────────────────────────────────────────────────────────────────────

def _extract_curso(soup: BeautifulSoup, url: str) -> dict:
    """
    Extrai dados de um curso do portal UFPel.

    URL do portal: /cursos/cod/XXX  (não /cursos/id/)

    Campos na ficha-dados:
      "Nome do Curso /Conceitos (*)" → nome (strip das anotações)
      "Nível / Grau"                 → nivel
      "Modalidade"                   → modalidade
      "Turno"                        → turno
      "Código UFPel"                 → codigo_ufpel
      "Unidade"                      → unidade
      "Coordenador"                  → coordenador

    Seções:
      #curriculo   → matriz curricular (tabelas por semestre)
      #professores → tabela com <span class="p-name"> e <span class="tabela-detalhe-info">
      #turmas      → turmas ofertadas (tabela)
      #informacoes → accordions (Contextualização, Objetivos, Perfil do Egresso, etc.)
                     pode estar vazio se carregado via JS
    """
    _strip_noise(soup)
    ficha      = _extract_ficha(soup)
    accordions = _extract_accordions(soup)

    # "Nome do Curso /Conceitos (*)" — _ficha_get normaliza e ignora a anotação (*)
    nome = _first_valid(
        _ficha_get(ficha, "Nome do Curso /Conceitos (*)", "Nome do Curso", "Nome"),
        _soup_text(soup.find("h1")),
    )

    # Objetivos podem estar no accordion (se carregados) ou em campo da ficha
    objetivos = _first_valid(
        accordions.pop("Objetivos", ""),
        accordions.pop("Objetivo", ""),
        _ficha_get(ficha, "Objetivos"),
    )

    # ── Matriz curricular — seção #curriculo ────────────────────────────────
    # Estrutura: accordion por semestre → tabela com colunas:
    #   Código | Disciplina / Pré-requisitos | Caráter | Cr. | Horas
    # As células de Código e Disciplina têm <a href="/disciplinas/cod/XXX">
    matriz: list[dict] = []
    curriculo_sec = soup.find(id="curriculo")
    if curriculo_sec:
        sem_num = 0
        for acc_h3 in curriculo_sec.find_all("h3", class_="cor-fundo"):
            sem_num += 1
            sem_label = acc_h3.get_text(strip=True) or f"{sem_num}º Semestre"
            content_div = acc_h3.find_next_sibling(attrs={"data-content": True})
            if not content_div:
                continue
            table = content_div.find("table")
            if not table:
                continue
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            for tr in table.find_all("tr")[1:]:
                cells = tr.find_all("td")
                if not cells:
                    continue
                row: dict = {"semestre": sem_label}
                for i, cell in enumerate(cells):
                    if i >= len(headers):
                        break
                    header = headers[i]
                    # Captura URL da disciplina nas colunas de código e nome
                    link = cell.find("a", href=re.compile(r"/disciplinas/"))
                    if link:
                        row[header] = _clean(link.get_text(strip=True))
                        row[header + "_url"] = urljoin(url, link["href"])
                    else:
                        row[header] = _clean(cell.get_text(strip=True))
                if any(v not in (EMPTY, "") for k, v in row.items() if k != "semestre"):
                    matriz.append(row)
    # Fallback: tabelas com cabeçalho relevante
    if not matriz:
        for table in soup.find_all("table"):
            hdrs = " ".join(th.get_text(strip=True).lower()
                            for th in table.find_all("th"))
            if any(k in hdrs for k in ("disciplina", "crédito", "horas")):
                matriz.extend(_extract_table_rows(table))

    # ── Professores — seção #professores ────────────────────────────────────
    # Estrutura real do portal:
    #   <td class="h-card">
    #     <a class="u-url" href="/servidores/id/XXX">
    #       <span class="tabela-detalhe-titulo p-name">NOME</span>
    #       <span class="p-org">UNIDADE</span>
    #     </a>
    #   </td>
    professores: list[dict[str, str]] = []
    prof_sec = soup.find(id="professores")
    if prof_sec:
        for td in prof_sec.find_all("td", class_="h-card"):
            link      = td.find("a", class_="u-url")
            nome_span = td.find(class_="p-name") or td.find(class_="tabela-detalhe-titulo")
            unit_span = td.find(class_="p-org") or td.find(class_="tabela-detalhe-info")
            prof_nome = _clean(nome_span.get_text(strip=True)) if nome_span else EMPTY
            prof_unit = _clean(unit_span.get_text(strip=True)) if unit_span else EMPTY
            prof_url  = urljoin(url, link["href"]) if link and link.get("href") else ""
            if prof_nome != EMPTY:
                professores.append({
                    "nome":    prof_nome,
                    "unidade": prof_unit,
                    "url":     prof_url,
                    "lattes":  "",  # preenchido no pós-processamento via doc_completos
                })

    # ── Turmas ofertadas — seção #turmas ────────────────────────────────────
    # Estrutura: accordion por semestre → tabela com colunas:
    #   Disciplina / Professores / Horários | Turma | Vagas | Matric.
    # A coluna principal tem: link da disciplina + span com professor + tooltip de horário
    turmas: list[dict] = []
    turmas_sec = soup.find(id="turmas")
    if turmas_sec:
        for table in turmas_sec.find_all("table", class_=re.compile(r"tabela-dados")):
            for tr in table.find_all("tr")[1:]:
                cells = tr.find_all("td")
                if len(cells) < 2:
                    continue
                disc_cell = cells[0]
                # Extrai link + nome da disciplina (ignora tooltip aninhado)
                disc_link = disc_cell.find("a", href=re.compile(r"/disciplinas/"))
                if not disc_link:
                    continue
                disc_nome = _clean(disc_link.get_text(strip=True))
                disc_url  = urljoin(url, disc_link["href"])
                # Extrai professores via spans .tabela-detalhe-info
                profs_turma = [
                    _clean(sp.get_text(strip=True)
                           .replace("Professor responsável pela turma:", "")
                           .replace("Professor Regente:", "")
                           .strip())
                    for sp in disc_cell.find_all("span", class_="tabela-detalhe-info")
                    if sp.get_text(strip=True)
                ]
                # As colunas Turma / Vagas / Matric. são as últimas 3 células
                # (a célula [0] pode ter colunas extras do tooltip aninhado)
                last3 = cells[-3:]
                turma_id = _clean(last3[0].get_text(strip=True)) if len(last3) > 0 else EMPTY
                vagas    = _clean(last3[1].get_text(strip=True)) if len(last3) > 1 else EMPTY
                matricul = _clean(last3[2].get_text(strip=True)) if len(last3) > 2 else EMPTY
                turmas.append({
                    "disciplina":     disc_nome,
                    "disciplina_url": disc_url,
                    "professores":    profs_turma,
                    "turma":          turma_id,
                    "vagas":          vagas,
                    "matriculados":   matricul,
                })

    # ── Formas de ingresso ────────────────────────────────────────────────
    ingresso: list[str] = []
    val = _first_valid(
        accordions.pop("Formas de Ingresso", ""),
        _ficha_get(ficha, "Formas de Ingresso"),
    )
    if val != EMPTY:
        ingresso = [val]
    # Vagas e Formas de Ingresso podem estar no campo da ficha-campo
    vagas_campo = soup.find(class_="ficha-campo", string=re.compile(r"Vagas e Formas", re.I))
    if not ingresso and vagas_campo:
        ingresso = [_soup_text(vagas_campo)]

    dados = {
        "nome":              _clean(nome),
        "nivel":             _ficha_get(ficha, "Nível / Grau", "Nível", "Grau"),
        "modalidade":        _ficha_get(ficha, "Modalidade"),
        "turno":             _ficha_get(ficha, "Turno"),
        "codigo_ufpel":      _ficha_get(ficha, "Código UFPel", "Código"),
        "codigo_emec":       _ficha_get(ficha, "Código e-MEC"),
        "unidade":           _ficha_get(ficha, "Unidade"),
        "coordenador":       _ficha_get(ficha, "Coordenador"),
        "objetivos":         objetivos,
        "contextualizacao":  accordions.pop("Contextualização", EMPTY),
        "perfil_egresso":    accordions.pop("Perfil do Egresso", EMPTY),
        "formas_ingresso":   ingresso or [EMPTY],
        "matriz_curricular": matriz,
        "professores":       professores or [{"nome": EMPTY, "unidade": EMPTY, "url": "", "lattes": ""}],
        "turmas_ofertadas":  turmas,
        **accordions,
    }

    # Constrói embedding_text rico para busca semântica
    profs_texto = "\n".join(
        f"  - {p['nome']}"
        + (f" ({p['unidade']})" if p.get("unidade") and p["unidade"] != EMPTY else "")
        + (f" | Lattes: {p['lattes']}" if p.get("lattes") else "")
        for p in professores
        if p.get("nome") and p["nome"] != EMPTY
    ) or EMPTY

    # Todas as disciplinas da grade (sem limite para busca completa)
    disciplinas_grade = "\n".join(
        f"  - {row.get('Disciplina / Pré-requisitos', row.get('Disciplina', ''))} "
        f"[{row.get('semestre', '')}]"
        f" ({row.get('Caráter', '')})"
        for row in matriz
        if isinstance(row, dict) and row.get("Disciplina / Pré-requisitos")
    )

    contextualizacao = dados.get("contextualizacao", EMPTY)
    perfil_egresso   = dados.get("perfil_egresso", EMPTY)

    partes = [
        f"Curso de {dados['nivel']} em {dados['nome']} - UFPel",
        f"Modalidade: {dados['modalidade']} | Turno: {dados['turno']}",
        f"Unidade responsável: {dados['unidade']}",
        f"Coordenador: {dados['coordenador']}",
    ]
    if contextualizacao and contextualizacao != EMPTY:
        partes.append(f"\nSobre o curso:\n{contextualizacao}")
    if objetivos and objetivos != EMPTY:
        partes.append(f"\nObjetivos:\n{objetivos}")
    if perfil_egresso and perfil_egresso != EMPTY:
        partes.append(f"\nPerfil do egresso:\n{perfil_egresso}")
    if profs_texto and profs_texto != EMPTY:
        partes.append(f"\nProfessores e docentes do curso:\n{profs_texto}")
    if disciplinas_grade:
        partes.append(f"\nMatriz curricular (disciplinas):\n{disciplinas_grade}")

    embedding_text = "\n".join(partes)

    return {
        "titulo":          dados["nome"],
        "embedding_text":  embedding_text,
        "metadata": {
            "url":        url,
            "nivel":      dados["nivel"],
            "modalidade": dados["modalidade"],
            "unidade":    dados["unidade"],
            "crawled_at": _now(),
        },
        "dados_completos": dados,
    }


def _extract_disciplina(soup: BeautifulSoup, url: str) -> dict:
    """
    Extrai dados de uma disciplina do portal UFPel.

    Campos na ficha-dados do portal:
      "Nome da Atividade" → nome
      "CÓDIGO"            → codigo       (campo em maiúsculas no HTML)
      "Carga Horária"     → carga_horaria
      "Tipo de Atividade" → tipo_atividade
      "Periodicidade"     → periodicidade
      "Unidade responsável" → unidade_responsavel
      "CRÉDITOS"          → creditos
      "CARGA HORÁRIA TEÓRICA"    → ch_teorica
      "CARGA HORÁRIA OBRIGATÓRIA" → ch_obrigatoria
      "FREQUÊNCIA APROVAÇÃO"     → freq_aprovacao
    """
    _strip_noise(soup)
    ficha      = _extract_ficha(soup)
    accordions = _extract_accordions(soup)

    # _ficha_get faz lookup case-insensitive e ignora anotações como (*)
    nome   = _first_valid(
        _ficha_get(ficha, "Nome da Atividade", "Nome", "Disciplina"),
        _soup_text(soup.find("h1")),
    )
    codigo = _ficha_get(ficha, "CÓDIGO", "Código", "Código da Disciplina", "codigo")

    objetivos = _first_valid(
        accordions.pop("Objetivos", ""),
        accordions.pop("Objetivo", ""),
    )
    ementa   = accordions.pop("Ementa", EMPTY)
    conteudo = accordions.pop("Conteúdo Programático", EMPTY)
    biblio   = accordions.pop("Bibliografia", EMPTY)

    # Equivalências
    equiv: list[str] = []
    equiv_val = _first_valid(
        accordions.pop("Equivalências", ""),
        accordions.pop("Equivalencia", ""),
        accordions.pop("Disciplinas Equivalentes", ""),
    )
    if equiv_val != EMPTY:
        equiv = [equiv_val]
    else:
        equiv_sec = soup.find(id=re.compile(r"equiv", re.I))
        if equiv_sec:
            equiv = _extract_list_items(equiv_sec.find("ul")) or [_soup_text(equiv_sec)]

    # Turmas ofertadas
    turmas: list[dict] = []
    turmas_sec = soup.find(id=re.compile(r"turma", re.I))
    if turmas_sec:
        turmas = _extract_table_rows(turmas_sec.find("table"))

    dados = {
        "codigo":               codigo,
        "nome":                 _clean(nome),
        "tipo_atividade":       _ficha_get(ficha, "Tipo de Atividade"),
        "periodicidade":        _ficha_get(ficha, "Periodicidade"),
        "creditos":             _ficha_get(ficha, "CRÉDITOS", "Créditos"),
        "carga_horaria":        _ficha_get(ficha, "Carga Horária"),
        "ch_teorica":           _ficha_get(ficha, "CARGA HORÁRIA TEÓRICA", "Carga Horária Teórica"),
        "ch_obrigatoria":       _ficha_get(ficha, "CARGA HORÁRIA OBRIGATÓRIA", "Carga Horária Obrigatória"),
        "freq_aprovacao":       _ficha_get(ficha, "FREQUÊNCIA APROVAÇÃO", "Frequência Aprovação"),
        "unidade_responsavel":  _ficha_get(ficha, "Unidade responsável", "Unidade Responsável", "Unidade"),
        "objetivos":            objetivos,
        "ementa":               ementa,
        "conteudo_programatico": conteudo,
        "bibliografia":         biblio,
        "equivalencias":        equiv or [EMPTY],
        "turmas_ofertadas":     turmas,
        **accordions,
    }

    embedding_text = _first_valid(objetivos, ementa, _clean(nome))

    return {
        "titulo":          dados["nome"],
        "embedding_text":  embedding_text,
        "metadata": {
            "url":        url,
            "codigo":     codigo,
            "unidade":    dados["unidade_responsavel"],
            "crawled_at": _now(),
        },
        "dados_completos": dados,
    }


def _extract_projeto(soup: BeautifulSoup, url: str) -> dict | None:
    """Retorna None se o projeto não for vigente."""
    _strip_noise(soup)
    ficha      = _extract_ficha(soup)
    accordions = _extract_accordions(soup)

    titulo = _first_valid(
        ficha.get("Título", ""),
        ficha.get("Nome", ""),
        _soup_text(soup.find("h1")),
    )
    resumo = _first_valid(
        ficha.get("Resumo", ""),
        accordions.get("Resumo", ""),
    )
    data_inicio = _first_valid(
        ficha.get("Data de Início", ""),
        ficha.get("Data Início", ""),
        ficha.get("Início", ""),
    )
    data_fim = _first_valid(
        ficha.get("Data de Término", ""),
        ficha.get("Data Fim", ""),
        ficha.get("Término", ""),
        ficha.get("Fim", ""),
    )
    situacao = _first_valid(ficha.get("Situação", ""), ficha.get("Status", ""))

    if not _is_vigente(data_fim, situacao):
        log.debug("Não vigente (data_fim=%s) — %s", data_fim, url[:70])
        return None

    # Equipe
    equipe: list[str] = []
    equipe_sec = (
        soup.find(id=re.compile(r"equipe|membro|participante", re.I))
        or soup.find(class_=re.compile(r"equipe|membro", re.I))
    )
    if equipe_sec:
        equipe = _extract_list_items(equipe_sec.find("ul"))
        if not equipe:
            for row in _extract_table_rows(equipe_sec.find("table")):
                line = " — ".join(v for v in row.values() if v != EMPTY)
                if line:
                    equipe.append(line)

    # Financeiro
    financeiro: dict[str, str] = {}
    fin_sec = soup.find(id=re.compile(r"financ|orcamento|orçamento|budget", re.I))
    if fin_sec:
        financeiro = _extract_ficha(fin_sec) or {"info": _soup_text(fin_sec)}

    skip_keys = {
        "Título", "Nome", "Resumo", "Data de Início", "Data Início",
        "Início", "Data de Término", "Data Fim", "Término", "Fim",
        "Situação", "Status", "Coordenador", "Área", "Unidade",
    }
    dados = {
        "titulo":       _clean(titulo),
        "resumo":       resumo,
        "data_inicio":  data_inicio,
        "data_fim":     data_fim,
        "situacao":     situacao,
        "coordenador":  ficha.get("Coordenador", EMPTY),
        "area":         ficha.get("Área", EMPTY),
        "unidade":      ficha.get("Unidade", EMPTY),
        "equipe":       equipe or [EMPTY],
        "financeiro":   financeiro or {"info": EMPTY},
        **{k: v for k, v in ficha.items() if k not in skip_keys},
    }

    return {
        "titulo":          dados["titulo"],
        "embedding_text":  _first_valid(resumo, _clean(titulo)),
        "metadata": {
            "url":         url,
            "data_inicio": data_inicio,
            "data_fim":    data_fim,
            "situacao":    situacao,
            "coordenador": dados["coordenador"],
            "crawled_at":  _now(),
        },
        "dados_completos": dados,
    }


def _is_vigente(data_fim_str: str, situacao: str = "") -> bool:
    """
    Retorna True se o projeto for vigente.
    Critério: data_fim >= hoje OU status indica projeto ativo.
    Projetos sem data_fim e sem status negativo são incluídos.
    """
    today = date.today()
    sit_lower = situacao.lower()

    encerrados = {"concluído", "concluido", "encerrado", "finalizado",
                  "cancelado", "suspenso", "inativo"}
    vigentes   = {"em execução", "ativo", "em andamento", "vigente",
                  "aberto", "aprovado", "em vigor"}

    if any(v in sit_lower for v in encerrados):
        d = _parse_date(data_fim_str)
        return bool(d and d >= today)

    if any(v in sit_lower for v in vigentes):
        return True

    d = _parse_date(data_fim_str)
    if d:
        return d >= today

    # Sem data_fim e sem status negativo: inclui por padrão
    return True


def _extract_servidor(soup: BeautifulSoup, url: str) -> dict:
    _strip_noise(soup)
    ficha = _extract_ficha(soup)

    nome     = _first_valid(
        ficha.get("Nome do Servidor", ""),
        ficha.get("Nome", ""),
        _soup_text(soup.find("h1")),
    )
    matricula = ficha.get("Matrícula", EMPTY)
    cargo     = _first_valid(ficha.get("Cargo", ""), ficha.get("Função", ""))
    lotacao   = _first_valid(
        ficha.get("Lotação", ""),
        ficha.get("Unidade de Lotação", ""),
        ficha.get("Unidade", ""),
    )
    regime = _first_valid(
        ficha.get("Regime de Trabalho", ""),
        ficha.get("Regime", ""),
    )

    # Currículo Lattes — seção #lattes
    curriculo_text = EMPTY
    lattes_url     = EMPTY
    lattes_sec     = soup.find(id="lattes")
    if lattes_sec:
        link = lattes_sec.find("a", href=re.compile(r"lattes\.cnpq", re.I))
        if link:
            lattes_url = link["href"]
        parts: list[str] = []
        for h3 in lattes_sec.find_all("h3"):
            label = h3.get_text(strip=True)
            sib_texts: list[str] = []
            for sib in h3.next_siblings:
                if getattr(sib, "name", None) == "h3":
                    break
                if hasattr(sib, "get_text"):
                    t = _clean(sib.get_text(strip=True))
                    if t not in (EMPTY, "Informações extraídas do Currículo Lattes"):
                        sib_texts.append(t)
            if sib_texts:
                parts.append(f"{label}: {' '.join(sib_texts)}")
        curriculo_text = _clean("\n".join(parts)) if parts else EMPTY

    # PGD (Programa de Gestão e Desempenho)
    pgd_text = EMPTY
    pgd_sec  = soup.find(id=re.compile(r"\bpgd\b", re.I))
    if pgd_sec:
        pgd_text = _soup_text(pgd_sec)

    # Projetos vigentes vinculados ao servidor
    projetos_vigentes: list[dict[str, str]] = []
    proj_sec = soup.find(id=re.compile(r"projeto", re.I))
    if proj_sec:
        for a in proj_sec.find_all("a", href=True):
            if "/projetos/id/" in a["href"]:
                projetos_vigentes.append({
                    "titulo": _clean(a.get_text(strip=True)),
                    "url":    _normalize_url(urljoin(url, a["href"])),
                })

    skip_keys = {
        "Nome do Servidor", "Nome", "Matrícula", "Cargo", "Função",
        "Lotação", "Unidade de Lotação", "Unidade",
        "Regime de Trabalho", "Regime", "Contato", "E-mail",
    }
    dados = {
        "nome":              _clean(nome),
        "matricula":         matricula,
        "cargo":             cargo,
        "lotacao":           lotacao,
        "regime":            regime,
        "contato":           ficha.get("Contato", EMPTY),
        "email":             ficha.get("E-mail", EMPTY),
        "curriculo":         curriculo_text,
        "lattes_url":        lattes_url,
        "pgd":               pgd_text,
        "projetos_vigentes": projetos_vigentes,
        **{k: v for k, v in ficha.items() if k not in skip_keys},
    }

    # Extrai campos estruturados do currículo Lattes para o embedding
    areas      = dados.get("Áreas de atuação", dados.get("Área de Atuação", EMPTY))
    titulacao  = dados.get("Titulação", dados.get("Título", EMPTY))
    resumo_lattes = EMPTY
    if curriculo_text != EMPTY:
        # Tenta isolar o Resumo dentro do curriculo_text
        import re as _re
        m = _re.search(r"Resumo[:\s]+(.+?)(?=\n[A-ZÁÉÍÓÚÃÕÂÊÔÇ][a-záéíóúãõâêôç]|$)",
                       curriculo_text, _re.DOTALL)
        resumo_lattes = m.group(1).strip() if m else curriculo_text

    partes_serv = [f"Professor/Servidor: {_clean(nome)}"]
    if cargo and cargo != EMPTY:
        partes_serv.append(f"Cargo: {cargo}")
    if lotacao and lotacao != EMPTY:
        partes_serv.append(f"Lotação/Unidade: {lotacao}")
    if titulacao and titulacao != EMPTY:
        partes_serv.append(f"Titulação: {titulacao}")
    if areas and areas != EMPTY:
        partes_serv.append(f"\nÁreas de atuação:\n{areas}")
    if resumo_lattes and resumo_lattes != EMPTY:
        partes_serv.append(f"\nResumo:\n{resumo_lattes}")

    embedding_text = "\n".join(partes_serv) if len(partes_serv) > 1 else _clean(nome)

    return {
        "titulo":          dados["nome"],
        "embedding_text":  embedding_text,
        "metadata": {
            "url":        url,
            "matricula":  matricula,
            "cargo":      cargo,
            "lotacao":    lotacao,
            "crawled_at": _now(),
        },
        "dados_completos": dados,
    }


def _extract_unidade(soup: BeautifulSoup, url: str) -> dict:
    _strip_noise(soup)
    ficha      = _extract_ficha(soup)
    accordions = _extract_accordions(soup)

    nome  = _first_valid(
        ficha.get("Nome da Unidade", ""),
        ficha.get("Unidade", ""),
        ficha.get("Nome", ""),
        _soup_text(soup.find("h1")),
    )
    sigla = ficha.get("Sigla", EMPTY)

    # Estrutura organizacional
    estrutura_sec = soup.find(id=re.compile(r"estrutura|organograma", re.I))
    estrutura = _soup_text(estrutura_sec) if estrutura_sec \
        else accordions.pop("Estrutura", EMPTY)

    # Graduação e Pós-Graduação
    grad_sec    = soup.find(id=re.compile(r"graduacao|graduação", re.I))
    posgrad_sec = soup.find(id=re.compile(r"pos.grad|pós.grad", re.I))
    graduacao   = _soup_text(grad_sec)    if grad_sec    else accordions.pop("Graduação",     EMPTY)
    pos_grad    = _soup_text(posgrad_sec) if posgrad_sec else accordions.pop("Pós-Graduação", EMPTY)

    # Orçamento
    orc_sec   = soup.find(id=re.compile(r"orçamento|orcamento|budget", re.I))
    orcamento = _soup_text(orc_sec) if orc_sec else accordions.pop("Orçamento", EMPTY)

    # Servidores da unidade
    servidores: list[str] = []
    serv_sec = soup.find(id=re.compile(r"servidor|membro|equipe|quadro", re.I))
    if serv_sec:
        servidores = _extract_list_items(serv_sec.find("ul"))
        if not servidores:
            for row in _extract_table_rows(serv_sec.find("table")):
                servidores.append(next(iter(row.values()), EMPTY))

    # Telefones e contatos
    telefones: list[str] = []
    tel_sec = soup.find(id=re.compile(r"telefone|contato|fone", re.I))
    if tel_sec:
        telefones = _extract_list_items(tel_sec.find("ul"))
        if not telefones:
            for m in re.findall(r"\(?\d{2}\)?\s*\d{4,5}[-.\s]?\d{4}",
                                tel_sec.get_text()):
                telefones.append(m.strip())

    skip_keys = {
        "Nome da Unidade", "Unidade", "Nome", "Sigla",
        "Diretor", "Diretor Geral", "E-mail", "Endereço",
    }
    dados = {
        "nome":          _clean(nome),
        "sigla":         sigla,
        "estrutura":     estrutura,
        "graduacao":     graduacao,
        "pos_graduacao": pos_grad,
        "orcamento":     orcamento,
        "diretor":       _first_valid(
            ficha.get("Diretor", ""),
            ficha.get("Diretor Geral", ""),
        ),
        "email":         ficha.get("E-mail", EMPTY),
        "endereco":      ficha.get("Endereço", EMPTY),
        "servidores":    servidores or [EMPTY],
        "telefones":     telefones or [EMPTY],
        **{k: v for k, v in ficha.items() if k not in skip_keys},
        **accordions,
    }

    emb_parts = [p for p in (_clean(nome), sigla, estrutura) if p != EMPTY]

    return {
        "titulo":          dados["nome"],
        "embedding_text":  " ".join(emb_parts) or EMPTY,
        "metadata": {
            "url":        url,
            "sigla":      sigla,
            "diretor":    dados["diretor"],
            "crawled_at": _now(),
        },
        "dados_completos": dados,
    }


def _extract_gestao(soup: BeautifulSoup, url: str) -> dict:
    _strip_noise(soup)
    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"conteudo|content|main", re.I))
        or soup.find(class_=re.compile(r"conteudo|content|main", re.I))
        or soup.find("body")
    )
    full_text = _soup_text(main)

    # Sub-seções: Reitoria, Vice-Reitoria, Pró-Reitorias, Direções, etc.
    secoes: dict[str, str] = {}
    if main:
        for heading in main.find_all(["h2", "h3"]):
            label = heading.get_text(strip=True)
            parts: list[str] = []
            for sib in heading.next_siblings:
                if getattr(sib, "name", None) in ("h2", "h3"):
                    break
                if hasattr(sib, "get_text"):
                    t = _clean(sib.get_text(strip=True))
                    if t != EMPTY:
                        parts.append(t)
            if parts:
                secoes[label] = " ".join(parts)

    return {
        "titulo":          "Gestão Institucional — UFPel",
        "embedding_text":  full_text,
        "metadata":        {"url": url, "crawled_at": _now()},
        "dados_completos": {"texto": full_text, "secoes": secoes},
    }


def _extract_sobre(soup: BeautifulSoup, url: str) -> dict:
    _strip_noise(soup)
    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"conteudo|content|main", re.I))
        or soup.find(class_=re.compile(r"conteudo|content|main", re.I))
        or soup.find("body")
    )
    full_text = _soup_text(main)

    # Sub-seções: História, Missão, Visão, Valores, etc.
    secoes: dict[str, str] = {}
    if main:
        for heading in main.find_all(["h2", "h3"]):
            label = heading.get_text(strip=True)
            parts: list[str] = []
            for sib in heading.next_siblings:
                if getattr(sib, "name", None) in ("h2", "h3"):
                    break
                if hasattr(sib, "get_text"):
                    t = _clean(sib.get_text(strip=True))
                    if t != EMPTY:
                        parts.append(t)
            if parts:
                secoes[label] = " ".join(parts)

    return {
        "titulo":          "Sobre a UFPel",
        "embedding_text":  full_text,
        "metadata":        {"url": url, "crawled_at": _now()},
        "dados_completos": {"texto": full_text, "secoes": secoes},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Construtor de documento padrão
# ─────────────────────────────────────────────────────────────────────────────

def _build_doc(tipo: str, extracted: dict) -> dict:
    return {
        "id":              str(uuid.uuid4()),
        "tipo":            tipo,
        "titulo":          extracted.get("titulo", EMPTY),
        "embedding_text":  extracted.get("embedding_text", EMPTY),
        "metadata":        extracted.get("metadata", {}),
        "dados_completos": extracted.get("dados_completos", {}),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Crawler principal (assíncrono)
# ─────────────────────────────────────────────────────────────────────────────

class UFPelCrawler:
    """
    Crawler assíncrono do Portal Institucional UFPel.

    Estratégia RAG:
      1. Busca vetorial (campo embedding_text)
      2. Recuperação dos documentos mais relevantes pelo ID
      3. Consulta ao banco para dados_completos
      4. Envio ao LLM para síntese da resposta

    Exemplo:
      "Quais disciplinas existem no curso de Ciência da Computação?"
      → busca vetorial encontra o curso
      → recupera dados_completos: matriz_curricular, professores, turmas
      → LLM sintetiza a resposta com contexto completo
    """

    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        delay: float = DEFAULT_DELAY,
    ):
        self._concurrency = concurrency
        self._delay       = delay

    async def _crawl_listed_section(
        self,
        tipo: str,
        listing_url: str,
        url_pattern: str,
        extractor: Callable,
        limit: int | None = None,
    ) -> list[dict]:
        async with RateLimitedClient(self._concurrency, self._delay) as client:
            urls = await get_listing_urls(client, listing_url, url_pattern)
            if limit is not None:
                urls = urls[:limit]
            log.info("[%s] Processando %d itens...", tipo.upper(), len(urls))

            tasks = [
                self._process_item(client, url, tipo, extractor)
                for url in urls
            ]
            results = await asyncio.gather(*tasks)

        docs = [r for r in results if r is not None]
        log.info("[%s] %d/%d documentos coletados.", tipo.upper(), len(docs), len(urls))
        return docs

    async def _process_item(
        self,
        client: RateLimitedClient,
        url: str,
        tipo: str,
        extractor: Callable,
    ) -> dict | None:
        soup = await client.fetch(url)
        if soup is None:
            return None
        try:
            extracted = extractor(soup, url)
            if extracted is None:
                return None
            return _build_doc(tipo, extracted)
        except Exception as exc:
            log.error("[%s] Erro ao extrair %s: %s", tipo, url[:70], exc)
            return None

    async def crawl_cursos(self, limit: int | None = None) -> list[dict]:
        return await self._crawl_listed_section(
            "curso", SECTION_URLS["cursos"], URL_PATTERNS["cursos"],
            _extract_curso, limit,
        )

    async def crawl_disciplinas(self, limit: int | None = None) -> list[dict]:
        return await self._crawl_listed_section(
            "disciplina", SECTION_URLS["disciplinas"], URL_PATTERNS["disciplinas"],
            _extract_disciplina, limit,
        )

    async def crawl_projetos(self, limit: int | None = None) -> list[dict]:
        return await self._crawl_listed_section(
            "projeto", SECTION_URLS["projetos"], URL_PATTERNS["projetos"],
            _extract_projeto, limit,
        )

    async def crawl_servidores(self, limit: int | None = None) -> list[dict]:
        return await self._crawl_listed_section(
            "servidor", SECTION_URLS["servidores"], URL_PATTERNS["servidores"],
            _extract_servidor, limit,
        )

    async def crawl_unidades(self, limit: int | None = None) -> list[dict]:
        return await self._crawl_listed_section(
            "unidade", SECTION_URLS["unidades"], URL_PATTERNS["unidades"],
            _extract_unidade, limit,
        )

    async def crawl_gestao(self) -> list[dict]:
        async with RateLimitedClient(1, 0.0) as client:
            url  = SECTION_URLS["gestao"]
            soup = await client.fetch(url)
            if soup is None:
                return []
            try:
                return [_build_doc("gestao", _extract_gestao(soup, url))]
            except Exception as exc:
                log.error("[GESTAO] Erro: %s", exc)
                return []

    async def crawl_sobre(self) -> list[dict]:
        async with RateLimitedClient(1, 0.0) as client:
            url  = SECTION_URLS["sobre"]
            soup = await client.fetch(url)
            if soup is None:
                return []
            try:
                return [_build_doc("sobre", _extract_sobre(soup, url))]
            except Exception as exc:
                log.error("[SOBRE] Erro: %s", exc)
                return []

    async def crawl_all(
        self,
        cursos_max:      int | None = None,
        disciplinas_max: int | None = None,
        projetos_max:    int | None = None,
        servidores_max:  int | None = None,
        unidades_max:    int | None = None,
    ) -> list[dict]:
        """Coleta todas as seções em sequência, respeitando o rate limit."""
        log.info("[Crawler] Coleta completa iniciada — %s", _now())
        all_docs: list[dict] = []

        plan = [
            ("cursos",      self.crawl_cursos(cursos_max)),
            ("disciplinas", self.crawl_disciplinas(disciplinas_max)),
            ("projetos",    self.crawl_projetos(projetos_max)),
            ("servidores",  self.crawl_servidores(servidores_max)),
            ("unidades",    self.crawl_unidades(unidades_max)),
            ("gestao",      self.crawl_gestao()),
            ("sobre",       self.crawl_sobre()),
        ]

        for tipo, coro in plan:
            docs = await coro
            all_docs.extend(docs)
            log.info("[Resumo] %s: +%d docs (total=%d)", tipo, len(docs), len(all_docs))

        return all_docs


# ─────────────────────────────────────────────────────────────────────────────
# Persistência
# ─────────────────────────────────────────────────────────────────────────────

def save_json(docs: list[dict], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)

    by_tipo: dict[str, int] = {}
    for d in docs:
        t = d.get("tipo", "?")
        by_tipo[t] = by_tipo.get(t, 0) + 1

    log.info("[Saída] %d documentos salvos em '%s'", len(docs), output_path)
    for tipo, n in sorted(by_tipo.items()):
        log.info("        ↳ %-20s: %d", tipo, n)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Crawler Async do Portal Institucional UFPel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Saída e performance")
    g.add_argument("--output",           default=DEFAULT_OUTPUT, metavar="FILE",
                   help="Arquivo JSON de saída")
    g.add_argument("--concurrency",      type=int, default=DEFAULT_CONCURRENCY, metavar="N",
                   help="Requisições simultâneas por seção")
    g.add_argument("--delay",            type=float, default=DEFAULT_DELAY, metavar="SECS",
                   help="Delay (s) entre requests por worker")

    g2 = p.add_argument_group("Seleção de seções")
    g2.add_argument("--all-types",        action="store_true",
                    help="Coleta todas as seções")
    g2.add_argument("--cursos-only",      action="store_true")
    g2.add_argument("--disciplinas-only", action="store_true")
    g2.add_argument("--projetos-only",    action="store_true")
    g2.add_argument("--servidores-only",  action="store_true")
    g2.add_argument("--unidades-only",    action="store_true")
    g2.add_argument("--gestao-only",      action="store_true")
    g2.add_argument("--sobre-only",       action="store_true")

    g3 = p.add_argument_group("Limites por seção (padrão: sem limite)")
    g3.add_argument("--cursos-max",       type=int, default=None, metavar="N")
    g3.add_argument("--disciplinas-max",  type=int, default=None, metavar="N")
    g3.add_argument("--projetos-max",     type=int, default=None, metavar="N")
    g3.add_argument("--servidores-max",   type=int, default=None, metavar="N")
    g3.add_argument("--unidades-max",     type=int, default=None, metavar="N")
    return p


async def _amain(args: argparse.Namespace) -> None:
    crawler = UFPelCrawler(concurrency=args.concurrency, delay=args.delay)
    t0   = datetime.now(timezone.utc)
    docs: list[dict] = []

    any_specific = any([
        args.all_types, args.cursos_only, args.disciplinas_only,
        args.projetos_only, args.servidores_only, args.unidades_only,
        args.gestao_only, args.sobre_only,
    ])

    if not any_specific or args.all_types:
        docs = await crawler.crawl_all(
            cursos_max=args.cursos_max,
            disciplinas_max=args.disciplinas_max,
            projetos_max=args.projetos_max,
            servidores_max=args.servidores_max,
            unidades_max=args.unidades_max,
        )
    else:
        if args.cursos_only:
            docs += await crawler.crawl_cursos(args.cursos_max)
        if args.disciplinas_only:
            docs += await crawler.crawl_disciplinas(args.disciplinas_max)
        if args.projetos_only:
            docs += await crawler.crawl_projetos(args.projetos_max)
        if args.servidores_only:
            docs += await crawler.crawl_servidores(args.servidores_max)
        if args.unidades_only:
            docs += await crawler.crawl_unidades(args.unidades_max)
        if args.gestao_only:
            docs += await crawler.crawl_gestao()
        if args.sobre_only:
            docs += await crawler.crawl_sobre()

    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("[Crawler] Concluído em %.1fs — %d documentos.", elapsed, len(docs))
    save_json(docs, args.output)


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
