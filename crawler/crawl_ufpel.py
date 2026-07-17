"""
Crawler Async do Portal Institucional UFPel — Cursos de Computação
===================================================================
Deep-crawl focado em 4 cursos:

  ┌────────┬───────────────────────────┬───────────────┬────────────────────┐
  │ Código │ Curso                     │ Nível         │ Grau               │
  ├────────┼───────────────────────────┼───────────────┼────────────────────┤
  │ 3900   │ Ciência da Computação     │ Graduação     │ Bacharelado        │
  │ 8102   │ Computação                │ Pós-Graduação │ Doutorado          │
  │ 7057   │ Computação                │ Pós-Graduação │ Mestrado Acadêmico │
  │ 3910   │ Engenharia de Computação  │ Graduação     │ Bacharelado        │
  └────────┴───────────────────────────┴───────────────┴────────────────────┘

Fluxo de captura (4 fases):
  1. CURSOS       → ficha-dados, aba Informações (todas as subabas/accordions),
                    Matriz Curricular, Professores, Turmas Ofertadas e
                    rodapé (conceitos de curso + formas de ingresso + vagas)
  2. DISCIPLINAS  → cada disciplina da matriz/turmas dos cursos:
                    informações gerais, Ementa, Objetivos, Conteúdo
                    Programático, Bibliografia e Turmas Ofertadas
  3. PROFESSORES  → cada professor dos cursos (apenas situação ativa):
                    informações gerais, currículo (Resumo, Formação
                    acadêmica, Áreas de atuação), projetos ATIVOS e
                    disciplinas que ministra
  4. PROJETOS     → cada projeto ativo dos professores: informações gerais,
                    aba Informações, Equipe (apenas membros vigentes) e
                    Financeiro (ou "Não há informações disponíveis")

Schema de saída (por documento):
  {
    "id":             <uuid>,
    "tipo":           curso | disciplina | servidor | projeto,
    "titulo":         <nome/título do registro>,
    "embedding_text": <resumo do registro — usado para embedding no pgvector>,
    "metadata":       { url, crawled_at, ...campos filtráveis },
    "dados_completos":{ todos os dados estruturados — vai para doc_completos
                        (JSONB), consultável via SQL }
  }

Cada tipo é ingerido em sua própria collection no pgvector
(ufpel_cursos, ufpel_disciplinas, ufpel_servidores, ufpel_projetos).

Campos vazios, nulos ou inexistentes são preenchidos com:
  "Não há informações disponíveis"

Uso:
    python crawl_ufpel.py --output dados_ufpel.json
    python crawl_ufpel.py --concurrency 5 --delay 1.0
    python crawl_ufpel.py --cursos 3900 3910        # subconjunto dos alvos

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
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

BASE_URL = "https://institucional.ufpel.edu.br"

# Cursos alvo do crawl — código UFPel → identificação
TARGET_CURSOS: dict[str, dict[str, str]] = {
    "3900": {"nome": "Ciência da Computação",    "nivel": "Graduação",
             "grau": "Bacharelado",        "turno": "Integral (matutino+vespertino)"},
    "8102": {"nome": "Computação",               "nivel": "Pós-Graduação",
             "grau": "Doutorado",          "turno": "Integral"},
    "7057": {"nome": "Computação",               "nivel": "Pós-Graduação",
             "grau": "Mestrado Acadêmico", "turno": "Integral"},
    "3910": {"nome": "Engenharia de Computação", "nivel": "Graduação",
             "grau": "Bacharelado",        "turno": "Integral (matutino+vespertino)"},
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

# Situações de servidor consideradas INATIVAS — registros com essas situações
# são descartados (o portal cria um ID por vínculo; mantemos só o ativo).
INACTIVE_SITUACOES: frozenset[str] = frozenset({
    "aposentado", "falecido", "exonerado", "demitido", "excluído", "excluido",
    "exclusão", "exclusao", "redistribuído", "redistribuido", "cedido",
    "contrato encerrado", "rescindido", "desligado", "vacância", "vacancia",
})

NOISE_TAGS = ["script", "style", "noscript", "iframe", "nav",
              "header", "footer", "aside", "form", "button"]

DIAS_SEMANA = ("SEG", "TER", "QUA", "QUI", "SEX", "SAB", "SÁB", "DOM")

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
    Normaliza: remove espaços extras, ignora case e anotações como (*).
    """
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"\s*\([^)]*\)", "", s)).strip().lower()

    normalized = {_norm(k): v for k, v in ficha.items()}

    for key in keys:
        if key in ficha and ficha[key] != EMPTY:
            return ficha[key]
        nk = _norm(key)
        if nk in normalized and normalized[nk] != EMPTY:
            return normalized[nk]
    return EMPTY


def _direct_rows(table: Any) -> list:
    """
    Retorna apenas as <tr> diretas da tabela, ignorando linhas de tabelas
    aninhadas (o portal aninha tabelas de horários dentro das células).
    """
    if not table:
        return []
    return [tr for tr in table.find_all("tr") if tr.find_parent("table") is table]


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
# Extratores compartilhados (padrões HTML do portal UFPel)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ficha(root: Any) -> dict[str, str]:
    """Extrai pares label→valor da estrutura ficha-dados do portal."""
    ficha = root.find(class_="ficha-dados") if root else None
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
    """
    Extrai TODAS as seções de acordeão (subabas) de um container.
    Ex.: em #informacoes de curso: Contextualização, Objetivos, Perfil do
    Egresso, Competências e habilidades, Organização Curricular, etc.
    """
    container = soup.find(id=container_id)
    if not container:
        return {}
    result: dict[str, str] = {}
    for acc in container.find_all(class_=re.compile(r"accordion", re.I)):
        if "accordion-content" in (acc.get("class") or []):
            continue
        heading = acc.find(["h2", "h3", "h4"])
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
    for tr in _direct_rows(table)[1 if headers else 0:]:
        cells = tr.find_all(["td", "th"], recursive=False)
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


def _parse_horarios(cell: Any) -> list[str]:
    """
    Extrai horários de aula da tabela aninhada `grade-horarios` do portal:
    células por período (Manhã/Tarde/Noite) com o dia marcado em
    <span class="grade-horarios-dia">SEG</span> seguido do horário.
    Retorna lista tipo ["SEG 10:00 - 10:50", "QUA 08:00 - 08:50", ...].
    """
    horarios: list[str] = []
    for tab in cell.find_all("table", class_=re.compile(r"grade-horarios")):
        for td in tab.find_all("td"):
            dia = ""
            for child in td.children:
                if getattr(child, "name", None) == "span":
                    t = child.get_text(strip=True)
                    if t:
                        dia = t
                elif isinstance(child, str) and child.strip():
                    hora = child.strip()
                    horarios.append(f"{dia} {hora}".strip() if dia else hora)
    return horarios


def _split_lines(el: Any) -> list[str]:
    """
    Extrai itens de um elemento onde cada item é separado por <br> ou <li>
    (padrão do portal em Formação acadêmica e Áreas de atuação).
    """
    if not el:
        return []
    itens = _extract_list_items(el)
    if itens:
        return itens
    return [t for line in el.get_text("\n").splitlines()
            if (t := _clean(line)) != EMPTY]


def _table_after_heading(soup: BeautifulSoup, heading_re: str) -> Any:
    """Encontra a primeira tabela após um h2/h3 cujo texto casa com heading_re."""
    for h in soup.find_all(["h2", "h3"]):
        if re.search(heading_re, h.get_text(strip=True), re.I):
            return h.find_next("table")
    return None


def _is_vigente(data_fim_str: str) -> bool:
    """Vigente = sem data final OU data final >= hoje."""
    d = _parse_date(data_fim_str)
    return d is None or d >= date.today()


def _is_situacao_ativa(situacao: str) -> bool:
    """True se a situação funcional do servidor for ativa atualmente."""
    s = situacao.lower()
    return not any(t in s for t in INACTIVE_SITUACOES)


# ─────────────────────────────────────────────────────────────────────────────
# Extrator: CURSO
# ─────────────────────────────────────────────────────────────────────────────

def _extract_curso_rodape(soup: BeautifulSoup) -> dict[str, Any]:
    """
    Extrai o rodapé (div.conteudo-rodape / #notas) do curso:
      "(*) Conceitos de curso:"   → lista de itens/links
      "(**) Formas de ingresso:"  → dict {sigla: descrição}
    """
    rodape = soup.find(class_="conteudo-rodape")
    if not rodape:
        return {"conceitos_curso": [EMPTY], "formas_ingresso": {"info": EMPTY}}

    conceitos: list[str] = []
    formas: dict[str, str] = {}
    current: str | None = None

    for el in rodape.find_all(["p", "ul", "div"], recursive=False):
        if el.name == "p":
            texto = el.get_text(strip=True).lower()
            if "conceito" in texto:
                current = "conceitos"
            elif "ingresso" in texto:
                current = "ingresso"
            else:
                current = None
        elif current == "conceitos":
            conceitos.extend(_extract_list_items(el) or [_soup_text(el)])
        elif current == "ingresso":
            table = el if el.name == "table" else el.find("table")
            if table:
                for tr in _direct_rows(table):
                    tds = tr.find_all("td")
                    if len(tds) >= 2:
                        sigla = _clean(tds[0].get_text(strip=True))
                        desc  = _clean(tds[1].get_text(strip=True))
                        if sigla != EMPTY and sigla not in formas:
                            formas[sigla] = desc
            else:
                items = _extract_list_items(el)
                for i, item in enumerate(items):
                    formas[f"item_{i + 1}"] = item

    return {
        "conceitos_curso":  conceitos or [EMPTY],
        "formas_ingresso":  formas or {"info": EMPTY},
    }


def _extract_curso_vagas(soup: BeautifulSoup, ficha_el: Any = None) -> list[dict[str, str]]:
    """
    Extrai a tabela de vagas por forma de ingresso que segue o label
    "Vagas e Formas de Ingresso (**)" na ficha-dados (table.tabela-fixed).
    """
    vagas: list[dict[str, str]] = []
    ficha = soup.find(class_="ficha-dados")
    if not ficha:
        return vagas
    for label_el in ficha.find_all(class_="ficha-label"):
        if "vagas" not in label_el.get_text(strip=True).lower():
            continue
        table = label_el.find_next_sibling("table") or label_el.find_next("table")
        if not table:
            continue
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        headers = [h if h else "Processo" for h in headers]
        for tr in _direct_rows(table)[1:]:
            cells = tr.find_all("td")
            if not cells:
                continue
            row: dict[str, str] = {"processo": _clean(cells[0].get_text(strip=True))}
            for i, cell in enumerate(cells[1:], start=1):
                if i < len(headers):
                    valor = cell.get_text(strip=True)
                    if valor:
                        row[headers[i]] = valor
            vagas.append(row)
        break
    return vagas


def _extract_curso(soup: BeautifulSoup, url: str, codigo: str) -> dict:
    """
    Extrai todos os dados de um curso alvo:
      - ficha-dados (dados gerais)
      - #informacoes: todas as subabas/accordions
      - #curriculo: matriz curricular com URL de cada disciplina
      - #professores: nome, unidade e URL de cada professor
      - #turmas: turmas ofertadas
      - rodapé: conceitos de curso + formas de ingresso + vagas
    """
    _strip_noise(soup)
    ficha      = _extract_ficha(soup)
    accordions = _extract_accordions(soup)

    alvo = TARGET_CURSOS.get(codigo, {})
    nome = _first_valid(
        _ficha_get(ficha, "Nome do Curso /Conceitos (*)", "Nome do Curso", "Nome"),
        alvo.get("nome", ""),
        _soup_text(soup.find("h1")),
    )

    # ── Matriz curricular — #curriculo (accordion por semestre) ─────────────
    matriz: list[dict] = []
    curriculo_sec = soup.find(id="curriculo")
    if curriculo_sec:
        for acc_h3 in curriculo_sec.find_all("h3", class_="cor-fundo"):
            sem_label = acc_h3.get_text(strip=True)
            content_div = acc_h3.find_next_sibling(attrs={"data-content": True}) \
                or acc_h3.find_next_sibling("div")
            if not content_div:
                continue
            table = content_div.find("table")
            if not table:
                continue
            headers = [th.get_text(strip=True) for th in table.find_all("th")]
            for tr in _direct_rows(table)[1:]:
                cells = tr.find_all("td")
                if not cells:
                    continue
                row: dict = {"semestre": sem_label}
                for i, cell in enumerate(cells):
                    if i >= len(headers):
                        break
                    header = headers[i]
                    link = cell.find("a", href=re.compile(r"/disciplinas/"))
                    if link:
                        row[header] = _clean(link.get_text(strip=True))
                        row[header + "_url"] = _normalize_url(urljoin(url, link["href"]))
                    else:
                        row[header] = _clean(cell.get_text(strip=True))
                if any(v not in (EMPTY, "") for k, v in row.items() if k != "semestre"):
                    matriz.append(row)

    # ── Professores — #professores ──────────────────────────────────────────
    professores: list[dict[str, str]] = []
    prof_sec = soup.find(id="professores")
    if prof_sec:
        for td in prof_sec.find_all("td", class_="h-card"):
            link      = td.find("a", class_="u-url")
            nome_span = td.find(class_="p-name") or td.find(class_="tabela-detalhe-titulo")
            unit_span = td.find(class_="p-org") or td.find(class_="tabela-detalhe-info")
            prof_nome = _clean(nome_span.get_text(strip=True)) if nome_span else EMPTY
            prof_unit = _clean(unit_span.get_text(strip=True)) if unit_span else EMPTY
            prof_url  = _normalize_url(urljoin(url, link["href"])) \
                if link and link.get("href") else ""
            if prof_nome != EMPTY:
                professores.append({
                    "nome":    prof_nome,
                    "unidade": prof_unit,
                    "url":     prof_url,
                })

    # ── Turmas ofertadas — #turmas ──────────────────────────────────────────
    turmas: list[dict] = []
    turmas_sec = soup.find(id="turmas")
    if turmas_sec:
        for table in turmas_sec.find_all("table", class_=re.compile(r"tabela-dados")):
            for tr in _direct_rows(table)[1:]:
                cells = tr.find_all("td", recursive=False)
                if len(cells) < 2:
                    continue
                disc_cell = cells[0]
                disc_link = disc_cell.find("a", href=re.compile(r"/disciplinas/"))
                if not disc_link:
                    continue
                disc_nome = _clean(disc_link.get_text(strip=True))
                disc_url  = _normalize_url(urljoin(url, disc_link["href"]))
                profs_turma = [
                    _clean(sp.get_text(strip=True)
                           .replace("Professor responsável pela turma:", "")
                           .replace("Professor Regente:", "")
                           .strip())
                    for sp in disc_cell.find_all("span", class_="tabela-detalhe-info")
                    if sp.get_text(strip=True)
                ]
                horarios = _parse_horarios(disc_cell)
                last3 = cells[-3:]
                turmas.append({
                    "disciplina":     disc_nome,
                    "disciplina_url": disc_url,
                    "professores":    profs_turma,
                    "horarios":       horarios or [EMPTY],
                    "turma":          _clean(last3[0].get_text(strip=True)) if len(last3) > 0 else EMPTY,
                    "vagas":          _clean(last3[1].get_text(strip=True)) if len(last3) > 1 else EMPTY,
                    "matriculados":   _clean(last3[2].get_text(strip=True)) if len(last3) > 2 else EMPTY,
                })

    rodape = _extract_curso_rodape(soup)
    vagas_ingresso = _extract_curso_vagas(soup)

    dados = {
        "nome":                   _clean(nome),
        "codigo_ufpel":           _first_valid(_ficha_get(ficha, "Código UFPel", "Código"), codigo),
        "nivel":                  _first_valid(_ficha_get(ficha, "Nível / Grau", "Nível"), alvo.get("nivel", "")),
        "grau":                   alvo.get("grau", EMPTY),
        "modalidade":             _ficha_get(ficha, "Modalidade"),
        "turno":                  _first_valid(_ficha_get(ficha, "Turno"), alvo.get("turno", "")),
        "codigo_emec":            _ficha_get(ficha, "Código e-MEC"),
        "codigo_capes":           _ficha_get(ficha, "Código CAPES"),
        "unidade":                _ficha_get(ficha, "Unidade"),
        "programa":               _ficha_get(ficha, "Programa"),
        "coordenador":            _ficha_get(ficha, "Coordenador"),
        "criacao_reconhecimento": _ficha_get(ficha, "Criação e Reconhecimento"),
        "informacoes":            accordions or {"info": EMPTY},
        "matriz_curricular":      matriz,
        "professores":            professores,
        "turmas_ofertadas":       turmas,
        "conceitos_curso":        rodape["conceitos_curso"],
        "formas_ingresso":        rodape["formas_ingresso"],
        "vagas_por_ingresso":     vagas_ingresso or [EMPTY],
    }

    # ── embedding_text: resumo do curso (subabas descritivas) ───────────────
    # Graduação usa "Contextualização"; pós-graduação usa "Apresentação"
    contextualizacao = _first_valid(
        accordions.get("Contextualização", ""), accordions.get("Apresentação", ""),
    )
    objetivos        = _first_valid(accordions.get("Objetivos", ""), accordions.get("Objetivo", ""))
    perfil_egresso   = accordions.get("Perfil do Egresso", EMPTY)
    area_conc        = accordions.get("Área de Concentração", EMPTY)
    linhas_pesquisa  = accordions.get("Linhas de Pesquisa", EMPTY)

    partes = [
        f"Curso: {dados['nome']} ({dados['grau']}) — UFPel",
        f"Nível: {dados['nivel']} | Modalidade: {dados['modalidade']} | Turno: {dados['turno']}",
        f"Unidade responsável: {dados['unidade']}",
        f"Coordenador: {dados['coordenador']}",
    ]
    if dados["programa"] != EMPTY:
        partes.append(f"Programa: {dados['programa']}")
    if contextualizacao != EMPTY:
        partes.append(f"\nSobre o curso:\n{contextualizacao}")
    if objetivos != EMPTY:
        partes.append(f"\nObjetivos:\n{objetivos}")
    if perfil_egresso != EMPTY:
        partes.append(f"\nPerfil do egresso:\n{perfil_egresso}")
    if area_conc != EMPTY:
        partes.append(f"\nÁrea de concentração:\n{area_conc}")
    if linhas_pesquisa != EMPTY:
        partes.append(f"\nLinhas de pesquisa:\n{linhas_pesquisa}")

    # Nível da ficha já inclui o grau (ex.: "Graduação / Bacharelado",
    # "Pós-Graduação / DOUTORADO") — diferencia Mestrado de Doutorado no título
    return {
        "titulo":          f"{dados['nome']} ({dados['nivel']})",
        "embedding_text":  "\n".join(partes),
        "metadata": {
            "url":          url,
            "codigo_ufpel": dados["codigo_ufpel"],
            "nivel":        dados["nivel"],
            "grau":         dados["grau"],
            "modalidade":   dados["modalidade"],
            "turno":        dados["turno"],
            "unidade":      dados["unidade"],
            "crawled_at":   _now(),
        },
        "dados_completos": dados,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extrator: DISCIPLINA
# ─────────────────────────────────────────────────────────────────────────────

def _extract_disciplina(soup: BeautifulSoup, url: str,
                        cursos_rel: list[str] | None = None) -> dict:
    """
    Extrai dados de uma disciplina:
      - ficha-dados (informações gerais: código, carga horária, créditos, ...)
      - #informacoes: Ementa, Objetivos, Conteúdo Programático, Bibliografia
      - Turmas Ofertadas (tabela após heading, com horários e professores)
    """
    _strip_noise(soup)
    ficha      = _extract_ficha(soup)
    accordions = _extract_accordions(soup)

    nome = _first_valid(
        _ficha_get(ficha, "Nome da Atividade", "Nome", "Disciplina"),
        _soup_text(soup.find("h1")),
    )
    codigo = _ficha_get(ficha, "CÓDIGO", "Código", "Código da Disciplina")

    ementa    = accordions.pop("Ementa", EMPTY)
    objetivos = _first_valid(accordions.pop("Objetivos", ""), accordions.pop("Objetivo", ""))
    conteudo  = accordions.pop("Conteúdo Programático", EMPTY)
    biblio    = accordions.pop("Bibliografia", EMPTY)

    # ── Turmas ofertadas — tabela após o heading "Turmas Ofertadas" ─────────
    turmas: list[dict] = []
    turmas_table = _table_after_heading(soup, r"turmas\s+ofertadas")
    if turmas_table:
        headers = [th.get_text(strip=True)
                   for th in turmas_table.find_all("th")
                   if th.find_parent("table") is turmas_table]
        for tr in _direct_rows(turmas_table)[1:]:
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 4:
                continue
            curso_cell = cells[4] if len(cells) > 4 else None
            prof_cell  = cells[5] if len(cells) > 5 else None
            curso_link = curso_cell.find("a", href=re.compile(r"/cursos/")) if curso_cell else None
            professores_turma = []
            if prof_cell:
                professores_turma = [
                    _clean(t) for t in prof_cell.stripped_strings
                    if t and "responsável" not in t.lower() and "regente" not in t.lower()
                ]
            turmas.append({
                "turma":        _clean(cells[0].get_text(strip=True)),
                "periodo":      _clean(cells[1].get_text(strip=True)),
                "vagas":        _clean(cells[2].get_text(strip=True)),
                "matriculados": _clean(cells[3].get_text(strip=True)),
                "curso":        _clean(curso_link.get_text(strip=True)) if curso_link
                                else (_clean(next(iter(curso_cell.stripped_strings), ""))
                                      if curso_cell else EMPTY),
                "horarios":     _parse_horarios(curso_cell) if curso_cell else [EMPTY],
                "professores":  professores_turma or [EMPTY],
            })

    dados = {
        "codigo":                codigo,
        "nome":                  _clean(nome),
        "tipo_atividade":        _ficha_get(ficha, "Tipo de Atividade"),
        "periodicidade":         _ficha_get(ficha, "Periodicidade"),
        "creditos":              _ficha_get(ficha, "CRÉDITOS", "Créditos"),
        "carga_horaria":         _ficha_get(ficha, "Carga Horária"),
        "ch_teorica":            _ficha_get(ficha, "CARGA HORÁRIA TEÓRICA"),
        "ch_pratica":            _ficha_get(ficha, "CARGA HORÁRIA PRÁTICA"),
        "ch_obrigatoria":        _ficha_get(ficha, "CARGA HORÁRIA OBRIGATÓRIA"),
        "freq_aprovacao":        _ficha_get(ficha, "FREQUÊNCIA APROVAÇÃO"),
        "unidade_responsavel":   _ficha_get(ficha, "Unidade responsável", "Unidade"),
        "ementa":                ementa,
        "objetivos":             objetivos,
        "conteudo_programatico": conteudo,
        "bibliografia":          biblio,
        "turmas_ofertadas":      turmas,
        "cursos_relacionados":   cursos_rel or [],
        **accordions,
    }

    # embedding: ementa é o "resumo" da disciplina
    partes = [f"Disciplina: {dados['nome']} (código {codigo}) — UFPel"]
    if ementa != EMPTY:
        partes.append(f"Ementa: {ementa}")
    if objetivos != EMPTY:
        partes.append(f"Objetivos: {objetivos}")
    if cursos_rel:
        partes.append(f"Cursos: {', '.join(cursos_rel)}")

    return {
        "titulo":          dados["nome"],
        "embedding_text":  "\n".join(partes),
        "metadata": {
            "url":        url,
            "codigo":     codigo,
            "unidade":    dados["unidade_responsavel"],
            "cursos":     ", ".join(cursos_rel or []),
            "crawled_at": _now(),
        },
        "dados_completos": dados,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extrator: PROFESSOR (SERVIDOR)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_lattes(soup: BeautifulSoup) -> dict[str, Any]:
    """
    Extrai o currículo (#lattes): Resumo, Formação acadêmica, Áreas de
    atuação e demais seções, cada uma sob um <h3>.
    """
    resultado: dict[str, Any] = {
        "resumo":            EMPTY,
        "formacao_academica": [EMPTY],
        "areas_atuacao":     [EMPTY],
        "lattes_url":        EMPTY,
    }
    sec = soup.find(id="lattes")
    if not sec:
        return resultado

    link = sec.find("a", href=re.compile(r"lattes\.cnpq", re.I))
    if link:
        resultado["lattes_url"] = link["href"]

    for h3 in sec.find_all("h3"):
        label = h3.get_text(strip=True).lower()
        textos: list[str] = []
        itens:  list[str] = []
        for sib in h3.next_siblings:
            if getattr(sib, "name", None) == "h3":
                break
            if hasattr(sib, "get_text"):
                t = _clean(sib.get_text(" ", strip=True))
                if t not in (EMPTY,) and "extraídas do lattes" not in t.lower():
                    textos.append(t)
                    itens.extend(x for x in _split_lines(sib)
                                 if "extraídas do lattes" not in x.lower())
            elif isinstance(sib, str) and sib.strip():
                textos.append(_clean(sib))
                itens.append(_clean(sib))

        if not textos:
            continue
        if "resumo" in label:
            resultado["resumo"] = " ".join(textos)
        elif "formação" in label or "formacao" in label:
            resultado["formacao_academica"] = itens or textos
        elif "área" in label or "area" in label or "atuação" in label:
            resultado["areas_atuacao"] = itens or textos
        else:
            resultado[h3.get_text(strip=True)] = " ".join(textos)

    return resultado


def _extract_servidor_projetos(soup: BeautifulSoup, url: str) -> list[dict]:
    """
    Extrai os projetos do servidor (#projetos), mantendo APENAS os ativos
    (data final vazia ou >= hoje). Cada tabela vem precedida de um heading
    com a ênfase (Extensão, Pesquisa, Ensino).
    """
    sec = soup.find(id="projetos")
    if not sec:
        return []

    projetos: dict[str, dict] = {}
    for table in sec.find_all("table"):
        # A ênfase (Extensão / Pesquisa / Ensino) é o primeiro <th> da tabela
        primeiro_th = table.find("th")
        enfase = _clean(primeiro_th.get_text(strip=True)) if primeiro_th else EMPTY
        for tr in _direct_rows(table)[1:]:
            cells = tr.find_all("td", recursive=False)
            if not cells:
                continue
            link = cells[0].find("a", href=re.compile(r"/projetos/"))
            if not link:
                continue
            titulo   = _clean(link.get_text(strip=True))
            proj_url = _normalize_url(urljoin(url, link["href"]))
            inicio   = _clean(cells[1].get_text(strip=True)) if len(cells) > 1 else EMPTY
            fim      = _clean(cells[2].get_text(strip=True)) if len(cells) > 2 else EMPTY
            ch       = _clean(cells[3].get_text(strip=True)) if len(cells) > 3 else EMPTY

            if not _is_vigente(fim if fim != EMPTY else ""):
                continue

            atual = projetos.get(proj_url)
            novo = {
                "titulo":      titulo,
                "url":         proj_url,
                "enfase":      enfase,
                "data_inicio": inicio,
                "data_fim":    fim,
                "ch_semanal":  ch,
            }
            # Linhas duplicadas por vínculo: mantém a com mais informação (CH)
            if atual is None or (atual.get("ch_semanal") == EMPTY and ch != EMPTY):
                projetos[proj_url] = novo

    return list(projetos.values())


def _extract_servidor_disciplinas(soup: BeautifulSoup, url: str) -> list[dict]:
    """
    Extrai as disciplinas ministradas (#disciplinas) nos últimos semestres:
    Ano/Semestre, Turma, Disciplina (com URL), CH, Curso e horários.
    """
    sec = soup.find(id="disciplinas")
    if not sec:
        return []

    ministradas: list[dict] = []
    for table in sec.find_all("table"):
        if table.find_parent("table"):
            continue  # tabelas de horários aninhadas
        for tr in _direct_rows(table)[1:]:
            cells = tr.find_all("td", recursive=False)
            if len(cells) < 3:
                continue
            disc_cell = next((c for c in cells if c.find("a", href=re.compile(r"/disciplinas/"))), None)
            if not disc_cell:
                continue
            disc_link  = disc_cell.find("a", href=re.compile(r"/disciplinas/"))
            curso_cell = next((c for c in cells if c.find("a", href=re.compile(r"/cursos/"))), None)
            curso_link = curso_cell.find("a", href=re.compile(r"/cursos/")) if curso_cell else None
            ministradas.append({
                "ano_semestre":   _clean(cells[0].get_text(strip=True)),
                "turma":          _clean(cells[1].get_text(strip=True)),
                "disciplina":     _clean(disc_link.get_text(strip=True)),
                "disciplina_url": _normalize_url(urljoin(url, disc_link["href"])),
                "horarios":       _parse_horarios(disc_cell) or [EMPTY],
                "carga_horaria":  _clean(cells[-2].get_text(strip=True)) if len(cells) >= 2 else EMPTY,
                "curso":          _clean(curso_link.get_text(strip=True)) if curso_link else EMPTY,
            })
    return ministradas


def _extract_servidor(soup: BeautifulSoup, url: str,
                      cursos_rel: list[str] | None = None) -> dict | None:
    """
    Extrai dados de um professor/servidor. Retorna None se a situação
    funcional NÃO estiver ativa (registros antigos de vínculos encerrados).
    """
    _strip_noise(soup)
    ficha = _extract_ficha(soup)

    nome = _first_valid(
        _ficha_get(ficha, "Nome do Servidor", "Nome"),
        _soup_text(soup.find("h1")),
    )
    situacao = _ficha_get(ficha, "Situação")

    if situacao != EMPTY and not _is_situacao_ativa(situacao):
        log.debug("Servidor com situação inativa (%s) — ignorado: %s", situacao, nome)
        return None

    lattes      = _extract_lattes(soup)
    projetos    = _extract_servidor_projetos(soup, url)
    ministradas = _extract_servidor_disciplinas(soup, url)

    dados = {
        "nome":                 _clean(nome),
        "matricula_siape":      _ficha_get(ficha, "Matrícula SIAPE", "Matrícula"),
        "categoria":            _ficha_get(ficha, "Categoria"),
        "cargo":                _ficha_get(ficha, "Cargo", "Função"),
        "classe_nivel":         _ficha_get(ficha, "Classe / Nível"),
        "titulacao":            _ficha_get(ficha, "Titulação"),
        "lotacao":              _ficha_get(ficha, "Lotação", "Unidade"),
        "regime_jornada":       _ficha_get(ficha, "Regime / Jornada de Trabalho", "Regime de Trabalho"),
        "situacao":             situacao,
        "data_ingresso_servico": _ficha_get(ficha, "Data de ingresso no serviço público"),
        "data_ingresso_ufpel":  _ficha_get(ficha, "Data de ingresso na UFPel"),
        "data_ingresso_cargo":  _ficha_get(ficha, "Data de ingresso no cargo"),
        "email":                _ficha_get(ficha, "E-mail", "Contato"),
        "curriculo_resumo":     lattes["resumo"],
        "formacao_academica":   lattes["formacao_academica"],
        "areas_atuacao":        lattes["areas_atuacao"],
        "lattes_url":           lattes["lattes_url"],
        "projetos_ativos":      projetos or [EMPTY],
        "disciplinas_ministradas": ministradas or [EMPTY],
        "cursos_relacionados":  cursos_rel or [],
    }

    # embedding: resumo do currículo + áreas de atuação
    partes = [f"Professor(a): {dados['nome']} — UFPel"]
    if dados["cargo"] != EMPTY:
        partes.append(f"Cargo: {dados['cargo']} | Titulação: {dados['titulacao']}")
    if dados["lotacao"] != EMPTY:
        partes.append(f"Lotação: {dados['lotacao']}")
    if lattes["resumo"] != EMPTY:
        partes.append(f"\nResumo do currículo:\n{lattes['resumo']}")
    if lattes["areas_atuacao"] != [EMPTY]:
        partes.append("\nÁreas de atuação:\n" +
                      "\n".join(f"  - {a}" for a in lattes["areas_atuacao"]))
    if cursos_rel:
        partes.append(f"\nProfessor(a) dos cursos: {', '.join(cursos_rel)}")

    return {
        "titulo":          dados["nome"],
        "embedding_text":  "\n".join(partes),
        "metadata": {
            "url":        url,
            "matricula":  dados["matricula_siape"],
            "cargo":      dados["cargo"],
            "titulacao":  dados["titulacao"],
            "lotacao":    dados["lotacao"],
            "situacao":   situacao,
            "cursos":     ", ".join(cursos_rel or []),
            "crawled_at": _now(),
        },
        "dados_completos": dados,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Extrator: PROJETO
# ─────────────────────────────────────────────────────────────────────────────

def _extract_projeto(soup: BeautifulSoup, url: str,
                     professores_rel: list[str] | None = None) -> dict | None:
    """
    Extrai dados de um projeto. Retorna None se o projeto NÃO estiver ativo
    (data final < hoje).

    Captura:
      - ficha-dados (informações gerais: nome, ênfase, datas, coordenador...)
      - #informacoes: accordions (Objetivo Geral, etc.)
      - #equipe: apenas membros vigentes (data final vazia ou >= hoje)
      - #financeiro: dados financeiros ou "Não há informações disponíveis"
    """
    _strip_noise(soup)
    ficha = _extract_ficha(soup)

    titulo = _first_valid(
        _ficha_get(ficha, "Nome do Projeto", "Título", "Nome"),
        _soup_text(soup.find("h1")),
    )

    # "Data inicial - Data final" vem num único campo
    periodo = _ficha_get(ficha, "Data inicial - Data final")
    datas   = re.findall(r"\d{2}/\d{2}/\d{4}", periodo)
    data_inicio = datas[0] if datas else _ficha_get(ficha, "Data de Início", "Início")
    data_fim    = datas[1] if len(datas) > 1 else _ficha_get(ficha, "Data de Término", "Fim")

    if data_fim != EMPTY and not _is_vigente(data_fim):
        log.debug("Projeto não ativo (fim=%s) — ignorado: %s", data_fim, titulo[:60])
        return None

    resumo     = _ficha_get(ficha, "Resumo")
    accordions = _extract_accordions(soup)

    # ── Equipe: apenas membros vigentes ─────────────────────────────────────
    equipe: list[dict] = []
    equipe_sec = soup.find(id="equipe")
    if equipe_sec:
        vistos: set[str] = set()
        for table in equipe_sec.find_all("table"):
            for tr in _direct_rows(table)[1:]:
                cells = tr.find_all("td", recursive=False)
                if not cells:
                    continue
                nome_membro = _clean(cells[0].get_text(strip=True))
                if nome_membro == EMPTY:
                    continue
                ch        = _clean(cells[1].get_text(strip=True)) if len(cells) > 1 else EMPTY
                d_inicio  = _clean(cells[2].get_text(strip=True)) if len(cells) > 2 else EMPTY
                d_fim     = _clean(cells[3].get_text(strip=True)) if len(cells) > 3 else EMPTY
                if not _is_vigente(d_fim if d_fim != EMPTY else ""):
                    continue
                link = cells[0].find("a", href=re.compile(r"/servidores/"))
                chave = nome_membro.lower()
                membro = {
                    "nome":        nome_membro,
                    "url":         _normalize_url(urljoin(url, link["href"])) if link else "",
                    "ch_semanal":  ch,
                    "data_inicio": d_inicio,
                    "data_fim":    d_fim,
                }
                if chave not in vistos:
                    vistos.add(chave)
                    equipe.append(membro)
                elif ch != EMPTY:
                    # linha duplicada com mais informação substitui a anterior
                    for i, m in enumerate(equipe):
                        if m["nome"].lower() == chave and m["ch_semanal"] == EMPTY:
                            equipe[i] = membro
                            break

    # ── Financeiro ──────────────────────────────────────────────────────────
    financeiro: dict[str, Any] = {"info": EMPTY}
    fin_sec = soup.find(id="financeiro")
    if fin_sec:
        fin_ficha  = _extract_ficha(fin_sec)
        fin_tables = [r for t in fin_sec.find_all("table") for r in _extract_table_rows(t)]
        fin_texto  = _soup_text(fin_sec)
        if fin_ficha:
            financeiro = fin_ficha
        elif fin_tables:
            financeiro = {"registros": fin_tables}
        elif fin_texto != EMPTY:
            financeiro = {"info": fin_texto}

    skip_keys = {
        "Nome do Projeto", "Título", "Nome", "Resumo",
        "Data inicial - Data final", "Data de Início", "Início",
        "Data de Término", "Fim", "Coordenador Atual", "Coordenador",
        "Ênfase", "Área CNPq", "Unidade de Origem",
        "Eixo Temático (Principal - Afim)", "Linha de Extensão",
    }
    dados = {
        "titulo":          _clean(titulo),
        "resumo":          resumo,
        "enfase":          _ficha_get(ficha, "Ênfase"),
        "data_inicio":     data_inicio,
        "data_fim":        data_fim,
        "situacao":        "Ativo",
        "coordenador":     _ficha_get(ficha, "Coordenador Atual", "Coordenador"),
        "unidade_origem":  _ficha_get(ficha, "Unidade de Origem", "Unidade"),
        "area_cnpq":       _ficha_get(ficha, "Área CNPq", "Área"),
        "eixo_tematico":   _ficha_get(ficha, "Eixo Temático (Principal - Afim)"),
        "linha_extensao":  _ficha_get(ficha, "Linha de Extensão"),
        "informacoes":     accordions or {"info": EMPTY},
        "equipe_vigente":  equipe or [EMPTY],
        "financeiro":      financeiro,
        "professores_relacionados": professores_rel or [],
        **{k: v for k, v in ficha.items() if k not in skip_keys},
    }

    # embedding: resumo do projeto
    partes = [f"Projeto ({dados['enfase']}): {dados['titulo']} — UFPel"]
    if resumo != EMPTY:
        partes.append(f"Resumo: {resumo}")
    partes.append(f"Coordenador: {dados['coordenador']} | Unidade: {dados['unidade_origem']}")

    return {
        "titulo":          dados["titulo"],
        "embedding_text":  "\n".join(partes),
        "metadata": {
            "url":         url,
            "enfase":      dados["enfase"],
            "data_inicio": data_inicio,
            "data_fim":    data_fim,
            "situacao":    "Ativo",
            "coordenador": dados["coordenador"],
            "crawled_at":  _now(),
        },
        "dados_completos": dados,
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
# Crawler principal (assíncrono) — deep-crawl dos cursos alvo
# ─────────────────────────────────────────────────────────────────────────────

class UFPelCrawler:
    """
    Deep-crawler dos cursos de Computação da UFPel.

    Estratégia RAG:
      1. Busca vetorial (embedding_text = resumo de cada registro)
      2. Recuperação dos documentos mais relevantes pelo doc_id
      3. Consulta SQL em doc_completos (JSONB) para os dados estruturados
      4. Envio ao LLM para síntese da resposta com contexto completo
    """

    def __init__(
        self,
        concurrency: int = DEFAULT_CONCURRENCY,
        delay: float = DEFAULT_DELAY,
        cursos: list[str] | None = None,
    ):
        self._concurrency = concurrency
        self._delay       = delay
        self._cursos      = cursos or list(TARGET_CURSOS.keys())

    # ── Fase 1: cursos ───────────────────────────────────────────────────────
    async def _crawl_cursos(
        self, client: RateLimitedClient,
    ) -> tuple[list[dict], dict[str, set[str]], dict[str, set[str]]]:
        """
        Retorna (docs, disciplina_urls→cursos, professor_urls→cursos).
        Os dicts mapeiam cada URL para o conjunto de nomes de cursos que a
        referenciam (vira metadado 'cursos' nos docs de disciplina/professor).
        """
        docs: list[dict] = []
        disc_urls: dict[str, set[str]] = {}
        prof_urls: dict[str, set[str]] = {}

        async def _one(codigo: str) -> None:
            url  = f"{BASE_URL}/cursos/cod/{codigo}"
            soup = await client.fetch(url)
            if soup is None:
                log.error("[CURSO %s] Página indisponível.", codigo)
                return
            try:
                extracted = _extract_curso(soup, url, codigo)
            except Exception as exc:
                log.error("[CURSO %s] Erro na extração: %s", codigo, exc)
                return
            doc = _build_doc("curso", extracted)
            docs.append(doc)

            rotulo = doc["titulo"]
            dados  = doc["dados_completos"]
            for row in dados.get("matriz_curricular", []):
                for k, v in row.items():
                    if k.endswith("_url") and "/disciplinas/" in v:
                        disc_urls.setdefault(v, set()).add(rotulo)
            for turma in dados.get("turmas_ofertadas", []):
                u = turma.get("disciplina_url", "")
                if u:
                    disc_urls.setdefault(u, set()).add(rotulo)
            for prof in dados.get("professores", []):
                u = prof.get("url", "")
                if u:
                    prof_urls.setdefault(u, set()).add(rotulo)

            log.info("[CURSO %s] %s — %d disciplinas na matriz, %d professores, %d turmas",
                     codigo, rotulo,
                     len(dados.get("matriz_curricular", [])),
                     len(dados.get("professores", [])),
                     len(dados.get("turmas_ofertadas", [])))

        await asyncio.gather(*[_one(c) for c in self._cursos])
        return docs, disc_urls, prof_urls

    # ── Fase 2: disciplinas ─────────────────────────────────────────────────
    async def _crawl_disciplinas(
        self, client: RateLimitedClient, disc_urls: dict[str, set[str]],
    ) -> list[dict]:
        docs: list[dict] = []
        vistos_codigo: set[str] = set()

        async def _one(url: str, cursos_rel: list[str]) -> None:
            soup = await client.fetch(url)
            if soup is None:
                return
            try:
                extracted = _extract_disciplina(soup, url, cursos_rel)
            except Exception as exc:
                log.error("[DISCIPLINA] Erro em %s: %s", url[:70], exc)
                return
            codigo = extracted["metadata"].get("codigo", "")
            if codigo != EMPTY and codigo in vistos_codigo:
                return
            vistos_codigo.add(codigo)
            docs.append(_build_doc("disciplina", extracted))

        await asyncio.gather(*[
            _one(url, sorted(cursos)) for url, cursos in sorted(disc_urls.items())
        ])
        return docs

    # ── Fase 3: professores ─────────────────────────────────────────────────
    async def _crawl_professores(
        self, client: RateLimitedClient, prof_urls: dict[str, set[str]],
    ) -> tuple[list[dict], dict[str, set[str]]]:
        """Retorna (docs, projeto_urls→professores)."""
        resultados: list[dict] = []
        proj_urls: dict[str, set[str]] = {}

        async def _one(url: str, cursos_rel: list[str]) -> None:
            soup = await client.fetch(url)
            if soup is None:
                return
            try:
                extracted = _extract_servidor(soup, url, cursos_rel)
            except Exception as exc:
                log.error("[PROFESSOR] Erro em %s: %s", url[:70], exc)
                return
            if extracted is None:      # situação inativa
                return
            resultados.append(extracted)

        await asyncio.gather(*[
            _one(url, sorted(cursos)) for url, cursos in sorted(prof_urls.items())
        ])

        # Dedup por nome: o portal cria um registro por vínculo — mantém o
        # registro ativo com currículo mais completo.
        por_nome: dict[str, dict] = {}
        for ext in resultados:
            chave = ext["titulo"].lower()
            atual = por_nome.get(chave)
            if atual is None or len(ext["embedding_text"]) > len(atual["embedding_text"]):
                por_nome[chave] = ext

        docs: list[dict] = []
        for ext in por_nome.values():
            doc = _build_doc("servidor", ext)
            docs.append(doc)
            nome = doc["titulo"]
            for proj in doc["dados_completos"].get("projetos_ativos", []):
                if isinstance(proj, dict) and proj.get("url"):
                    proj_urls.setdefault(proj["url"], set()).add(nome)

        descartados = len(resultados) - len(por_nome)
        if descartados:
            log.info("[PROFESSORES] %d registro(s) duplicado(s) descartado(s).", descartados)
        return docs, proj_urls

    # ── Fase 4: projetos ────────────────────────────────────────────────────
    async def _crawl_projetos(
        self, client: RateLimitedClient, proj_urls: dict[str, set[str]],
    ) -> list[dict]:
        docs: list[dict] = []

        async def _one(url: str, profs_rel: list[str]) -> None:
            soup = await client.fetch(url)
            if soup is None:
                return
            try:
                extracted = _extract_projeto(soup, url, profs_rel)
            except Exception as exc:
                log.error("[PROJETO] Erro em %s: %s", url[:70], exc)
                return
            if extracted is None:      # projeto não ativo
                return
            docs.append(_build_doc("projeto", extracted))

        await asyncio.gather(*[
            _one(url, sorted(profs)) for url, profs in sorted(proj_urls.items())
        ])
        return docs

    # ── Orquestração ────────────────────────────────────────────────────────
    async def crawl(self) -> list[dict]:
        """Executa as 4 fases do deep-crawl e retorna todos os documentos."""
        log.info("[Crawler] Deep-crawl dos cursos de Computação — %s", _now())
        log.info("[Crawler] Cursos alvo: %s",
                 ", ".join(f"{c} ({TARGET_CURSOS[c]['nome']} — {TARGET_CURSOS[c]['grau']})"
                           for c in self._cursos if c in TARGET_CURSOS))

        async with RateLimitedClient(self._concurrency, self._delay) as client:
            # Fase 1 — cursos
            cursos_docs, disc_urls, prof_urls = await self._crawl_cursos(client)
            log.info("[Fase 1] %d cursos | %d disciplinas únicas | %d professores únicos",
                     len(cursos_docs), len(disc_urls), len(prof_urls))

            # Fase 2 — disciplinas
            disc_docs = await self._crawl_disciplinas(client, disc_urls)
            log.info("[Fase 2] %d disciplinas capturadas", len(disc_docs))

            # Fase 3 — professores (apenas situação ativa)
            prof_docs, proj_urls = await self._crawl_professores(client, prof_urls)
            log.info("[Fase 3] %d professores ativos | %d projetos ativos únicos",
                     len(prof_docs), len(proj_urls))

            # Fase 4 — projetos ativos
            proj_docs = await self._crawl_projetos(client, proj_urls)
            log.info("[Fase 4] %d projetos ativos capturados", len(proj_docs))

            log.info("[HTTP] %s", client.stats)

        return cursos_docs + disc_docs + prof_docs + proj_docs


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
        description="Deep-crawler dos cursos de Computação do Portal UFPel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output",      default=DEFAULT_OUTPUT, metavar="FILE",
                   help="Arquivo JSON de saída")
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, metavar="N",
                   help="Requisições simultâneas")
    p.add_argument("--delay",       type=float, default=DEFAULT_DELAY, metavar="SECS",
                   help="Delay (s) entre requests por worker")
    p.add_argument("--cursos",      nargs="+", default=None, metavar="COD",
                   choices=list(TARGET_CURSOS.keys()),
                   help="Subconjunto dos códigos de curso alvo "
                        f"(padrão: todos — {' '.join(TARGET_CURSOS)})")
    return p


async def _amain(args: argparse.Namespace) -> None:
    crawler = UFPelCrawler(
        concurrency=args.concurrency,
        delay=args.delay,
        cursos=args.cursos,
    )
    t0   = datetime.now(timezone.utc)
    docs = await crawler.crawl()
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    log.info("[Crawler] Concluído em %.1fs — %d documentos.", elapsed, len(docs))
    save_json(docs, args.output)


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
