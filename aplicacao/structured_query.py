"""
Consultas Estruturadas ao doc_completos
=============================================================================
Para perguntas sobre dados tabulares de cursos (matriz curricular, professores,
turmas), acessa diretamente o banco de dados sem passar pelo pipeline RAG.

Custo: ~1 chamada LLM (síntese) + 1 query SQL  vs  RAG: embeddings + vector
search + LLM. Evita desperdício de tokens e latência desnecessária.

Tipos de consulta detectados:
  MATRIX_SEMESTRE   "disciplinas do Xº semestre do curso de Y"
  MATRIX_FULL       "disciplinas do curso de Y"
  PROFS_CURSO       "professores do curso de Y"
  TURMAS_CURSO      "turmas ofertadas no curso de Y"
  CREDITOS_FILTER   "disciplinas com 4 créditos no curso de Y"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import psycopg2

import config

EMPTY = "Não há informações disponíveis"

# Abreviações comuns de cursos da UFPel
_CURSO_ABBREVS: dict[str, str] = {
    "cc":   "Ciência da Computação",
    "ec":   "Engenharia de Computação",
    "ee":   "Engenharia Elétrica",
    "ea":   "Engenharia Agrícola",
    "med":  "Medicina",
    "enf":  "Enfermagem",
    "agro": "Agronomia",
    "vet":  "Medicina Veterinária",
    "dir":  "Direito",
    "ped":  "Pedagogia",
    "bio":  "Ciências Biológicas",
    "quim": "Química",
    "fis":  "Física",
    "mat":  "Matemática",
    "adm":  "Administração",
    "econ": "Ciências Econômicas",
}

# =============================================================================
# Classificação de consulta
# =============================================================================

@dataclass
class StructuredIntent:
    tipo: str           # MATRIX_SEMESTRE | MATRIX_FULL | PROFS_CURSO | TURMAS_CURSO | CREDITOS_FILTER
    curso_nome: str     # nome do curso extraído
    semestre: Optional[str] = None   # ex: "1º Semestre", "4º Semestre"
    creditos: Optional[int] = None   # ex: 4


# Padrões de número de semestre em linguagem natural
_SEM_MAP = {
    "1": "1º Semestre", "primeiro": "1º Semestre", "1°": "1º Semestre", "1º": "1º Semestre",
    "2": "2º Semestre", "segundo": "2º Semestre",  "2°": "2º Semestre", "2º": "2º Semestre",
    "3": "3º Semestre", "terceiro": "3º Semestre", "3°": "3º Semestre", "3º": "3º Semestre",
    "4": "4º Semestre", "quarto": "4º Semestre",   "4°": "4º Semestre", "4º": "4º Semestre",
    "5": "5º Semestre", "quinto": "5º Semestre",   "5°": "5º Semestre", "5º": "5º Semestre",
    "6": "6º Semestre", "sexto": "6º Semestre",    "6°": "6º Semestre", "6º": "6º Semestre",
    "7": "7º Semestre", "sétimo": "7º Semestre",   "7°": "7º Semestre", "7º": "7º Semestre",
    "8": "8º Semestre", "oitavo": "8º Semestre",   "8°": "8º Semestre", "8º": "8º Semestre",
    "9": "9º Semestre", "nono": "9º Semestre",     "9°": "9º Semestre", "9º": "9º Semestre",
    "10": "10º Semestre", "décimo": "10º Semestre",
}


def _extract_semestre(text: str) -> Optional[str]:
    """Extrai o semestre de uma string em linguagem natural."""
    # Ordinal com símbolo + "semestre" ou "sem." (abreviação)
    m = re.search(r"(\d{1,2})[°ºo]\s*sem(?:estre)?\.?", text, re.IGNORECASE)
    if m:
        return _SEM_MAP.get(m.group(1))
    # Por extenso: "primeiro semestre"
    m = re.search(
        r"(primeiro|segundo|terceiro|quarto|quinto|sexto|sétimo|oitavo|nono|décimo)\s+sem(?:estre)?\.?",
        text, re.IGNORECASE
    )
    if m:
        return _SEM_MAP.get(m.group(1).lower())
    # Número simples precedendo "semestre": "semestre 4"
    m = re.search(r"semestre\s+(\d{1,2})", text, re.IGNORECASE)
    if m:
        return _SEM_MAP.get(m.group(1))
    return None


def _extract_curso_name(text: str) -> Optional[str]:
    """Extrai o nome do curso de uma query em linguagem natural.

    Estratégia: captura tudo após "curso de" até marcadores de fim de cláusula.
    Palavras como "da/do/de" fazem parte de nomes de cursos (ex: "Ciência da
    Computação") e NÃO devem interromper a captura.
    """
    # Padrões de stop: início de oração relativa, frase nova ou indicadores de semestre/crédito
    _STOP = (
        r"(?:\?|$"
        r"|\s+(?:que\b|com\b|para\b|onde\b|quando\b|como\b"
        r"|possui[a-z]*\b|possuem\b|tem\b|têm\b"
        r"|no\s+\d|do\s+\d|\d+[°º]\s*sem))"  # "no 4° sem" "do 4° semestre"
    )

    # Prioritize "do curso de X" / "no curso de X" — mais específico
    m = re.search(r"\bcurso\s+de\s+(.+?)" + _STOP, text, re.IGNORECASE)
    if m:
        nome = m.group(1).strip().rstrip("?., ")
        # Remove prefixo de semestre caso tenha sido capturado
        nome = re.sub(r"^\d+[°º]\s*sem(?:estre)?\.\s*d[ae]\s*", "", nome, flags=re.IGNORECASE)
        return nome

    # "disciplinas/professores/turmas da/do/de/no CURSO_NAME"
    m = re.search(
        r"(?:disciplinas?|matérias?|grade|matriz|professores?|docentes?|turmas?)"
        r"[^?]*?\b(?:da|do|de|na|no)\s+(.+?)" + _STOP,
        text, re.IGNORECASE
    )
    if m:
        nome = m.group(1).strip().rstrip("?., ")
        # Remove prefixo de semestre/créditos capturado acidentalmente
        nome = re.sub(r"^\d+[°º]\s*sem(?:estre)?\.\s*d[ae]\s*", "", nome, flags=re.IGNORECASE)
        nome = re.sub(r"^(?:primeiro|segundo|terceiro|quarto|quinto|sexto|sétimo|oitavo)\s+semestre\s+d[ae]\s*", "", nome, flags=re.IGNORECASE)
        return nome

    return None


_STRUCT_TRIGGERS = re.compile(
    r"\b(?:disciplinas?|matérias?|grade\s+curricular|matriz\s+curricular"
    r"|professores?|docentes?|corpo\s+docente|turmas?\s+ofertadas?)\b",
    re.IGNORECASE,
)

_CURSO_TRIGGERS = re.compile(
    r"\b(?:curso\s+de|graduação\s+em|bacharelado\s+em|licenciatura\s+em|tecnólogo\s+em)\b",
    re.IGNORECASE,
)


def _expand_abbrevs(text: str) -> str:
    """Expande abreviações conhecidas de cursos no texto da query."""
    # Substitui apenas quando a abreviação está isolada (palavra inteira, maiúscula ou minúscula)
    for abbr, full in _CURSO_ABBREVS.items():
        text = re.sub(rf"\b{re.escape(abbr)}\b", full, text, flags=re.IGNORECASE)
    return text


def classify_query(query: str) -> Optional[StructuredIntent]:
    """
    Analisa a query e retorna um StructuredIntent se for uma consulta
    estrutural sobre um curso específico. Retorna None se deve usar RAG.
    """
    q = _expand_abbrevs(query.strip())
    q_lower = q.lower()

    # Deve mencionar pelo menos um dos triggers estruturados
    if not _STRUCT_TRIGGERS.search(q_lower):
        return None

    # Deve mencionar um curso (explicitamente ou implicitamente)
    curso_nome = _extract_curso_name(q_lower)
    if not curso_nome or len(curso_nome.split()) < 1:
        return None

    semestre = _extract_semestre(q_lower)
    creditos_m = re.search(r"(\d+)\s*cr[eé]ditos?", q_lower)
    creditos = int(creditos_m.group(1)) if creditos_m else None

    if re.search(r"\b(?:professores?|docentes?|corpo\s+docente)\b", q_lower):
        return StructuredIntent("PROFS_CURSO", curso_nome)

    if re.search(r"\bturmas?\b", q_lower):
        return StructuredIntent("TURMAS_CURSO", curso_nome)

    if semestre:
        if creditos:
            return StructuredIntent("CREDITOS_FILTER", curso_nome, semestre, creditos)
        return StructuredIntent("MATRIX_SEMESTRE", curso_nome, semestre)

    if creditos:
        return StructuredIntent("CREDITOS_FILTER", curso_nome, creditos=creditos)

    return StructuredIntent("MATRIX_FULL", curso_nome)


# =============================================================================
# Busca do curso no banco por similaridade de título
# =============================================================================

def _find_curso(nome: str) -> Optional[dict]:
    """
    Busca o curso mais similar ao nome fornecido no doc_completos.
    Retorna o registro (dict) ou None se não encontrar.

    Filtra por tipo='curso' para evitar retornar mestrado/doutorado
    quando o nome é ambíguo (ex: "Computação" pode ser bacharelado ou PPG).
    Prefere match exato de 'nivel' contendo 'Graduação' quando há empate.
    """
    conn = psycopg2.connect(**config.DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT doc_id, titulo, dados,
                   similarity(titulo, %(nome)s) AS sim
            FROM doc_completos
            WHERE tipo = 'curso'
              AND similarity(titulo, %(nome)s) > 0.3
            ORDER BY sim DESC
            LIMIT 5
        """, {"nome": nome.upper()})
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    best_sim = rows[0][3]
    candidates = [r for r in rows if r[3] >= best_sim - 0.05]

    # Dentre empatados, prefere graduação (bacharelado/licenciatura/tecnólogo)
    def _is_grad(dados: dict) -> bool:
        nivel = (dados.get("nivel") or "").lower()
        return any(k in nivel for k in ("graduação", "bacharelado", "licenciatura", "tecnólogo", "área básica"))

    for doc_id, titulo, dados, sim in candidates:
        if _is_grad(dados):
            return {"doc_id": doc_id, "titulo": titulo, "dados": dados, "sim": sim}

    # Sem preferência — retorna o de maior sim
    doc_id, titulo, dados, sim = rows[0]
    return {"doc_id": doc_id, "titulo": titulo, "dados": dados, "sim": sim}


# =============================================================================
# Extração de dados e formatação de contexto compacto
# =============================================================================

def _format_disciplinas(discs: list[dict], semestre_label: str = "") -> str:
    lines = []
    for d in discs:
        nome = d.get("Disciplina / Pré-requisitos", d.get("Disciplina", "?"))
        cod  = d.get("Código", "")
        cred = d.get("Cr.", d.get("créditos", ""))
        hor  = d.get("Horas", "")
        car  = d.get("Caráter", "")
        url  = d.get("Disciplina / Pré-requisitos_url", "")
        linha = f"  - {nome}"
        if cod:     linha += f" (cód. {cod})"
        if cred:    linha += f" | {cred} créditos"
        if hor:     linha += f" | {hor}h"
        if car:     linha += f" | {car}"
        if url:     linha += f" | {url}"
        lines.append(linha)
    return "\n".join(lines)


def _build_context_matrix(curso: dict, intent: StructuredIntent) -> str:
    dados   = curso["dados"]
    titulo  = curso["titulo"]
    nivel   = dados.get("nivel", "")
    matriz  = dados.get("matriz_curricular", [])
    url     = dados.get("url") or (dados.get("metadata") or {}).get("url") or ""

    header = f"Curso: {titulo} ({nivel})\nURL: {url}\n"

    if intent.tipo == "MATRIX_FULL":
        # Agrupa por semestre
        sem_map: dict[str, list] = {}
        for r in matriz:
            s = r.get("semestre", "Sem semestre")
            sem_map.setdefault(s, []).append(r)
        parts = [header, "=== MATRIZ CURRICULAR COMPLETA ==="]
        for sem, discs in sem_map.items():
            parts.append(f"\n{sem}:")
            parts.append(_format_disciplinas(discs))
        return "\n".join(parts)

    if intent.tipo == "MATRIX_SEMESTRE":
        discs = [r for r in matriz if r.get("semestre") == intent.semestre]
        return (
            header
            + f"=== DISCIPLINAS DO {intent.semestre.upper()} ===\n"
            + _format_disciplinas(discs)
        )

    if intent.tipo == "CREDITOS_FILTER":
        discs = [
            r for r in matriz
            if (intent.semestre is None or r.get("semestre") == intent.semestre)
            and str(r.get("Cr.", r.get("créditos", ""))) == str(intent.creditos)
        ]
        sem_label = f" do {intent.semestre}" if intent.semestre else ""
        cred_label = f" com {intent.creditos} créditos" if intent.creditos else ""
        header_line = f"=== DISCIPLINAS{sem_label.upper()}{cred_label.upper()} ==="

        if intent.semestre:
            # Semestre específico: lista plana
            return header + header_line + "\n" + _format_disciplinas(discs)
        else:
            # Sem semestre: agrupa por semestre para o LLM ter contexto claro
            sem_map: dict[str, list] = {}
            for r in discs:
                s = r.get("semestre", "Sem semestre")
                sem_map.setdefault(s, []).append(r)
            parts = [header, header_line]
            for sem, ds in sem_map.items():
                parts.append(f"\n{sem}:")
                parts.append(_format_disciplinas(ds))
            return "\n".join(parts)

    if intent.tipo == "PROFS_CURSO":
        profs = [
            p for p in dados.get("professores", [])
            if p.get("nome") and p["nome"] != EMPTY
        ]
        lines = [header, "=== CORPO DOCENTE ==="]
        for p in profs:
            linha = f"  - {p['nome']}"
            if p.get("unidade") and p["unidade"] != EMPTY:
                linha += f" ({p['unidade']})"
            if p.get("lattes"):
                linha += f" | Lattes: {p['lattes']}"
            elif p.get("url"):
                linha += f" | Perfil: {p['url']}"
            lines.append(linha)
        return "\n".join(lines)

    if intent.tipo == "TURMAS_CURSO":
        turmas = dados.get("turmas_ofertadas", [])
        lines = [header, "=== TURMAS OFERTADAS ==="]
        for t in turmas:
            disc = t.get("disciplina", "?")
            profs_t = ", ".join(t.get("professores", []))
            turma_id = t.get("turma", "")
            vagas    = t.get("vagas", "")
            matric   = t.get("matriculados", "")
            linha = f"  - {disc}"
            if profs_t: linha += f" | Prof: {profs_t}"
            if turma_id and turma_id != EMPTY: linha += f" | Turma {turma_id}"
            if vagas and vagas != EMPTY: linha += f" | Vagas: {vagas}"
            if matric and matric != EMPTY: linha += f" | Matrículas: {matric}"
            lines.append(linha)
        return "\n".join(lines)

    return header


# =============================================================================
# Geração de resposta em linguagem natural (LLM leve)
# =============================================================================

# =============================================================================
# Busca por área/tema de atuação de professores (full-text SQL)
# =============================================================================

# Padrões para detectar perguntas "qual professor atua/pesquisa com X"
# Ordem importa: do mais específico para o mais genérico
_VERB_AREA = r"(?:atuam?|trabalham?|pesquisam?|estudam?|lecionam?|ministram?|coordenam?|desenvolvem?|têm?\s+expertise|tem\s+expertise)"
# PREP_OPT: preposição OPCIONAL — envolve tudo num grupo com ? fora
_PREP_OPT  = r"(?:(?:com|em|sobre|na\s+área\s+de|em\s+área\s+de|de)\s+)?"
_PROF_WORD = r"(?:professor(?:es|a(?:s)?)?|prof\.?|docente(?:s)?|pesquisador(?:es|a(?:s)?)?|servidor(?:es|a(?:s)?)?)"

_PROF_AREA_PATTERNS = [
    # "qual/quais professor(es) ... VERBO [PREP] X?"
    re.compile(
        rf"\b(?:qual(?:is)?)\b.{{0,60}}\b{_PROF_WORD}\b"
        rf".{{0,50}}\b{_VERB_AREA}\b\s*{_PREP_OPT}(.+)",
        re.IGNORECASE,
    ),
    # "quem VERBO [PREP] X?"
    re.compile(
        rf"\bquem\b\s+{_VERB_AREA}\b\s*{_PREP_OPT}(.+)",
        re.IGNORECASE,
    ),
    # "professores/pesquisadores que VERBO [PREP] X"
    re.compile(
        rf"\b(?:{_PROF_WORD}|pesquisadores?)\b.{{0,20}}"
        rf"\bque\s+{_VERB_AREA}\b\s*{_PREP_OPT}(.+)",
        re.IGNORECASE,
    ),
    # "professores/pesquisadores VERB [PREP] X" (sem "que")
    re.compile(
        rf"\b(?:{_PROF_WORD}|pesquisadores?)\b\s+{_VERB_AREA}\b\s*{_PREP_OPT}(.+)",
        re.IGNORECASE,
    ),
    # "professores com/que têm expertise/especialidade em X"
    re.compile(
        rf"\b{_PROF_WORD}\b.{{0,30}}"
        rf"\b(?:têm?|com)\b\s+(?:expertise|especialidade|área)\b\s*(?:em|de|na)?\s+(.+)",
        re.IGNORECASE,
    ),
]

# Conexão reutilizável para buscas de área
_AREA_CONN = None


def _get_area_conn():
    global _AREA_CONN
    try:
        if _AREA_CONN is None or _AREA_CONN.closed:
            _AREA_CONN = psycopg2.connect(**config.DB_CONFIG)
            _AREA_CONN.autocommit = True
        return _AREA_CONN
    except Exception:
        _AREA_CONN = psycopg2.connect(**config.DB_CONFIG)
        _AREA_CONN.autocommit = True
        return _AREA_CONN


def _is_professor_area_query(query: str) -> bool:
    """Retorna True se a query parece ser sobre área/atuação de professor."""
    return any(p.search(query) for p in _PROF_AREA_PATTERNS)


def search_professors_by_area(query: str, max_results: int = 50) -> Optional[str]:
    """
    Detecta perguntas "qual professor atua/pesquisa com X" e responde
    via busca full-text (ILIKE) direto nas tabelas do banco.

    Custo: 2 queries SQL (~5ms) + 1 LLM call.
    Sem embedding, sem vector search.
    """
    if not _is_professor_area_query(query):
        return None

    from keyword_extractor import extract_area_keywords
    terms = extract_area_keywords(query)
    if not terms:
        return None

    print(f"[Área] Termos extraídos: {terms}")

    try:
        conn = _get_area_conn()
        cur  = conn.cursor()

        # --- Busca 1: conteúdo dos servidores (embedding_text) ---
        # Filtra termos muito curtos (< 4 chars) para evitar falsos positivos
        safe_terms = [t for t in terms if len(t) >= 4]
        if not safe_terms:
            safe_terms = terms  # fallback: usa tudo se todos forem curtos
        ilike_clauses = " OR ".join(["lpe.document ILIKE %s"] * len(safe_terms))
        ilike_params  = [f"%{t}%" for t in safe_terms]

        cur.execute(f"""
            SELECT DISTINCT
                lpe.cmetadata->>'titulo'   AS nome,
                lpe.cmetadata->>'source'   AS url,
                lpe.document               AS conteudo
            FROM langchain_pg_embedding lpe
            JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
            WHERE lpc.name = 'ufpel_servidores'
              AND ({ilike_clauses})
            LIMIT %s
        """, ilike_params + [max_results])
        prof_rows = cur.fetchall()

        # --- Busca 2: projetos relacionados à área ---
        proj_ilike = " OR ".join(["(lpe.document ILIKE %s OR lpe.cmetadata::text ILIKE %s)"] * len(safe_terms))
        proj_params = []
        for t in safe_terms:
            proj_params += [f"%{t}%", f"%{t}%"]

        cur.execute(f"""
            SELECT DISTINCT
                lpe.cmetadata->>'titulo' AS titulo,
                lpe.document             AS conteudo
            FROM langchain_pg_embedding lpe
            JOIN langchain_pg_collection lpc ON lpe.collection_id = lpc.uuid
            WHERE lpc.name = 'ufpel_projetos'
              AND ({proj_ilike})
            LIMIT 5
        """, proj_params)
        proj_rows = cur.fetchall()

    except Exception as exc:
        print(f"[Área] Erro SQL: {exc}")
        return None

    if not prof_rows and not proj_rows:
        print(f"[Área] Nenhum resultado para: {terms}")
        return None

    # --- Monta contexto para o LLM ---
    ctx_parts: list[str] = []

    if prof_rows:
        ctx_parts.append("=== PROFESSORES / SERVIDORES COM ATUAÇÃO NA ÁREA ===")
        seen_profs: set[str] = set()
        for nome, url, conteudo in prof_rows:
            if not nome or nome in seen_profs:
                continue
            seen_profs.add(nome)
            ctx_parts.append(f"\n👤 **{nome}**")
            if url:
                ctx_parts.append(f"  Perfil: {url}")
            # Extrai seções relevantes do conteúdo (até 400 chars das partes com o termo)
            for term in terms:
                idx = conteudo.lower().find(term.lower())
                if idx >= 0:
                    snippet_start = max(0, idx - 100)
                    snippet_end   = min(len(conteudo), idx + 300)
                    snippet = conteudo[snippet_start:snippet_end].strip()
                    ctx_parts.append(f"  Trecho relevante: ...{snippet}...")
                    break
            else:
                # Sem trecho específico — usa início do conteúdo
                ctx_parts.append(f"  {conteudo[:250].strip()}")

    if proj_rows:
        ctx_parts.append("\n\n=== PROJETOS RELACIONADOS À ÁREA ===")
        seen_proj: set[str] = set()
        for titulo, conteudo in proj_rows:
            if not titulo or titulo == EMPTY or titulo in seen_proj:
                continue
            seen_proj.add(titulo)
            ctx_parts.append(f"\n📋 **{titulo}**")
            ctx_parts.append(f"  {conteudo[:300].strip()}")

    context = "\n".join(ctx_parts)

    # --- LLM síntese ---
    from providers import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    _area_prompt = ChatPromptTemplate.from_template(
        "Você é um assistente da UFPel. Com base nos dados abaixo, responda "
        "à pergunta do usuário de forma clara e objetiva em português.\n\n"
        "DADOS ENCONTRADOS:\n{context}\n\n"
        "PERGUNTA: {question}\n\n"
        "INSTRUÇÕES:\n"
        "- Liste os professores encontrados com nome e área de atuação.\n"
        "- Se houver projetos relacionados, mencione-os vinculando ao professor.\n"
        "- Inclua o link de perfil quando disponível.\n"
        "- Se os dados forem insuficientes para responder com certeza, diga isso claramente.\n"
        "- Use markdown com ícones (👤 para professor, 📋 para projeto).\n"
        "- NÃO invente informações fora dos dados acima.\n"
    )
    llm    = get_llm()
    chain  = _area_prompt | llm | StrOutputParser()
    print(f"[Área] Sintetizando com {len(prof_rows)} prof(s) e {len(proj_rows)} projeto(s)")
    response = chain.invoke({"context": context, "question": query})
    return response


# =============================================================================
# Prompt LLM para síntese de cursos
# =============================================================================

_STRUCT_PROMPT_TEMPLATE = """\
Você é um assistente da UFPel. Com base nos dados abaixo, responda \
à pergunta do usuário de forma clara e objetiva em português.

DADOS:
{context}

PERGUNTA: {question}

INSTRUÇÕES:
- Liste as informações encontradas em formato organizado (use markdown).
- Se houver URL de disciplina ou Lattes, inclua como link.
- Se os dados estiverem vazios para a pergunta, informe claramente.
- Ao final, inclua a fonte (URL do curso).
- NÃO invente informações que não estejam nos dados acima.
"""


def answer_structured(query: str) -> Optional[str]:
    """
    Tenta responder à query via consulta estruturada ao banco.
    Retorna a resposta em linguagem natural ou None se a query não for
    estrutural (deve seguir para o RAG).

    Custo: 1 SQL + 1 chamada LLM pequena (sem embeddings nem vector search).
    """
    intent = classify_query(query)
    if intent is None:
        return None

    print(f"[Struct] Intent={intent.tipo} | curso='{intent.curso_nome}'"
          + (f" | sem='{intent.semestre}'" if intent.semestre else "")
          + (f" | cred={intent.creditos}" if intent.creditos else ""))

    curso = _find_curso(intent.curso_nome)
    if not curso:
        print(f"[Struct] Curso não encontrado: '{intent.curso_nome}'")
        return None  # Passa para o RAG

    print(f"[Struct] Match: '{curso['titulo']}' (sim={curso['sim']:.2f})")

    # Monta contexto compacto
    context = _build_context_matrix(curso, intent)

    # Chama LLM para síntese em linguagem natural
    from providers import get_llm
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_template(_STRUCT_PROMPT_TEMPLATE)
    llm    = get_llm()
    chain  = prompt | llm | StrOutputParser()

    response = chain.invoke({"context": context, "question": query})

    # Adiciona fonte
    src_url = curso["dados"].get("url") or (curso["dados"].get("metadata") or {}).get("url") or ""
    fonte = f"\n\n**Fontes consultadas:**\n  🔗 [{curso['titulo']}]({src_url})" if src_url else ""
    return response + fonte
