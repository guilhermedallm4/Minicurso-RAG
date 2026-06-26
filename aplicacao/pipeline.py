"""
Pipeline RAG e Validação de Resposta
=============================================================================
Conceito — SEÇÕES 6 e 7 do curso

RAG (Retrieval-Augmented Generation) = busca semântica + LLM

Fluxo online (executado a cada pergunta do usuário):
  1. query do usuário → embedding da query
  2. embedding → busca os K chunks mais próximos no pgvector
  3. chunks recuperados → formatados e inseridos no prompt
  4. prompt → LLM → resposta em linguagem natural

Vantagem sobre LLM puro:
  - Respostas fundamentadas em documentos reais (menos alucinação)
  - Base de conhecimento atualizável sem re-treinar o modelo
  - Rastreabilidade: sabemos qual documento gerou a resposta

Validação (SEÇÃO 7):
  Antes de chamar o LLM (operação mais cara), verificamos se os documentos
  recuperados são suficientemente relevantes para a query. Se nenhum superar
  o RELEVANCE_THRESHOLD, retornamos uma mensagem padrão sem invocar o LLM.

RAGConfig (módulos avançados):
  Habilita busca híbrida, reranking e guardrails de forma composicional.
  Cada feature é opcional e pode ser ligada/desligada sem alterar o restante.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "crawler"))

import config
from providers import get_llm
from store import get_vector_store
from router import route_query, search_all_collections

try:
    from ingest_ufpel import (
        fetch_dados_completos as _fetch_dados_completos,
        lookup_by_title as _lookup_by_title,
    )
    _DADOS_COMPLETOS_AVAILABLE = True
except ImportError:
    _DADOS_COMPLETOS_AVAILABLE = False
    def _fetch_dados_completos(doc_ids):  # noqa: E302
        return {}
    def _lookup_by_title(query, tipo=None, top_k=3, similarity_threshold=0.3):  # noqa: E302
        return []


# =============================================================================
# Configuração do Pipeline (composicional)
# =============================================================================

@dataclass
class RAGConfig:
    """
    Controla quais módulos avançados são ativados no pipeline.

    Os valores padrão são lidos de config.py — altere RAG_TOP_K_FINAL,
    RAG_TOP_K_RETRIEVE e RAG_MAX_CHARS_PER_DOC lá para ajuste global.

    Combinações típicas para o curso:
      RAGConfig()                               → pipeline base
      RAGConfig(use_hybrid=True)                → + BM25
      RAGConfig(use_reranker=True)              → + cross-encoder
      RAGConfig(use_hybrid=True,
                use_reranker=True)              → pipeline completo
      RAGConfig(use_keyword_extraction=True)    → extrai keywords + síntese
      RAGConfig(top_k_final=6)                  → mais docs ao LLM
    """
    # --- Retrieval ---
    use_hybrid:      bool  = False
    hybrid_alpha:    float = 0.5                     # 0=BM25 puro, 1=semântico puro
    top_k_retrieve:  int   = config.RAG_TOP_K_RETRIEVE  # candidatos antes do reranking

    # --- Reranking ---
    use_reranker:    bool  = False
    top_k_final:     int   = config.RAG_TOP_K_FINAL    # docs enviados ao LLM (mínimo: 4)

    # --- Keyword Extraction ---
    use_keyword_extraction: bool = False
    max_chars_per_doc:      int  = config.RAG_MAX_CHARS_PER_DOC

    # --- Guardrails ---
    use_input_guard:  bool = False
    use_output_guard: bool = False


# =============================================================================
# Prompt RAG
# =============================================================================

RAG_PROMPT = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Responda à pergunta usando EXCLUSIVAMENTE as informações do contexto abaixo.\n"
    "Se o contexto não contiver a informação necessária, informe claramente.\n\n"
    "CONTEXTO:\n{context}\n\n"
    "PERGUNTA: {question}\n\n"
    "RESPOSTA:"
)

# Instrução base comum a todos os prompts de síntese
_BASE_INSTRUCTION = (
    "⚠️  Use APENAS as informações dos documentos acima — não invente dados.\n"
    "Se um campo não estiver disponível no contexto, escreva «não informado» ou omita.\n"
    "Ao citar a origem de cada informação, use o formato: 🔗 *Fonte: <título> — <URL>*"
)

# Instrução de fallback (quando não há informação suficiente)
_NO_RESULTS_MSG = (
    "Não encontrei informações suficientes nos documentos disponíveis para responder "
    "a essa pergunta com precisão.\n\n"
    "💡 *Sugestões:*\n"
    "- Tente reformular usando termos mais específicos\n"
    "- Verifique se o nome está correto (ex.: nome completo do professor)\n"
    "- Consulte diretamente o portal: https://institucional.ufpel.edu.br"
)

_PROMPT_SERVIDOR = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Pergunta do usuário: **{question}**\n\n"
    "DOCUMENTOS RECUPERADOS ({n_docs} documento(s)):\n"
    "{context}\n\n"
    "{instruction}\n\n"
    "Para cada professor/servidor encontrado nos documentos, apresente:\n\n"
    "---\n"
    "### 👤 <Nome completo>\n\n"
    "| Campo | Valor |\n"
    "|-------|-------|\n"
    "| **Cargo** | <cargo e classe/nível, se disponível> |\n"
    "| **Lotação** | <unidade ou departamento> |\n"
    "| **Regime de trabalho** | <regime e jornada> |\n"
    "| **Titulação** | <maior titulação> |\n"
    "| **Situação** | <situação funcional> |\n"
    "| **Ingresso na UFPel** | <data> |\n\n"
    "**Áreas de atuação:**\n"
    "<liste cada área em tópico>\n\n"
    "**Resumo profissional:**\n"
    "<resumo do Lattes ou da página do portal>\n\n"
    "**Formação acadêmica:**\n"
    "<cada titulação em linha separada>\n\n"
    "🔗 *Fonte: <título> — <URL do portal>*\n"
    "📋 *Lattes: <URL do Lattes, se disponível>*\n"
    "---\n\n"
    "RESPOSTA:"
)

_PROMPT_DISCIPLINA = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Pergunta do usuário: **{question}**\n\n"
    "DOCUMENTOS RECUPERADOS ({n_docs} documento(s)):\n"
    "{context}\n\n"
    "{instruction}\n\n"
    "Para cada disciplina encontrada nos documentos, apresente:\n\n"
    "---\n"
    "### 📚 <Nome da Disciplina>\n\n"
    "| Campo | Valor |\n"
    "|-------|-------|\n"
    "| **Código** | <código> |\n"
    "| **Carga horária** | <CH total> |\n"
    "| **Créditos** | <N> |\n"
    "| **Unidade responsável** | <departamento/instituto> |\n"
    "| **Periodicidade** | <semestral/anual> |\n"
    "| **Tipo** | <obrigatória/optativa> |\n\n"
    "**Ementa:**\n"
    "<texto da ementa>\n\n"
    "**Objetivos:**\n"
    "<texto dos objetivos, se disponível>\n\n"
    "**Conteúdo programático:**\n"
    "<conteúdo, se disponível>\n\n"
    "🔗 *Fonte: <título> — <URL>*\n"
    "---\n\n"
    "RESPOSTA:"
)

_PROMPT_PROJETO = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Pergunta do usuário: **{question}**\n\n"
    "DOCUMENTOS RECUPERADOS ({n_docs} documento(s)):\n"
    "{context}\n\n"
    "{instruction}\n\n"
    "Para cada projeto encontrado nos documentos, apresente:\n\n"
    "---\n"
    "### 🔬 <Nome do Projeto>\n\n"
    "| Campo | Valor |\n"
    "|-------|-------|\n"
    "| **Ênfase** | <Pesquisa / Extensão / Ensino> |\n"
    "| **Coordenador** | <nome> |\n"
    "| **Unidade** | <unidade de origem> |\n"
    "| **Área CNPq** | <grande área — área> |\n"
    "| **Período** | <data início> a <data fim> |\n\n"
    "**Resumo:**\n"
    "<resumo do projeto>\n\n"
    "**Objetivo geral:**\n"
    "<objetivo, se disponível>\n\n"
    "**Equipe:**\n"
    "<membros e papéis, se disponível>\n\n"
    "🔗 *Fonte: <título> — <URL>*\n"
    "---\n\n"
    "RESPOSTA:"
)

_PROMPT_CURSO = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Pergunta do usuário: **{question}**\n\n"
    "DOCUMENTOS RECUPERADOS ({n_docs} documento(s)):\n"
    "{context}\n\n"
    "{instruction}\n\n"
    "Para cada curso encontrado nos documentos, apresente:\n\n"
    "---\n"
    "### 🎓 <Nome do Curso>\n\n"
    "| Campo | Valor |\n"
    "|-------|-------|\n"
    "| **Nível** | <Graduação / Pós-Graduação / Técnico e grau> |\n"
    "| **Modalidade** | <Presencial / EaD> |\n"
    "| **Turno** | <Diurno / Noturno / Integral> |\n"
    "| **Unidade** | <faculdade/instituto responsável> |\n"
    "| **Coordenador(a)** | <nome> |\n"
    "| **Código UFPel** | <código, se disponível> |\n\n"
    "**Sobre o curso:**\n"
    "<contextualização do curso>\n\n"
    "**Objetivos:**\n"
    "<objetivos do curso>\n\n"
    "**Perfil do egresso:**\n"
    "<perfil, se disponível>\n\n"
    "**Corpo docente (amostra):**\n"
    "<liste até 5 professores>\n\n"
    "🔗 *Fonte: <título> — <URL>*\n"
    "---\n\n"
    "RESPOSTA:"
)

_PROMPT_UNIDADE = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Pergunta do usuário: **{question}**\n\n"
    "DOCUMENTOS RECUPERADOS ({n_docs} documento(s)):\n"
    "{context}\n\n"
    "{instruction}\n\n"
    "Para cada unidade encontrada nos documentos, apresente:\n\n"
    "---\n"
    "### 🏛️ <Nome da Unidade>\n\n"
    "| Campo | Valor |\n"
    "|-------|-------|\n"
    "| **Sigla** | <sigla, se disponível> |\n"
    "| **Código SIORG** | <código> |\n"
    "| **Direção / Chefia** | <nome> |\n"
    "| **Contato** | <e-mail ou telefone, se disponível> |\n\n"
    "**Estrutura:**\n"
    "<subunidades ou informações organizacionais>\n\n"
    "🔗 *Fonte: <título> — <URL>*\n"
    "---\n\n"
    "RESPOSTA:"
)

_PROMPT_GERAL = ChatPromptTemplate.from_template(
    "Você é um assistente do portal institucional da UFPel.\n"
    "Pergunta do usuário: **{question}**\n\n"
    "DOCUMENTOS RECUPERADOS ({n_docs} documento(s)):\n"
    "{context}\n\n"
    "{instruction}\n\n"
    "Apresente as informações de forma clara e organizada:\n"
    "- Use **negrito** para títulos e campos importantes\n"
    "- Separe múltiplos itens com `---`\n"
    "- Se a pergunta for sobre uma entidade específica, responda diretamente\n"
    "- Se envolver lista, numere ou use tópicos\n"
    "- Ao final: 🔗 *Fonte: <título> — <URL>*\n\n"
    "Se não houver informação suficiente nos documentos, informe claramente "
    "e sugira consultar https://institucional.ufpel.edu.br\n\n"
    "RESPOSTA:"
)

# Mapeamento coleção → prompt
_COLLECTION_PROMPTS: dict[str, ChatPromptTemplate] = {
    "ufpel_servidores":  _PROMPT_SERVIDOR,
    "ufpel_disciplinas": _PROMPT_DISCIPLINA,
    "ufpel_projetos":    _PROMPT_PROJETO,
    "ufpel_cursos":      _PROMPT_CURSO,
    "ufpel_unidades":    _PROMPT_UNIDADE,
}

# mantido para compatibilidade com build_rag_chain / validate_and_answer
SYNTHESIS_PROMPT = _PROMPT_GERAL


def _get_prompt(collection_name: str | None) -> ChatPromptTemplate:
    """Retorna o prompt estruturado adequado para a coleção."""
    return _COLLECTION_PROMPTS.get(collection_name or "", _PROMPT_GERAL)


def _format_docs(docs: List[Document], max_chars_per_doc: int = 0) -> str:
    """Formata os chunks recuperados para o prompt com atribuição de fonte e título.

    Quando disponível, inclui dados_completos do documento para contexto mais rico.
    max_chars_per_doc > 0 trunca o conteúdo de cada doc para evitar estourar o contexto da API.
    """
    import json as _json

    # Busca dados_completos para os docs que têm doc_id
    doc_ids = [d.metadata["doc_id"] for d in docs if d.metadata.get("doc_id")]
    dados_map = _fetch_dados_completos(doc_ids) if doc_ids else {}

    # Agrupa chunks pelo mesmo doc_id para evitar repetir dados_completos
    seen_doc_ids: set = set()
    parts = []
    for d in docs:
        titulo = d.metadata.get("titulo", "")
        source = d.metadata.get("source", "documento")
        header = f"[Título: {titulo} | Fonte: {source}]" if titulo else f"[Fonte: {source}]"
        content = d.page_content
        if max_chars_per_doc > 0 and len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "…"

        doc_id = d.metadata.get("doc_id", "")
        dados_completos = dados_map.get(doc_id)
        if dados_completos and doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            # Serializa de forma legível para o LLM
            dados_str = _json.dumps(dados_completos, ensure_ascii=False, indent=2)
            # Limita tamanho para não exceder contexto (máx 4000 chars por doc)
            if len(dados_str) > 4000:
                dados_str = dados_str[:4000] + "\n... (truncado)"
            parts.append(f"{header}\n{content}\n\n[Dados completos do documento]\n{dados_str}")
        else:
            parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


def _select_sources(results: List[Tuple], threshold: float = 0.85) -> List[str]:
    """
    Retorna apenas as URLs das fontes cujo score está dentro do threshold do melhor score.

    Evita que registros duplicados de um mesmo servidor (IDs diferentes no portal,
    mesma pessoa) apareçam todos como fontes na resposta.

    threshold=0.85 → só mostra fontes com score >= 85% do score máximo.
    """
    if not results:
        return []
    scores_by_source: dict[str, float] = {}
    for doc, score in results:
        src = doc.metadata.get("source", "documento")
        if src not in scores_by_source or score > scores_by_source[src]:
            scores_by_source[src] = score

    best = max(scores_by_source.values())
    cutoff = best * threshold
    return sorted(src for src, score in scores_by_source.items() if score >= cutoff)


def _sources_cited_in_response(response: str, results: List[Tuple]) -> List[str]:
    """
    Retorna apenas as URLs dos documentos cujo título foi mencionado na resposta.

    O LLM recebe a instrução de citar o título de cada documento usado. Após a
    resposta, verificamos quais títulos aparecem no texto e devolvemos somente
    essas fontes. Se nenhum título for encontrado (ex.: resposta negativa),
    cai no fallback por score via _select_sources().
    """
    response_lower = response.lower()
    cited: list[str] = []
    seen_sources: set[str] = set()

    for doc, _ in results:
        titulo = doc.metadata.get("titulo", "")
        source = doc.metadata.get("source", "")
        if not titulo or not source or source in seen_sources:
            continue
        # Checa se pelo menos metade das palavras do título estão na resposta
        words = [w for w in titulo.lower().split() if len(w) > 3]
        if words and sum(1 for w in words if w in response_lower) >= max(1, len(words) // 2):
            cited.append(source)
            seen_sources.add(source)

    return sorted(cited) if cited else _select_sources(results)


# =============================================================================
# Construção do Retriever (base, híbrido, reranker)
# =============================================================================

def build_retriever(store: PGVector, cfg: RAGConfig):
    """Constrói o retriever de acordo com as opções do RAGConfig."""
    if cfg.use_hybrid:
        from hybrid_search import build_hybrid_retriever
        base = build_hybrid_retriever(store, alpha=cfg.hybrid_alpha, top_k=cfg.top_k_retrieve)
    else:
        base = store.as_retriever(search_kwargs={"k": cfg.top_k_retrieve})

    if cfg.use_reranker:
        from reranker import rerank_documents

        # Wrapper: invoca o retriever base e aplica o reranker sobre os candidatos
        class RerankedRetriever:
            def __init__(self, base_retriever, top_k):
                self._base = base_retriever
                self._top_k = top_k

            def invoke(self, query: str) -> List[Document]:
                candidates = self._base.invoke(query)
                return rerank_documents(query, candidates, top_k=self._top_k)

            # compatibilidade com LCEL (| pipe)
            def __or__(self, other):
                from langchain_core.runnables import RunnableLambda
                return RunnableLambda(self.invoke) | other

        return RerankedRetriever(base, cfg.top_k_final)

    return base


# =============================================================================
# Construção da Cadeia RAG
# =============================================================================

def build_rag_chain(
    store: Optional[PGVector] = None,
    rag_config: Optional[RAGConfig] = None,
) -> Tuple:
    """
    Constrói a cadeia RAG com LangChain Expression Language (LCEL).

    O operador | (pipe) encadeia componentes:
      retriever → formata docs → prompt → LLM → parser

    Args:
        store      : PGVector store (criado automaticamente se None)
        rag_config : configuração avançada (padrão: pipeline base)
    """
    store  = store or get_vector_store()
    cfg    = rag_config or RAGConfig()
    llm    = get_llm()
    retriever = build_retriever(store, cfg)
    print(retriever)
    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# =============================================================================
# Resposta com validação de relevância e guardrails
# =============================================================================

def validate_and_answer(
    query: str,
    chain,
    store: PGVector,
    rag_config: Optional[RAGConfig] = None,
) -> str:
    """
    Fluxo completo de resposta:
      [input guard] → relevância → LLM → [output guard] → usuário
    """
    cfg = rag_config or RAGConfig()

    # --- Guard de entrada ---  (sem cache nesta função — usada internamente)
    if cfg.use_input_guard:
        from guardrails import DEFAULT_INPUT_GUARD
        result = DEFAULT_INPUT_GUARD.check(query)
        if not result.passed:
            print(f"[Guard:entrada:{result.check}] Bloqueado")
            return f"Pergunta bloqueada: {result.reason}"

    # --- Validação de relevância ---
    results = store.similarity_search_with_relevance_scores(query, k=4)
    if not results:
        return "Não há documentos na base. Execute a ingestão primeiro."

    docs, scores = zip(*results)
    best_score = max(scores)
    print(f"[Validação] Score de relevância: {best_score:.3f} "
          f"(limiar={config.RELEVANCE_THRESHOLD})")

    if best_score < config.RAG_RELEVANCE_THRESHOLD:
        return _NO_RESULTS_MSG

    # --- Geração ---
    context  = _format_docs(list(docs))
    response = chain.invoke(query)

    # --- Guard de saída ---
    if cfg.use_output_guard:
        from guardrails import DEFAULT_OUTPUT_GUARD
        out_result = DEFAULT_OUTPUT_GUARD.check(response, context=context, query=query)
        if not out_result.passed:
            print(f"[Guard:saída:{out_result.check}] Bloqueado")
            return f"Resposta bloqueada pelo sistema de segurança: {out_result.reason}"

    sources = _sources_cited_in_response(response, results)
    return f"{response}\n\n[Fontes: {', '.join(sources)}]"


# =============================================================================
# Pipeline com roteamento multi-coleção (Portal UFPel)
# =============================================================================

def _answer_from_results(
    query: str,
    results: list,
    cfg: RAGConfig,
) -> str:
    """Gera resposta do LLM a partir de (Document, score) já recuperados."""
    if not results:
        return "Não há documentos na base. Execute a ingestão primeiro."

    docs, _ = zip(*results)
    context  = _format_docs(list(docs), max_chars_per_doc=cfg.max_chars_per_doc)
    llm      = get_llm()
    response = (RAG_PROMPT | llm | StrOutputParser()).invoke(
        {"context": context, "question": query}
    )

    if cfg.use_output_guard:
        from guardrails import DEFAULT_OUTPUT_GUARD
        out_result = DEFAULT_OUTPUT_GUARD.check(response, context=context, query=query)
        if not out_result.passed:
            print(f"[Guard:saída:{out_result.check}] Bloqueado")
            return f"Resposta bloqueada pelo sistema de segurança: {out_result.reason}"

    sources = _sources_cited_in_response(response, results)
    return f"{response}\n\n[Fontes: {', '.join(sources)}]"


def _search_with_keywords(
    collection_name: str,
    keywords: List[str],
    top_k: int = 4,
) -> List[Tuple]:
    """
    Busca na coleção usando cada keyword individualmente e deduplica os resultados.

    Cada keyword contribui com candidatos; o ranking final é por score decrescente.
    Documentos duplicados (mesmo conteúdo inicial) são descartados.
    """
    store = get_vector_store(collection_name=collection_name)
    per_kw = max(2, (top_k + len(keywords) - 1) // len(keywords))

    seen: set = set()
    all_results: List[Tuple] = []

    for kw in keywords:
        try:
            results = store.similarity_search_with_relevance_scores(kw, k=per_kw + 1)
            for doc, score in results:
                doc_key = doc.page_content[:150]
                if doc_key not in seen:
                    seen.add(doc_key)
                    all_results.append((doc, score))
        except Exception as exc:
            print(f"[Keywords] Erro ao buscar '{kw}': {exc}")

    all_results.sort(key=lambda x: x[1], reverse=True)
    top = all_results[:top_k]
    if top:
        print(f"[Keywords] {len(top)} docs únicos após busca por {len(keywords)} keyword(s)")
    return top


_PROJ_CONN: Optional[object] = None  # conexão reutilizável (lazy singleton)


def _get_proj_conn():
    """Retorna conexão psycopg2 reutilizável para consultas de projetos."""
    global _PROJ_CONN
    try:
        if _PROJ_CONN is None or _PROJ_CONN.closed:
            import psycopg2 as _pg
            _PROJ_CONN = _pg.connect(**config.DB_CONFIG)
        _PROJ_CONN.autocommit = True
        return _PROJ_CONN
    except Exception:
        import psycopg2 as _pg
        _PROJ_CONN = _pg.connect(**config.DB_CONFIG)
        _PROJ_CONN.autocommit = True
        return _PROJ_CONN


def _fetch_professor_projects(prof_names: list[str], max_per_prof: int = 5) -> str:
    """
    Busca projetos relacionados a uma lista de professores via SQL direto
    no doc_completos. Custo: 1-2 queries SQL reutilizando conexão (~10-30ms),
    sem embedding nem LLM extra.

    Estratégia em 2 passos (do mais rápido ao mais lento):
      1. Coordenador (índice funcional: ultra-rápido)
      2. Membro da equipe (fallback: busca por texto no array)

    Retorna string formatada para injetar no contexto, ou '' se vazio.
    """
    if not prof_names:
        return ""

    EMPTY = "Não há informações disponíveis"
    names_upper = [n.upper() for n in prof_names]
    placeholders = ", ".join(["%s"] * len(names_upper))
    limit = max_per_prof * max(len(prof_names), 1)

    try:
        conn = _get_proj_conn()
        cur = conn.cursor()

        # Passo 1: busca por coordenador (usa idx_dc_projeto_coord — rápido)
        cur.execute(f"""
            SELECT
                dados->>'Nome do Projeto'   AS nome,
                dados->>'Coordenador Atual' AS coord,
                dados->>'Área CNPq'         AS area,
                dados->>'resumo'            AS resumo,
                dados->>'data_inicio'       AS inicio,
                dados->>'data_fim'          AS fim,
                'coordenador'               AS papel
            FROM doc_completos
            WHERE tipo = 'projeto'
              AND upper(dados->>'Coordenador Atual') = ANY(ARRAY[{placeholders}])
            ORDER BY dados->>'data_inicio' DESC NULLS LAST
            LIMIT %s
        """, names_upper + [limit])
        rows_coord = cur.fetchall()

        # Passo 2: busca por membro da equipe (apenas se tiver poucos resultados)
        rows_member: list = []
        if len(rows_coord) < limit // 2:
            coord_names_found = {r[1] for r in rows_coord if r[1]}
            cur.execute(f"""
                SELECT
                    dados->>'Nome do Projeto'   AS nome,
                    dados->>'Coordenador Atual' AS coord,
                    dados->>'Área CNPq'         AS area,
                    dados->>'resumo'            AS resumo,
                    dados->>'data_inicio'       AS inicio,
                    dados->>'data_fim'          AS fim,
                    'membro'                    AS papel
                FROM doc_completos
                WHERE tipo = 'projeto'
                  AND upper(dados->>'Coordenador Atual') != ALL(ARRAY[{placeholders}])
                  AND EXISTS (
                      SELECT 1
                      FROM jsonb_array_elements_text(dados->'equipe') AS membro
                      WHERE upper(split_part(membro, ' — ', 1)) = ANY(ARRAY[{placeholders}])
                  )
                ORDER BY dados->>'data_inicio' DESC NULLS LAST
                LIMIT %s
            """, names_upper * 2 + [limit - len(rows_coord)])
            rows_member = cur.fetchall()

    except Exception as exc:
        print(f"[Projetos] Erro ao buscar projetos: {exc}")
        return ""

    rows = rows_coord + rows_member
    if not rows:
        return ""

    if not rows:
        return ""

    EMPTY = "Não há informações disponíveis"
    lines = ["\n\n=== PROJETOS RELACIONADOS (via doc_completos) ==="]
    seen: set[str] = set()
    for nome, coord, area, resumo, inicio, fim, papel in rows:
        nome = nome or EMPTY
        if nome == EMPTY or nome in seen:
            continue
        seen.add(nome)
        papel_label = "(coordenador)" if papel == "coordenador" else "(membro da equipe)"
        lines.append(f"\n📋 **{nome}** {papel_label}")
        if coord and coord != EMPTY:
            lines.append(f"  Coordenador: {coord}")
        if area and area != EMPTY:
            lines.append(f"  Área CNPq: {area}")
        periodo = " a ".join(p for p in [inicio, fim] if p and p != EMPTY)
        if periodo:
            lines.append(f"  Período: {periodo}")
        if resumo and resumo != EMPTY:
            lines.append(f"  Resumo: {resumo[:250]}{'...' if len(resumo) > 250 else ''}")

    return "\n".join(lines) if len(lines) > 1 else ""


def _synthesize_from_results(
    query: str,
    results: List[Tuple],
    keywords: List[str],
    cfg: RAGConfig,
    collection_name: str | None = None,
) -> str:
    """Gera resposta estruturada usando os top N docs recuperados."""
    docs, _ = zip(*results)
    context  = _format_docs(list(docs), max_chars_per_doc=cfg.max_chars_per_doc)

    # Enriquecimento com projetos: apenas para consultas sobre servidores.
    # Custo: 1 SQL (~5-15ms), zero chamadas de LLM/embedding extras.
    projects_ctx = ""
    if collection_name == "ufpel_servidores":
        prof_names = list({
            doc.metadata.get("titulo", "").split(" | ")[0].strip()
            for doc in docs
            if doc.metadata.get("tipo") == "servidor"
        })
        if prof_names:
            projects_ctx = _fetch_professor_projects(prof_names)
            if projects_ctx:
                print(f"[Projetos] Enriquecido com {projects_ctx.count('📋')} projeto(s) de {len(prof_names)} prof(s)")

    full_context = context + projects_ctx

    from reliability import record_request as _record_req
    llm    = get_llm()
    prompt = _get_prompt(collection_name)
    with _record_req("llm", context=f"synthesize:{collection_name or 'multi'}"):
        response = (prompt | llm | StrOutputParser()).invoke({
            "context":     full_context,
            "question":    query,
            "n_docs":      len(docs),
            "instruction": _BASE_INSTRUCTION,
        })

    if cfg.use_output_guard:
        from guardrails import DEFAULT_OUTPUT_GUARD
        out_result = DEFAULT_OUTPUT_GUARD.check(response, context=context, query=query)
        if not out_result.passed:
            print(f"[Guard:saída:{out_result.check}] Bloqueado")
            return f"Resposta bloqueada pelo sistema de segurança: {out_result.reason}"

    sources = _sources_cited_in_response(response, results)
    return f"{response}\n\n[Fontes: {', '.join(sources)}]"


def _format_curso_context(titulo: str, dados: dict, src_url: str, sim: float) -> str:
    """Formata dados de um curso de forma estruturada e compacta para o LLM."""
    EMPTY = "Não há informações disponíveis"
    lines = [
        f"[Curso: {titulo}" + (f" | URL: {src_url}" if src_url else "") + f" | relevância={sim:.2f}]",
        f"Nome: {dados.get('nome', titulo)}",
        f"Nível: {dados.get('nivel', EMPTY)}",
        f"Modalidade: {dados.get('modalidade', EMPTY)} | Turno: {dados.get('turno', EMPTY)}",
        f"Unidade: {dados.get('unidade', EMPTY)}",
        f"Coordenador: {dados.get('coordenador', EMPTY)}",
    ]
    if dados.get("objetivos") and dados["objetivos"] != EMPTY:
        lines.append(f"\nObjetivos:\n{dados['objetivos'][:400]}")

    # Matriz curricular agrupada por semestre
    matriz = dados.get("matriz_curricular", [])
    if matriz:
        lines.append("\n=== MATRIZ CURRICULAR ===")
        sem_map: dict[str, list[dict]] = {}
        for row in matriz:
            sem = row.get("semestre", "Sem semestre")
            sem_map.setdefault(sem, []).append(row)
        for sem, discs in sem_map.items():
            lines.append(f"\n{sem}:")
            for d in discs:
                nome = d.get("Disciplina / Pré-requisitos", d.get("Disciplina", "?"))
                cod  = d.get("Código", "")
                cred = d.get("Cr.", d.get("créditos", ""))
                hor  = d.get("Horas", "")
                car  = d.get("Caráter", "")
                url  = d.get("Disciplina / Pré-requisitos_url", d.get("Código_url", ""))
                linha = f"  - {nome}"
                if cod:
                    linha += f" (cód. {cod})"
                if cred:
                    linha += f" | {cred} créditos"
                if hor:
                    linha += f" | {hor}h"
                if car:
                    linha += f" | {car}"
                if url:
                    linha += f" | {url}"
                lines.append(linha)

    # Professores
    professores = dados.get("professores", [])
    profs_validos = [p for p in professores if p.get("nome") and p["nome"] != EMPTY]
    if profs_validos:
        lines.append("\n=== CORPO DOCENTE ===")
        for p in profs_validos:
            linha = f"  - {p['nome']}"
            if p.get("unidade") and p["unidade"] != EMPTY:
                linha += f" ({p['unidade']})"
            if p.get("lattes"):
                linha += f" | Lattes: {p['lattes']}"
            elif p.get("url"):
                linha += f" | Perfil: {p['url']}"
            lines.append(linha)

    # Turmas ofertadas (resumo)
    turmas = dados.get("turmas_ofertadas", [])
    if turmas:
        lines.append("\n=== TURMAS OFERTADAS ===")
        for t in turmas[:20]:
            disc = t.get("disciplina", "?")
            profs_t = ", ".join(t.get("professores", []))
            turma_id = t.get("turma", "")
            vagas    = t.get("vagas", "")
            lines.append(
                f"  - {disc}"
                + (f" | Prof: {profs_t}" if profs_t else "")
                + (f" | Turma: {turma_id}" if turma_id and turma_id != EMPTY else "")
                + (f" | Vagas: {vagas}" if vagas and vagas != EMPTY else "")
            )
        if len(turmas) > 20:
            lines.append(f"  ... e mais {len(turmas)-20} turmas")

    return "\n".join(lines)


def _format_dados_completos_context(matches: list[dict]) -> str:
    """Formata registros de doc_completos como contexto estruturado para o LLM."""
    import json as _json
    parts = []
    for m in matches:
        titulo = m.get("titulo", "")
        tipo   = m.get("tipo", "")
        dados  = m.get("dados", {})
        sim    = m.get("similarity", 0)
        src_url = (
            dados.get("url")
            or (dados.get("metadata") or {}).get("url")
            or ""
        )
        # Cursos recebem formatação estruturada (matriz completa, corpo docente)
        if tipo == "curso":
            parts.append(_format_curso_context(titulo, dados, src_url, sim))
            continue

        header = (
            f"[{tipo.capitalize()}: {titulo}"
            + (f" | URL: {src_url}" if src_url else "")
            + f" | relevância={sim:.2f}]"
        )
        dados_str = _json.dumps(dados, ensure_ascii=False, indent=2)
        if len(dados_str) > 6000:
            dados_str = dados_str[:6000] + "\n... (truncado)"
        parts.append(f"{header}\n{dados_str}")
    return "\n\n---\n\n".join(parts)


# Palavras de relação que indicam que a query é sobre uma entidade nomeada
_ENTITY_TRIGGER_PATTERNS = [
    r"(?:o que (?:é|faz|estuda|pesquisa)|quem é|fale sobre|me fale sobre|informações sobre|área d[aeo]|"
    r"trabalha com|atua com|lotad[ao] em|cargo d[ao]|titulação d[ao]|currículo d[ao]|lattes d[ao]|"
    r"coordenador[a]? d[ao]|sobre o curso|o que é o curso|o que trata|ementa d[ae])\s+(.+)",
    r"(?:professor[a]?|prof\.?|servidor[a]?|docente)\s+(.+?)(?:\?|$)",
    # Queries sobre grade/disciplinas/professores/turmas de um curso específico
    # Captura o nome do curso após "do curso de" / "no curso de"
    r"(?:disciplinas?|matérias?|grade|matriz|professores?|docentes?|turmas?)[^?]*\bcurso\s+de\s+(.+?)(?:\?|$)",
    # "disciplinas do/no/da X" — captura X após preposição
    r"(?:disciplinas?|matérias?|grade|matriz|professores?|docentes?|turmas?)\s+(?:\w+\s+)*(?:d[ao]|n[ao]|no)\s+(.+?)(?:\?|$)",
    r"(?:curso de|graduação em|pós-graduação em|bacharelado em|licenciatura em|tecnólogo em)\s+(.+?)(?:\?|$)",
    r"(?:disciplina|matéria)\s+(.+?)(?:\?|$)",
]

_TIPO_HINTS = {
    "professor": "servidor", "prof": "servidor", "docente": "servidor", "servidor": "servidor",
    "curso": "curso", "graduação": "curso", "bacharelado": "curso", "licenciatura": "curso",
    "pós-graduação": "curso", "tecnólogo": "curso",
    # Queries sobre matriz/grade de curso → busca em cursos (não em disciplinas)
    "grade curricular": "curso", "matriz curricular": "curso",
    "semestre do curso": "curso", "disciplinas do curso": "curso",
    "unidade": "unidade", "centro": "unidade", "instituto": "unidade", "faculdade": "unidade",
}


def _try_direct_lookup(query: str, cfg: "RAGConfig") -> Optional[str]:
    """
    Tenta responder a query via busca direta por título em doc_completos,
    sem gerar embeddings. Retorna None se não encontrar match suficientemente
    forte, sinalizando que o pipeline semântico deve continuar.

    Estratégia:
    1. Detecta se a query menciona uma entidade nomeada via regex.
    2. Extrai o candidato de nome/entidade.
    3. Busca por similaridade de título no doc_completos (pg_trgm).
    4. Se similaridade >= 0.4, formata os dados como contexto e chama o LLM.
    """
    import re as _re

    q_lower = query.lower().strip()

    # Infere tipo sugerido a partir de palavras-chave na query
    tipo_hint: Optional[str] = None
    for kw, tipo in _TIPO_HINTS.items():
        if kw in q_lower:
            tipo_hint = tipo
            break

    # Extrai candidato de nome via padrões
    candidate: Optional[str] = None
    for pattern in _ENTITY_TRIGGER_PATTERNS:
        m = _re.search(pattern, q_lower, _re.IGNORECASE)
        if m:
            candidate = m.group(m.lastindex).strip().rstrip("?. ")
            break

    # Se não achou padrão, tenta usar a query inteira como candidato de nome
    # apenas se for curta (provável nome próprio) e sem palavras interrogativas genéricas
    if candidate is None:
        generic_words = {"qual", "quais", "como", "quando", "onde", "por que", "porque", "liste"}
        words = set(q_lower.split())
        if not generic_words.intersection(words) and len(q_lower.split()) <= 6:
            candidate = q_lower.rstrip("?. ")

    if not candidate:
        return None

    matches = _lookup_by_title(
        candidate.upper(),           # títulos estão em maiúsculas no portal
        tipo=tipo_hint,
        top_k=3,
        similarity_threshold=0.35,
    )

    if not matches:
        # Tenta sem restricão de tipo
        if tipo_hint:
            matches = _lookup_by_title(candidate.upper(), top_k=3, similarity_threshold=0.35)

    if not matches or matches[0]["similarity"] < 0.35:
        return None

    best = matches[0]
    print(f"[Shortcut] Match direto: '{best['titulo']}' "
          f"(sim={best['similarity']:.2f}, tipo={best['tipo']})")

    # Se o match é um curso e a query tem filtros de semestre/créditos,
    # delega ao structured_query para garantir o filtro correto.
    if best["tipo"] == "curso":
        try:
            from structured_query import classify_query, _build_context_matrix
            intent = classify_query(query)
            if intent and intent.tipo in ("MATRIX_SEMESTRE", "CREDITOS_FILTER"):
                curso_data = {
                    "titulo": best["titulo"],
                    "dados":  best["dados"],
                    "sim":    best["similarity"],
                }
                context = _build_context_matrix(curso_data, intent)
            else:
                context = _format_dados_completos_context(matches)
        except Exception:
            context = _format_dados_completos_context(matches)
    else:
        context = _format_dados_completos_context(matches)

    # Enriquece servidores com projetos via SQL direto (sem custo de embedding/LLM)
    if best["tipo"] == "servidor":
        prof_names = [m.get("titulo", "").split(" | ")[0].strip() for m in matches]
        projects_ctx = _fetch_professor_projects(prof_names)
        if projects_ctx:
            print(f"[Projetos] {projects_ctx.count('📋')} projeto(s) para direct lookup")
            context = context + projects_ctx

    llm      = get_llm()
    prompt   = _get_prompt(config.collection_for_tipo(best["tipo"]))
    response = (prompt | llm | StrOutputParser()).invoke({
        "context":     context,
        "question":    query,
        "n_docs":      len(matches),
        "instruction": _BASE_INSTRUCTION,
    })

    if cfg.use_output_guard:
        from guardrails import DEFAULT_OUTPUT_GUARD
        out_result = DEFAULT_OUTPUT_GUARD.check(response, context=context, query=query)
        if not out_result.passed:
            return f"Resposta bloqueada pelo sistema de segurança: {out_result.reason}"

    # Extrai a URL de origem de cada match (campo 'url' varia por tipo de documento)
    def _extract_source_url(m: dict) -> str:
        dados = m.get("dados") or {}
        # Tenta campo direto 'url', depois campo 'metadata.url', depois lattes_url
        return (
            dados.get("url")
            or (dados.get("metadata") or {}).get("url")
            or ""
        )

    sources_with_title = [
        (m.get("titulo", ""), _extract_source_url(m))
        for m in matches
    ]
    source_lines = "\n".join(
        f"  🔗 [{titulo}]({src_url})" if src_url else f"  • {titulo}"
        for titulo, src_url in sources_with_title
        if titulo
    )
    suffix = f"\n\n**Fontes consultadas:**\n{source_lines}" if source_lines else ""
    return f"{response}{suffix}"


def answer_with_routing(
    query: str,
    rag_config: Optional[RAGConfig] = None,
    collection_name: Optional[str] = None,
) -> str:
    """
    Pipeline RAG com roteamento automático de coleção.

    Fluxo:
      [input guard] → LLM router → coleção alvo → validação de relevância
          → [score baixo? tenta fallback multi-coleção] → LLM → [output guard]

    Args:
        query           : pergunta do usuário
        rag_config      : configurações avançadas (híbrido, reranker, guardrails)
        collection_name : força uma coleção específica (ignora router)
    """
    cfg = rag_config or RAGConfig()

    # --- Cache de respostas ---------------------------------------------------
    # Nível 1 (exact): hash SHA-256 da query normalizada — O(1).
    # Nível 2 (fuzzy): trigrama, threshold 0.82 — captura variações mínimas.
    # TTL: 1 hora. Poupa toda a cadeia LLM + vetorial em hits.
    try:
        from cache import response_cache
        cached = response_cache.get(query)
        if cached is not None:
            return cached
    except Exception:
        pass

    # --- Guard de entrada ---
    if cfg.use_input_guard:
        from guardrails import DEFAULT_INPUT_GUARD
        result = DEFAULT_INPUT_GUARD.check(query)
        if not result.passed:
            print(f"[Guard:entrada:{result.check}] Bloqueado")
            return f"Pergunta bloqueada: {result.reason}"

    # --- Busca por área de atuação de professor (full-text SQL) ---------------
    # "Qual professor atua com X?" → ILIKE nos chunks de servidores + projetos.
    # Custo: 2 SQL (~5ms) + 1 LLM  vs  RAG: embedding + vector search + LLM.
    try:
        from structured_query import search_professors_by_area
        area_answer = search_professors_by_area(query)
        if area_answer is not None:
            return area_answer
    except Exception as _e:
        print(f"[Área] Erro: {_e} — seguindo para RAG")

    # --- Consulta estruturada: disciplinas/professores/turmas de curso --------
    # Acessa doc_completos via SQL direto sem embedding nem vector search.
    # Usada para perguntas sobre grade curricular, semestres, créditos, etc.
    # Custo: 1 SQL + 1 LLM leve  vs  RAG: embeddings + vector search + LLM.
    try:
        from structured_query import answer_structured
        struct_answer = answer_structured(query)
        if struct_answer is not None:
            return struct_answer
    except Exception as _e:
        print(f"[Struct] Erro: {_e} — seguindo para RAG")

    # --- Shortcut: consulta direta para entidade nomeada ---------------------
    # Evita embedding + busca vetorial quando a query menciona um nome
    # específico (professor, curso, disciplina, unidade).
    # Detecta padrões como: "O que o professor X faz?", "Qual a área de X?",
    # "O que é o curso de Y?", "Fale sobre X".
    direct_answer = _try_direct_lookup(query, cfg)
    if direct_answer is not None:
        return direct_answer

    from reliability import record_request, record_fallback

    # --- Roteamento ---
    target = collection_name or route_query(query)

    # --- Extração de keywords (ou usa a query direta) ---
    # use_keyword_extraction controla apenas COMO os keywords são obtidos.
    # A geração sempre usa _synthesize_from_results com top_k_final docs.
    if cfg.use_keyword_extraction:
        from keyword_extractor import extract_keywords
        keywords = extract_keywords(query)
    else:
        keywords = [query]

    # Mínimo de docs a enviar ao LLM (garante cobertura mínima)
    top_k = max(cfg.top_k_final, config.RAG_TOP_K_FINAL)

    if target is None:
        # Roteador não conseguiu identificar coleção — fallback multi-coleção
        record_fallback("rag_pipeline", "roteador sem confiança suficiente", fallback_to="todas as coleções")
        with record_request("rag_pipeline", context="fallback_all_collections"):
            results = search_all_collections(query, top_k=top_k)
        if not results or max(s for _, s in results) < config.RAG_RELEVANCE_THRESHOLD:
            return _NO_RESULTS_MSG
        return _synthesize_from_results(query, results, keywords, cfg, collection_name=None)

    # --- Busca na coleção roteada ---
    try:
        with record_request("rag_pipeline", context=f"vector_search:{target}"):
            if cfg.use_keyword_extraction:
                results = _search_with_keywords(target, keywords, top_k=top_k)
            else:
                store   = get_vector_store(collection_name=target)
                results = store.similarity_search_with_relevance_scores(query, k=top_k)
    except Exception as exc:
        record_fallback("rag_pipeline", f"busca vetorial falhou: {exc}", fallback_to="todas as coleções")
        results = search_all_collections(query, top_k=top_k)

    if not results:
        return _NO_RESULTS_MSG

    best_score = max(s for _, s in results)
    print(f"[Validação] Score: {best_score:.3f} (limiar={config.RAG_RELEVANCE_THRESHOLD})")

    if best_score < config.RAG_RELEVANCE_THRESHOLD:
        # Score baixo na coleção roteada — tenta todas as coleções
        record_fallback(
            "rag_pipeline",
            f"score baixo ({best_score:.3f}) em '{target}'",
            fallback_to="todas as coleções",
        )
        with record_request("rag_pipeline", context="fallback_low_score"):
            fallback = search_all_collections(query, top_k=top_k)
        if fallback and max(s for _, s in fallback) >= config.RAG_RELEVANCE_THRESHOLD:
            resp = _synthesize_from_results(query, fallback, keywords, cfg, collection_name=None)
            _cache_set(query, resp)
            return resp
        if results:
            print(f"[Pipeline] Respondendo com score baixo ({best_score:.3f}) — pode não ser preciso")
            resp = _synthesize_from_results(query, results, keywords, cfg, collection_name=target)
            _cache_set(query, resp)
            return resp
        return _NO_RESULTS_MSG

    resp = _synthesize_from_results(query, results, keywords, cfg, collection_name=target)
    _cache_set(query, resp)
    return resp


def _cache_set(query: str, response: str) -> None:
    try:
        from cache import response_cache
        response_cache.set(query, response)
    except Exception:
        pass


def demo_rag(query: str = "Qual o prazo para entrega da dissertação?"):
    """Demonstra o pipeline RAG em três modos: base, híbrido, completo."""
    print("=" * 62)
    print("  SEÇÃO 6+7 — Pipeline RAG e Validação")
    print("=" * 62)
    print(f"\n  Pergunta: {query}\n")

    store = get_vector_store()

    for label, cfg in [
        ("Base (semântico)",               RAGConfig()),
        ("Híbrido (BM25 + semântico)",     RAGConfig(use_hybrid=True)),
        ("Completo (híbrido + reranker)",  RAGConfig(use_hybrid=True, use_reranker=True)),
    ]:
        print(f"\n--- {label} ---")
        chain, _ = build_rag_chain(store, rag_config=cfg)
        resposta = validate_and_answer(query, chain, store, rag_config=cfg)
        print(resposta)


# =============================================================================
# Execução direta
# python pipeline.py
# python pipeline.py "horário da biblioteca"
# Pré-requisito: python store.py
# =============================================================================
if __name__ == "__main__":
    import sys
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 \
        else "Qual o prazo para entrega da dissertação de mestrado?"
    demo_rag(query)
