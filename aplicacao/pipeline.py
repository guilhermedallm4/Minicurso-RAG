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

import config
from providers import get_llm
from store import get_vector_store
from router import route_query, search_all_collections


# =============================================================================
# Configuração do Pipeline (composicional)
# =============================================================================

@dataclass
class RAGConfig:
    """
    Controla quais módulos avançados são ativados no pipeline.

    Combinações típicas para o curso:
      RAGConfig()                               → pipeline base
      RAGConfig(use_hybrid=True)                → + BM25
      RAGConfig(use_reranker=True)              → + cross-encoder
      RAGConfig(use_hybrid=True,
                use_reranker=True,
                use_guardrails=True)            → pipeline completo
      RAGConfig(use_keyword_extraction=True)    → extrai keywords + síntese
    """
    # --- Retrieval ---
    use_hybrid:      bool  = False   # BM25 + semântica (EnsembleRetriever)
    hybrid_alpha:    float = 0.5     # 0=BM25 puro, 1=semântico puro
    top_k_retrieve:  int   = 8       # candidatos antes do reranking

    # --- Reranking ---
    use_reranker:    bool  = False   # cross-encoder pós-retrieval
    top_k_final:     int   = 4       # docs após reranking (e no pipeline base)

    # --- Keyword Extraction ---
    use_keyword_extraction: bool = False  # extrai keywords da query e sintetiza resultados
    max_chars_per_doc:      int  = 700    # trunca cada doc antes de enviar ao LLM

    # --- Guardrails ---
    use_input_guard:  bool = False   # valida query antes de processar
    use_output_guard: bool = False   # valida resposta antes de entregar


# =============================================================================
# Prompt RAG
# =============================================================================

RAG_PROMPT = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da universidade.
Responda à pergunta usando EXCLUSIVAMENTE as informações do contexto abaixo.
Sempre mencione o título da disciplina ou projeto de onde veio a informação.
Se o contexto não contiver a informação necessária, diga claramente que não encontrou.

CONTEXTO RECUPERADO:
{context}

PERGUNTA DO USUÁRIO: {question}

RESPOSTA:""")

_BASE_INSTRUCTION = (
    "Responda usando EXCLUSIVAMENTE as informações dos documentos acima. "
    "Se uma informação não estiver no contexto, omita o campo (não invente). "
    "Ao final de cada entidade, informe a fonte no formato: "
    "(Fonte: <título> | <URL>)"
)

_PROMPT_SERVIDOR = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da UFPel.
O usuário quer saber: {question}

DOCUMENTOS RECUPERADOS (top {n_docs}):
{context}

{instruction}

Para CADA servidor/professor encontrado, use EXATAMENTE este formato:

---
**Informações de Servidor:**

Nome: <Nome do Servidor>
Matrícula SIAPE: <SIAPE>
Categoria: <Docente / Técnico Administrativo>
Cargo: <Cargo>, Classe <X> / Nível <Y>
Lotação: <Unidade / Departamento>
Regime / Jornada de Trabalho: <Regime> / <Jornada>
Situação: <Situação>
Data de ingresso na UFPel: <Data>

**Atuação e Titulação:**
Titulação: <Titulação>
Áreas de Atuação: <Áreas separadas por " | ">
Formação Acadêmica: <cada formação em linha separada>
Currículo Lattes: <URL do Lattes, se disponível>

**Resumo Profissional:**
<Resumo extraído do Lattes>

(Fonte: <título do documento> | <URL>)
---

RESPOSTA:""")

_PROMPT_DISCIPLINA = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da UFPel.
O usuário quer saber: {question}

DOCUMENTOS RECUPERADOS (top {n_docs}):
{context}

{instruction}

Para CADA disciplina encontrada, use EXATAMENTE este formato:

---
**Disciplina:** <Nome da Atividade>

Código: <Código> | Carga Horária: <CH> | Créditos: <N>
Unidade Responsável: <Departamento/Instituto>
Periodicidade: <Semestral/Anual>
Tipo: <Obrigatória/Optativa>

**Ementa:**
<Texto da ementa>

**Objetivos:**
<Texto dos objetivos, se disponível>

**Conteúdo Programático:**
<Conteúdo, se disponível>

(Fonte: <título do documento> | <URL>)
---

RESPOSTA:""")

_PROMPT_PROJETO = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da UFPel.
O usuário quer saber: {question}

DOCUMENTOS RECUPERADOS (top {n_docs}):
{context}

{instruction}

Para CADA projeto encontrado, use EXATAMENTE este formato:

---
**Projeto:** <Nome do Projeto>

Ênfase: <Pesquisa / Extensão / Ensino>
Coordenador: <Nome do Coordenador>
Unidade de Origem: <Unidade>
Área CNPq: <Área>
Eixo Temático: <Eixo Principal>
Período: <Data inicial> a <Data final>

**Resumo:**
<Resumo do projeto>

**Objetivo Geral:**
<Objetivo, se disponível>

**Equipe:**
<Lista de membros com papel, se disponível>

(Fonte: <título do documento> | <URL>)
---

RESPOSTA:""")

_PROMPT_CURSO = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da UFPel.
O usuário quer saber: {question}

DOCUMENTOS RECUPERADOS (top {n_docs}):
{context}

{instruction}

Para CADA curso encontrado, use EXATAMENTE este formato:

---
**Curso:** <Nome do Curso>

Nível: <Graduação / Pós-Graduação / Técnico>
Turno: <Diurno / Noturno / Integral>
Vagas: <N vagas por ano>
Duração: <N semestres>
Coordenação: <Nome do Coordenador>
Unidade: <Unidade/Instituto responsável>

**Informações Adicionais:**
<Demais informações relevantes disponíveis>

(Fonte: <título do documento> | <URL>)
---

RESPOSTA:""")

_PROMPT_UNIDADE = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da UFPel.
O usuário quer saber: {question}

DOCUMENTOS RECUPERADOS (top {n_docs}):
{context}

{instruction}

Para CADA unidade encontrada, use EXATAMENTE este formato:

---
**Unidade:** <Nome da Unidade> (<SIGLA>)

Código SIORG: <Código>
Direção / Chefia: <Nome>
Endereço / Contato: <Informações de contato, se disponíveis>

**Estrutura:**
<Subunidades ou informações organizacionais disponíveis>

(Fonte: <título do documento> | <URL>)
---

RESPOSTA:""")

_PROMPT_GERAL = ChatPromptTemplate.from_template("""
Você é um assistente especializado em documentos institucionais da UFPel.
O usuário quer saber: {question}

DOCUMENTOS RECUPERADOS (top {n_docs}):
{context}

{instruction}

Para cada item encontrado, apresente as informações de forma organizada com
título em negrito, campos relevantes e ao final a fonte no formato
(Fonte: <título> | <URL>).
Se a pergunta for sobre uma entidade específica, responda diretamente.
Se envolver múltiplos itens, liste cada um separadamente.

RESPOSTA:""")

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

    max_chars_per_doc > 0 trunca o conteúdo de cada doc para evitar estourar o contexto da API.
    """
    parts = []
    for d in docs:
        titulo = d.metadata.get("titulo", "")
        source = d.metadata.get("source", "documento")
        header = f"[Título: {titulo} | Fonte: {source}]" if titulo else f"[Fonte: {source}]"
        content = d.page_content
        if max_chars_per_doc > 0 and len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc] + "…"
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

    # --- Guard de entrada ---
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

    if best_score < config.RELEVANCE_THRESHOLD:
        return (
            f"Não encontrei documentos relevantes o suficiente "
            f"(score={best_score:.3f} < {config.RELEVANCE_THRESHOLD}).\n"
            "Tente reformular a pergunta ou verifique se o documento foi ingerido."
        )

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
    llm      = get_llm()
    prompt   = _get_prompt(collection_name)
    response = (prompt | llm | StrOutputParser()).invoke({
        "context":     context,
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

    # --- Guard de entrada ---
    if cfg.use_input_guard:
        from guardrails import DEFAULT_INPUT_GUARD
        result = DEFAULT_INPUT_GUARD.check(query)
        if not result.passed:
            print(f"[Guard:entrada:{result.check}] Bloqueado")
            return f"Pergunta bloqueada: {result.reason}"

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

    if target is None:
        # Fallback: busca em todas as coleções
        results = search_all_collections(query, top_k=cfg.top_k_final)
        if not results or max(s for _, s in results) < config.RELEVANCE_THRESHOLD:
            return (
                "Não encontrei documentos relevantes o suficiente em nenhuma coleção.\n"
                "Tente reformular a pergunta ou verifique se os dados foram ingeridos."
            )
        return _synthesize_from_results(query, results, keywords, cfg, collection_name=None)

    # --- Busca na coleção roteada ---
    if cfg.use_keyword_extraction:
        results = _search_with_keywords(target, keywords, top_k=cfg.top_k_final)
    else:
        store   = get_vector_store(collection_name=target)
        results = store.similarity_search_with_relevance_scores(query, k=cfg.top_k_final)

    if not results:
        return "Não há documentos na base. Execute a ingestão primeiro."

    best_score = max(s for _, s in results)
    print(f"[Validação] Score: {best_score:.3f} (limiar={config.RELEVANCE_THRESHOLD})")

    if best_score < config.RELEVANCE_THRESHOLD:
        print("[Pipeline] Score baixo — tentando fallback multi-coleção")
        fallback = search_all_collections(query, top_k=cfg.top_k_final)
        if fallback and max(s for _, s in fallback) >= config.RELEVANCE_THRESHOLD:
            return _synthesize_from_results(query, fallback, keywords, cfg, collection_name=None)
        return (
            f"Não encontrei documentos relevantes o suficiente "
            f"(score={best_score:.3f} < {config.RELEVANCE_THRESHOLD}).\n"
            "Tente reformular a pergunta ou verifique se o documento foi ingerido."
        )

    return _synthesize_from_results(query, results, keywords, cfg, collection_name=target)


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
