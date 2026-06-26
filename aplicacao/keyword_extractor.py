"""
Extração de palavras-chave da query do usuário via LLM.
=============================================================================
Usado após o roteamento para expandir a busca semântica com termos relevantes.

Exemplo:
  "Quais projetos relacionados a Inteligência Artificial?"
  → ["Inteligência Artificial", "IA", "machine learning", "aprendizado de máquina"]

Essas keywords são usadas para fazer buscas independentes na coleção
identificada pelo roteador, aumentando o recall sem explodir o contexto.
"""
import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from providers import get_llm

_KEYWORD_PROMPT = ChatPromptTemplate.from_template(
    "Você é um extrator de palavras-chave para busca em documentos institucionais da UFPel.\n\n"
    "Dada a pergunta abaixo, extraia os termos mais relevantes para localizar documentos "
    "sobre professores, cursos, disciplinas, projetos de pesquisa e unidades.\n\n"
    "Pergunta: {query}\n\n"
    "Regras:\n"
    "  - Extraia entre 1 e 5 termos ou expressões\n"
    "  - Para áreas de pesquisa/atuação: inclua a área principal E subáreas relacionadas\n"
    "    Ex: 'computação quântica' → ['computação quântica', 'algoritmos quânticos', 'física quântica']\n"
    "    Ex: 'inteligência artificial' → ['Inteligência Artificial', 'machine learning', 'deep learning']\n"
    "    Ex: 'linguagem natural' → ['Processamento de Linguagem Natural', 'NLP', 'PLN', 'análise de sentimento']\n"
    "  - Use os termos EXATOS como aparecem em currículos e resumos acadêmicos brasileiros\n"
    "  - Inclua a sigla quando existe uma consagrada (IA, NLP, PLN, IoT, HCI etc.)\n"
    "  - Nomes de pessoas: mantenha COMPLETOS (ex: 'Ulisses Brisolara Corrêa')\n"
    "  - NÃO inclua: artigos, preposições, verbos genéricos, ou termos fora do escopo da pergunta\n\n"
    "Exemplos:\n"
    "  'Qual professor pesquisa computação quântica?' → [\"computação quântica\", \"algoritmos quânticos\", \"mecânica quântica\"]\n"
    "  'Quem trabalha com visão computacional?' → [\"visão computacional\", \"processamento de imagens\", \"computer vision\"]\n"
    "  'Professor que atua com segurança?' → [\"Segurança da Informação\", \"cibersegurança\", \"criptografia\"]\n"
    "  'Quais projetos de IA existem?' → [\"Inteligência Artificial\", \"machine learning\", \"deep learning\"]\n"
    "  'Projetos sobre meio ambiente?' → [\"meio ambiente\", \"sustentabilidade\", \"ecologia\"]\n\n"
    "Responda SOMENTE com JSON válido, sem texto adicional:\n"
    '  {{"keywords": ["termo1", "termo2"]}}'
)

_AREA_KEYWORD_PROMPT = ChatPromptTemplate.from_template(
    "Você é um extrator de termos de busca acadêmica para documentos da UFPel.\n\n"
    "A pergunta abaixo é sobre a área de atuação ou pesquisa de professores. "
    "Extraia os termos que aparecem em currículos Lattes, resumos de pesquisa e "
    "projetos acadêmicos brasileiros para descrever essa área.\n\n"
    "Pergunta: {query}\n\n"
    "Retorne entre 2 e 6 termos, priorizando:\n"
    "  1. O termo exato da área (ex: 'Computação Quântica')\n"
    "  2. Subáreas e tópicos relacionados (ex: 'algoritmos quânticos', 'qubits')\n"
    "  3. Siglas consagradas (ex: 'NLP', 'PLN', 'IA', 'IoT')\n"
    "  4. Termos em inglês quando usados no Brasil (ex: 'machine learning', 'deep learning')\n\n"
    "NÃO inclua nomes de cursos de graduação, apenas áreas de pesquisa/atuação.\n\n"
    "Exemplos:\n"
    "  'computação quântica' → [\"computação quântica\", \"algoritmos quânticos\", \"informação quântica\", \"qubits\"]\n"
    "  'inteligência artificial' → [\"Inteligência Artificial\", \"machine learning\", \"deep learning\", \"redes neurais\"]\n"
    "  'linguagem natural' → [\"Processamento de Linguagem Natural\", \"PLN\", \"NLP\", \"análise de sentimento\"]\n"
    "  'banco de dados' → [\"banco de dados\", \"bases de dados\", \"SQL\", \"NoSQL\", \"data warehouse\"]\n"
    "  'robótica' → [\"robótica\", \"sistemas embarcados\", \"automação\", \"controle\"]\n\n"
    "Responda SOMENTE com JSON válido:\n"
    '  {{"keywords": ["termo1", "termo2"]}}'
)


def _invoke_prompt(prompt, query: str) -> list[str]:
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": query})
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not match:
        return []
    data = json.loads(match.group())
    return [kw.strip() for kw in data.get("keywords", []) if kw.strip()]


def extract_keywords(query: str) -> list[str]:
    """
    Extrai palavras-chave da query usando LLM.

    Retorna lista de termos relevantes para busca, ou [query] como fallback.
    """
    try:
        keywords = _invoke_prompt(_KEYWORD_PROMPT, query)
        if keywords:
            print(f"[Keywords] Extraídas: {keywords}")
            return keywords
    except Exception as exc:
        print(f"[Keywords] Erro: {exc} — usando query original")

    return [query]


def extract_area_keywords(query: str) -> list[str]:
    """
    Extrai termos de área de atuação/pesquisa para busca em perfis de professores.

    Usa prompt especializado em vocabulário acadêmico (Lattes, currículos, projetos).
    Retorna lista de termos ou [] se não conseguir extrair.
    """
    try:
        keywords = _invoke_prompt(_AREA_KEYWORD_PROMPT, query)
        if keywords:
            print(f"[Área:keywords] Extraídas: {keywords}")
            return keywords
    except Exception as exc:
        print(f"[Área:keywords] Erro: {exc}")

    return []


# =============================================================================
# Execução direta — teste rápido
# python keyword_extractor.py "Quais projetos de Inteligência Artificial?"
# =============================================================================
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Quais projetos relacionados a Inteligência Artificial?"
    print(f"Pergunta: {q}")
    kws = extract_keywords(q)
    print(f"Keywords: {kws}")
