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
    "Você é um extrator de palavras-chave para sistemas de busca em documentos institucionais.\n"
    "Dada a pergunta abaixo, extraia as palavras-chave mais relevantes para localizar os documentos corretos.\n\n"
    "Pergunta: {query}\n\n"
    "Regras:\n"
    "  - Extraia entre 1 e 4 palavras-chave ou expressões\n"
    "  - Inclua siglas e sinônimos relevantes\n"
    "  - Não inclua palavras genéricas como 'projeto', 'disciplina', 'quais', 'relacionados'\n"
    "  - Priorize termos técnicos, áreas, temas específicos\n\n"
    "Exemplos:\n"
    "  'Quais projetos de IA?' → [\"Inteligência Artificial\", \"IA\", \"machine learning\"]\n"
    "  'Projetos sobre meio ambiente?' → [\"meio ambiente\", \"sustentabilidade\", \"ecologia\"]\n"
    "  'Projetos de saúde pública?' → [\"saúde pública\", \"epidemiologia\", \"saúde coletiva\"]\n\n"
    "Responda SOMENTE com JSON válido, sem texto adicional:\n"
    '  {{"keywords": ["termo1", "termo2"]}}'
)


def extract_keywords(query: str) -> list[str]:
    """
    Extrai palavras-chave da query usando LLM.

    Retorna lista de termos relevantes para busca, ou [query] como fallback.
    """
    llm = get_llm()
    chain = _KEYWORD_PROMPT | llm | StrOutputParser()

    try:
        raw = chain.invoke({"query": query})
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            print(f"[Keywords] Sem JSON na resposta — usando query original")
            return [query]
        data = json.loads(match.group())
        keywords = [kw.strip() for kw in data.get("keywords", []) if kw.strip()]
        if keywords:
            print(f"[Keywords] Extraídas: {keywords}")
            return keywords
    except Exception as exc:
        print(f"[Keywords] Erro: {exc} — usando query original")

    return [query]


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
