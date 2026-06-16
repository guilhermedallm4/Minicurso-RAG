"""
Interface do Chatbot Institucional
=============================================================================
Conceito — SEÇÃO 8 do curso

Loop interativo que integra todos os módulos anteriores:
  - Roteamento automático: perguntas são dirigidas à coleção correta
    (disciplinas, projetos, servidores, cursos, unidades, portal_geral)
  - Ingestão sob demanda (comandos 'ingerir' e 'reset')
  - Demo comparativa de métricas (comando 'demo')

Comandos disponíveis:
  sair                   encerra o chatbot
  demo <pergunta>        compara as 4 métricas de distância
  ingerir <arquivo.pdf>  adiciona um PDF à base
  reset                  recria a coleção do zero e reingesta exemplos
  ajuda                  lista os comandos
"""
import os
from store import ingest_pdf, demo_ingestao
from pipeline import answer_with_routing, RAGConfig
from search import semantic_search_demo


_AJUDA = """
  Comandos disponíveis:
    sair                   — encerra o chatbot
    demo <pergunta>        — compara métricas de distância para a pergunta
    ingerir <arquivo.pdf>  — adiciona um PDF à base vetorial
    reset                  — apaga e reingesta os documentos de exemplo
    ajuda                  — exibe esta mensagem

  O chatbot roteia automaticamente cada pergunta para a base correta:
    → disciplinas  (ementa, carga horária, conteúdo programático…)
    → projetos     (pesquisa, extensão, coordenador, área CNPq…)
    → servidores   (professores, técnicos, lotação…)
    → cursos       (graduação, pós-graduação, grade curricular…)
    → unidades     (institutos, centros, departamentos…)
    → portal_geral (editais, notícias, eventos…)
"""


def run_chatbot():
    """Loop interativo do chatbot com roteamento automático por coleção."""
    print("\n" + "=" * 62)
    print("  CHATBOT INSTITUCIONAL — RAG com Roteamento")
    print(_AJUDA)
    print("=" * 62 + "\n")

    cfg = RAGConfig()

    while True:
        try:
            query = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not query:
            continue

        cmd = query.lower()

        if cmd == "sair":
            print("Encerrando chatbot.")
            break

        if cmd == "ajuda":
            print(_AJUDA)
            continue

        if cmd == "reset":
            demo_ingestao(reset=True)
            continue

        if cmd.startswith("demo "):
            demo_query = query[5:].strip()
            if demo_query:
                semantic_search_demo(demo_query)
            continue

        if cmd.startswith("ingerir "):
            path = query[8:].strip()
            if os.path.isfile(path):
                ingest_pdf(path)
            else:
                print(f"  Arquivo não encontrado: {path}")
            continue

        print("\nAssistente:", end=" ", flush=True)
        response = answer_with_routing(query, rag_config=cfg)
        print(response, "\n")


# =============================================================================
# Execução direta: inicia o chatbot interativo
# python chatbot.py
# Pré-requisito: python store.py (para popular o banco)
# =============================================================================
if __name__ == "__main__":
    run_chatbot()
