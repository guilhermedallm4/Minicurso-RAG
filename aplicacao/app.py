"""
Interface Streamlit — Chatbot RAG Institucional
=============================================================================
Execução:
  cd aplicacao
  streamlit run app.py
"""
import sys
import os
import tempfile

# garante que os módulos do projeto sejam encontrados
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Institucional",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Chatbot RAG Institucional")
st.caption(
    "Faça perguntas sobre disciplinas, projetos, servidores, cursos ou unidades. "
    "O sistema identifica automaticamente a base correta para cada pergunta."
)




# ─────────────────────────────────────────────────────────────────────────────
# Barra lateral — configurações
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurações do Pipeline")

    modo = st.selectbox(
        "Modo de recuperação",
        options=["base", "hibrido", "completo"],
        format_func=lambda x: {
            "base":     "Base (semântico)",
            "hibrido":  "Híbrido (BM25 + semântico)",
            "completo": "Completo (híbrido + reranker)",
        }[x],
    )

    usar_keywords = st.toggle(
        "Extração de palavras-chave",
        value=False,
        help=(
            "Ativa extração de keywords da pergunta antes da busca. "
            "Ideal para perguntas do tipo 'Quais projetos de X?', onde o sistema "
            "identifica os termos-chave, busca na coleção por cada um e "
            "sintetiza os resultados encontrados."
        ),
    )

    st.divider()

    st.subheader("📥 Ingestão de documentos")

    if st.button("Ingerir documentos de exemplo", use_container_width=True):
        with st.spinner("Ingerindo documentos de exemplo…"):
            try:
                from store import demo_ingestao
                demo_ingestao(reset=False)
                st.cache_resource.clear()
                st.success("Documentos ingeridos com sucesso!")
            except Exception as e:
                st.error(f"Erro na ingestão: {e}")

    if st.button("Recriar coleção do zero", use_container_width=True):
        with st.spinner("Recriando coleção e reingerindo exemplos…"):
            try:
                from store import demo_ingestao
                demo_ingestao(reset=True)
                st.cache_resource.clear()
                st.success("Coleção recriada com sucesso!")
            except Exception as e:
                st.error(f"Erro: {e}")

    uploaded = st.file_uploader("Enviar PDF", type=["pdf"])
    if uploaded:
        if st.button("Ingerir PDF enviado", use_container_width=True):
            with st.spinner(f"Ingerindo '{uploaded.name}'…"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    from store import ingest_pdf
                    ingest_pdf(tmp_path, source_name=uploaded.name)
                    os.unlink(tmp_path)
                    st.cache_resource.clear()
                    st.success(f"'{uploaded.name}' ingerido com sucesso!")
                except Exception as e:
                    st.error(f"Erro ao ingerir PDF: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Histórico de conversa
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─────────────────────────────────────────────────────────────────────────────
# Input e pipeline RAG
# ─────────────────────────────────────────────────────────────────────────────
pergunta = st.chat_input("Digite sua pergunta…")

if pergunta:
    # exibe pergunta
    st.session_state.messages.append({"role": "user", "content": pergunta})
    with st.chat_message("user"):
        st.markdown(pergunta)

    with st.chat_message("assistant"):
        spinner_msg = (
            "Extraindo keywords, buscando e sintetizando…"
            if usar_keywords
            else "Identificando base e processando RAG…"
        )
        with st.spinner(spinner_msg):
            try:
                from pipeline import RAGConfig, answer_with_routing

                cfg = RAGConfig(
                    use_hybrid=(modo in ("hibrido", "completo")),
                    use_reranker=(modo == "completo"),
                    use_keyword_extraction=usar_keywords,
                )

                resposta = answer_with_routing(pergunta, rag_config=cfg)

            except Exception as e:
                resposta = f"❌ Erro ao processar a pergunta: {e}"

        st.markdown(resposta)

    st.session_state.messages.append({"role": "assistant", "content": resposta})
