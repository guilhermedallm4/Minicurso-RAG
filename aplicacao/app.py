"""
Interface Streamlit — Chatbot RAG · H2IA / UFPel
=============================================================================
Execução:
  cd aplicacao
  streamlit run app.py
"""
import sys
import os
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────────────────────────────────────
_FAVICON_PATH = os.path.join(os.path.dirname(__file__), "static", "favicon.png")
_favicon = Image.open(_FAVICON_PATH) if os.path.exists(_FAVICON_PATH) else "🤖"

st.set_page_config(
    page_title="H2IA · Assistente UFPel",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Injeta <link rel="icon"> com o logo em base64 — garante que o navegador
# use o favicon correto independente do que o Streamlit injeta por padrão.
if os.path.exists(_FAVICON_PATH):
    import base64 as _b64
    with open(_FAVICON_PATH, "rb") as _f:
        _favicon_b64 = _b64.b64encode(_f.read()).decode()
    st.markdown(
        f'<link rel="shortcut icon" href="data:image/png;base64,{_favicon_b64}">',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CSS — paleta roxa clara + branco, visual limpo
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Layout base — ocupa toda a altura ──────────────────────────────────────── */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 1rem !important;
    max-width: 900px;
}

/* ── Header ─────────────────────────────────────────────────────────────────── */
.h2ia-header {
    background: linear-gradient(130deg, #7B2FBE 0%, #9B59D0 60%, #7B2FBE 100%);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 16px;
    box-shadow: 0 4px 18px rgba(123, 47, 190, 0.22);
}
.h2ia-header h1 {
    color: #FFFFFF;
    font-size: 1.3rem;
    font-weight: 600;
    margin: 0;
    line-height: 1.25;
}
.h2ia-header p {
    color: rgba(255, 255, 255, 0.70);
    font-size: 0.8rem;
    margin: 4px 0 0 0;
}
.h2ia-badge {
    background: rgba(255, 255, 255, 0.18);
    border: 1px solid rgba(255, 255, 255, 0.32);
    color: #FFFFFF;
    border-radius: 6px;
    padding: 4px 9px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: #F8F5FC !important;
    border-right: 1px solid #E5D9F5 !important;
}
section[data-testid="stSidebar"] * {
    color: #3D1A6B !important;
}
.sidebar-logo {
    text-align: center;
    padding: 20px 0 12px 0;
}
.sidebar-logo .logo-sub {
    color: #9B7FBE !important;
    font-size: 0.72rem;
    margin-top: 3px;
}
.sidebar-section-title {
    color: #7B2FBE !important;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 16px 0 10px 0;
    padding-bottom: 5px;
    border-bottom: 2px solid #E5D9F5;
}
.sidebar-footer {
    text-align: center;
    color: #B09FCF !important;
    font-size: 0.7rem;
    padding: 8px 0;
}
section[data-testid="stSidebar"] hr {
    border-color: #E5D9F5 !important;
    margin: 12px 0 !important;
}

/* ── Selectbox ──────────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] span {
    color: #3D1A6B !important;
    font-size: 0.85rem !important;
}
[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #D4BEF0 !important;
    border-radius: 8px !important;
}
[data-baseweb="select"] span { color: #3D1A6B !important; }

/* ── Toggle ─────────────────────────────────────────────────────────────────── */
[data-testid="stToggle"] label {
    color: #3D1A6B !important;
    font-size: 0.85rem !important;
}

/* ── Botões ─────────────────────────────────────────────────────────────────── */
.stButton > button {
    background: #7B2FBE !important;
    border: none !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: background 0.2s ease !important;
}
.stButton > button:hover { background: #6A22A8 !important; }

/* ── Área de mensagens — scroll interno ─────────────────────────────────────── */
.chat-area {
    height: calc(100vh - 260px);
    min-height: 300px;
    overflow-y: auto;
    padding: 0 4px 8px 4px;
    display: flex;
    flex-direction: column;
}

/* ── Chat messages ──────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 6px !important;
    max-width: 100% !important;
}

/* ── Chat input — remove container externo, mantém só o textarea ────────────── */
[data-testid="stChatInput"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
/* Div wrapper interno que cria o fundo cinza */
[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}
[data-testid="stChatInput"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #D4BEF0 !important;
    background: #FFFFFF !important;
    font-size: 0.92rem !important;
    padding: 14px 48px 14px 18px !important;
    resize: none !important;
    min-height: 52px !important;
    max-height: 120px !important;
    box-shadow: 0 2px 10px rgba(123, 47, 190, 0.08) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #7B2FBE !important;
    box-shadow: 0 0 0 3px rgba(123, 47, 190, 0.10) !important;
    outline: none !important;
}
/* Remove segundo textarea residual */
[data-testid="stChatInput"] textarea + textarea {
    display: none !important;
}

/* ── Timing tag ─────────────────────────────────────────────────────────────── */
.timing-tag {
    color: #B09FCF;
    font-size: 0.73rem;
}

/* ── Aviso ──────────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVIDADE MOBILE (≤ 768 px)
   ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    /* Padding mínimo nas laterais */
    .main .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
        padding-top: 0.75rem !important;
    }

    /* Header compacto: empilha verticalmente */
    .h2ia-header {
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
        padding: 16px 18px;
        border-radius: 10px;
    }
    .h2ia-header h1 { font-size: 1.05rem; }
    .h2ia-header p  { font-size: 0.78rem; }
    .h2ia-badge     { font-size: 0.65rem; }

    /* Mensagens ocupam largura total */
    [data-testid="stChatMessage"] {
        border-radius: 10px !important;
        font-size: 0.88rem !important;
    }

    /* Input de texto maior para toque */
    [data-testid="stChatInput"] > div { background: transparent !important; border: none !important; }
    [data-testid="stChatInput"] textarea {
        font-size: 1rem !important;
        min-height: 56px !important;
        padding: 16px 48px 16px 18px !important;
    }

    /* Sidebar ocupa tela inteira quando aberta */
    section[data-testid="stSidebar"] {
        width: 100vw !important;
        min-width: unset !important;
    }
    .sidebar-logo { padding: 12px 0 8px 0; }

    /* Área de chat com menos altura reservada */
    .chat-area {
        height: calc(100vh - 220px);
        min-height: 240px;
    }
}

@media (max-width: 480px) {
    .h2ia-header h1 { font-size: 0.95rem; }
    .h2ia-badge { display: none; }  /* muito pequeno — remove badge */
    [data-testid="stChatMessage"] { font-size: 0.85rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="h2ia-header">
    <div style="font-size:2.4rem; line-height:1;">🤖</div>
    <div style="flex:1;">
        <h1>Assistente Inteligente UFPel</h1>
        <p>Pergunte sobre cursos, disciplinas, professores, projetos e unidades institucionais.</p>
    </div>
    <div class="h2ia-badge">H2IA · UFPel</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Lock de concorrência
# ─────────────────────────────────────────────────────────────────────────────
if "processing_lock" not in st.session_state:
    st.session_state.processing_lock = threading.Lock()

_GLOBAL_LOCK: threading.Lock = st.session_state.processing_lock

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div style="
            display:inline-flex;
            align-items:center;
            justify-content:center;
            background: linear-gradient(135deg, #7B2FBE 0%, #9B59D0 100%);
            border-radius: 14px;
            padding: 14px 20px;
            box-shadow: 0 2px 12px rgba(123,47,190,0.20);
        ">
            <img src="https://ia.ufpel.edu.br/H2IALogo.png"
                 alt="H2IA"
                 style="width:100px;"
                 onerror="this.style.display='none'">
        </div>
        <div class="logo-sub" style="margin-top:10px;">Hub de Inovação em IA · UFPel</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown('<p class="sidebar-section-title">Configurações de busca</p>', unsafe_allow_html=True)

    modo = st.selectbox(
        "Modo de recuperação",
        options=["base", "hibrido", "completo"],
        format_func=lambda x: {
            "base":     "Semântico (padrão)",
            "hibrido":  "Híbrido (BM25 + semântico)",
            "completo": "Completo (híbrido + reranker)",
        }[x],
        help="Define como os documentos são recuperados antes de enviar ao LLM.",
    )

    usar_keywords = st.toggle(
        "Extração de palavras-chave",
        value=False,
        help=(
            "Usa um LLM para identificar os termos mais relevantes da pergunta "
            "antes de buscar. Aumenta o recall em perguntas temáticas, "
            "mas adiciona ~1s de latência."
        ),
    )

    st.divider()

    st.markdown(
        '<div class="sidebar-footer">h2ia@inf.ufpel.edu.br · ia.ufpel.edu.br</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Histórico de conversa
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(
            "Olá! Sou o assistente do **Hub de Inovação em Inteligência Artificial da UFPel**.\n\n"
            "Posso responder perguntas sobre:\n"
            "- 🎓 **Cursos e disciplinas** — grades curriculares, créditos, semestres\n"
            "- 👤 **Professores** — áreas de atuação, pesquisa, projetos\n"
            "- 📋 **Projetos de pesquisa** — projetos vigentes, coordenadores, equipes\n"
            "- 🏛️ **Unidades e centros** — localização, contato, estrutura\n\n"
            "Como posso ajudar?"
        )

for msg in st.session_state.messages:
    avatar = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────────────────────────────────────
# Input e pipeline RAG
# ─────────────────────────────────────────────────────────────────────────────
pergunta = st.chat_input("Digite sua pergunta sobre a UFPel…")

if pergunta:
    lock_acquired = _GLOBAL_LOCK.acquire(blocking=False)

    if not lock_acquired:
        st.warning("⏳ Uma pergunta já está sendo processada. Aguarde a resposta antes de enviar outra.")
    else:
        try:
            st.session_state.messages.append({"role": "user", "content": pergunta})
            with st.chat_message("user", avatar="👤"):
                st.markdown(pergunta)

            with st.chat_message("assistant", avatar="🤖"):
                spinner_msg = (
                    "Extraindo keywords e buscando na base…"
                    if usar_keywords
                    else "Buscando na base de conhecimento…"
                )
                with st.spinner(spinner_msg):
                    t_start = time.perf_counter()
                    timing_tag = ""
                    try:
                        from pipeline import RAGConfig, answer_with_routing
                        from reliability import record_request

                        cfg = RAGConfig(
                            use_hybrid=(modo in ("hibrido", "completo")),
                            use_reranker=(modo == "completo"),
                            use_keyword_extraction=usar_keywords,
                        )

                        with record_request("rag_pipeline", context="answer_with_routing"):
                            resposta = answer_with_routing(pergunta, rag_config=cfg)

                        elapsed = time.perf_counter() - t_start
                        extras = f"{modo}{' · keywords' if usar_keywords else ''}"
                        timing_tag = (
                            f"\n\n<span class='timing-tag'>⏱ {elapsed:.1f}s · {extras}</span>"
                        )

                    except Exception as exc:
                        from reliability import record_fallback
                        record_fallback("rag_pipeline", f"exceção não tratada: {exc}")
                        resposta = f"❌ Erro ao processar a pergunta: {exc}"
                        timing_tag = ""

                st.markdown(resposta + timing_tag, unsafe_allow_html=True)

            # Salva sem o timing tag — evita HTML cru ao re-renderizar o histórico
            st.session_state.messages.append({"role": "assistant", "content": resposta})

        finally:
            _GLOBAL_LOCK.release()
