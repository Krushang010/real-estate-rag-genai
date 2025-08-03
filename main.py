# @Author: Krushang Patel | GenAI RAG UI (FAISS)

import streamlit as st
from rag import load_and_prepare_docs, get_qa_chain, run_rag_pipeline

# --- Custom CSS Styling ---
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .block-container {
            padding: 2rem 3rem;
        }
        h1 {
            color: #4ADE80;
        }
        .stTextInput>div>div>input {
            background-color: #1E1E1E;
            color: #FAFAFA;
        }
        .stButton>button {
            background-color: #3B82F6;
            color: white;
            font-weight: 600;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
        }
        .stTextInput>div>label, .stSidebar>label {
            color: #A5B4FC;
            font-weight: 500;
        }
        .stCheckbox>div {
            color: #FACC15;
        }
        .highlight {
            background-color: #1F2937;
            padding: 1rem;
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🏠 Real Estate Research Assistant")

# --- Sidebar: Input URLs ---
st.sidebar.header("🔗 Input URLs")
url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")
reset_vectorstore = st.sidebar.checkbox("🗑️ Reset Vectorstore (Fresh Load)")
process_url_button = st.sidebar.button("🔄 Process URLs")

placeholder = st.empty()

# --- Session State Init ---
if "vectorstore" not in st.session_state or reset_vectorstore:
    st.session_state.vectorstore = None
    st.session_state.qa_chain = None
    st.session_state.llm = None

# --- URL Processing ---
if process_url_button:
    urls = [url for url in [url1, url2, url3] if url.strip()]
    if not urls:
        placeholder.warning("❗ Please provide at least one valid URL.")
    else:
        try:
            placeholder.info("⏳ Processing URLs and embedding...")
            vectorstore = load_and_prepare_docs(urls=urls)
            qa_chain, llm = get_qa_chain(vectorstore, return_sources=True)
            st.session_state.vectorstore = vectorstore
            st.session_state.qa_chain = qa_chain
            st.session_state.llm = llm
            placeholder.success("✅ URLs processed and vectorstore is ready.")
        except Exception as e:
            placeholder.error(f"❌ Failed to process URLs: {e}")

# --- Ask Question Section ---
st.markdown("---")
query = st.text_input("💬 Ask your question about real estate")

if query:
    if not st.session_state.qa_chain or not st.session_state.llm:
        st.warning("⚠️ Please process URLs first.")
    else:
        try:
            summary, sources = run_rag_pipeline(
                question=query,
                qa_chain=st.session_state.qa_chain,
                llm=st.session_state.llm
            )

            st.markdown("### 📌 Answer")
            st.markdown(f"<div class='highlight'>{summary}</div>", unsafe_allow_html=True)

            if sources:
                st.markdown("#### 🔗 Sources")
                for src in sources:
                    st.markdown(f"- {src}")

        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")
