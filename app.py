"""
app.py — AI Research Assistant  |  Streamlit UI
Run:  streamlit run app.py
"""
import time
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.config import settings
from src.ingestion import ingest_file, load_url
from src.vectorstore import get_vector_store
from src.rag_chain import RAGEngine, summarise_source
from src.llm import get_provider_info
from src.utils import is_valid_url, clean_url, source_icon, truncate
from src.logger import logger

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
  --primary:#6C63FF; --primary-light:#8B85FF; --accent:#FF6584;
  --success:#4ECDC4; --surface:#1A1A2E; --surface2:#16213E;
  --text:#E8E8F0; --text-muted:#9090B0;
  --border:rgba(108,99,255,0.25); --radius:12px;
}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:var(--text);}
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"]{display:none !important;}
.stApp{background:linear-gradient(135deg,#0D0D1A 0%,#1A1A2E 50%,#0F1923 100%);}

.navbar{display:flex;align-items:center;justify-content:space-between;
  padding:0.75rem 2rem;background:rgba(22,33,62,0.9);
  border-bottom:1px solid var(--border);backdrop-filter:blur(12px);
  position:sticky;top:0;z-index:999;margin:-1rem -1rem 1.5rem -1rem;}
.navbar-brand{font-family:'DM Serif Display',serif;font-size:1.3rem;
  background:linear-gradient(135deg,#6C63FF,#FF6584);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.navbar-badge{font-size:0.72rem;padding:0.2rem 0.7rem;border-radius:999px;
  font-family:'JetBrains Mono',monospace;background:rgba(108,99,255,0.15);
  border:1px solid rgba(108,99,255,0.3);color:var(--primary-light);}

.panel-title{font-size:0.78rem;font-weight:600;letter-spacing:0.08em;
  text-transform:uppercase;color:var(--text-muted);margin-bottom:0.5rem;}
.stat-card{background:rgba(22,33,62,0.7);border:1px solid var(--border);
  border-radius:var(--radius);padding:0.8rem;text-align:center;margin-bottom:0.5rem;}
.stat-number{font-family:'DM Serif Display',serif;font-size:1.8rem;color:var(--primary-light);}
.stat-label{font-size:0.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;}

.chat-bubble{padding:1rem 1.25rem;border-radius:var(--radius);
  margin:0.6rem 0;line-height:1.6;animation:fadeUp 0.25s ease;}
.chat-bubble.user{background:linear-gradient(135deg,rgba(108,99,255,0.18),rgba(108,99,255,0.08));
  border:1px solid rgba(108,99,255,0.3);margin-left:8%;}
.chat-bubble.assistant{background:rgba(22,33,62,0.55);border:1px solid var(--border);margin-right:4%;}
.chat-label{font-size:0.7rem;font-weight:600;letter-spacing:0.08em;
  text-transform:uppercase;margin-bottom:0.4rem;}
.chat-label.user{color:var(--primary-light);}
.chat-label.assistant{color:var(--success);}
@keyframes fadeUp{from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:translateY(0);}}

.source-pill{display:inline-block;padding:0.18rem 0.6rem;border-radius:999px;
  font-size:0.7rem;font-family:'JetBrains Mono',monospace;
  background:rgba(108,99,255,0.12);border:1px solid rgba(108,99,255,0.28);
  color:var(--primary-light);margin:0.15rem;}

.stButton>button{background:linear-gradient(135deg,var(--primary),#5A54D4) !important;
  color:#fff !important;border:none !important;border-radius:8px !important;
  font-family:'DM Sans',sans-serif !important;font-weight:500 !important;transition:all 0.2s !important;}
.stButton>button:hover{transform:translateY(-1px) !important;
  box-shadow:0 4px 18px rgba(108,99,255,0.4) !important;}

.stTextInput>div>div>input,.stTextArea>div>div>textarea{
  background:rgba(22,33,62,0.8) !important;border:1px solid var(--border) !important;
  color:var(--text) !important;border-radius:8px !important;}
[data-testid="stFileUploader"]{border:2px dashed var(--border) !important;
  border-radius:var(--radius) !important;background:rgba(22,33,62,0.3) !important;}
.streamlit-expanderHeader{background:rgba(22,33,62,0.5) !important;border-radius:8px !important;}
.stTabs [data-baseweb="tab-list"]{background:transparent !important;gap:4px;}
.stTabs [data-baseweb="tab"]{background:rgba(22,33,62,0.5) !important;
  border-radius:8px 8px 0 0 !important;color:var(--text-muted) !important;font-size:0.88rem !important;}
.stTabs [aria-selected="true"]{background:rgba(108,99,255,0.2) !important;
  color:var(--primary-light) !important;border-bottom:2px solid var(--primary) !important;}
code{font-family:'JetBrains Mono',monospace !important;background:rgba(108,99,255,0.1) !important;
  color:var(--primary-light) !important;padding:0.1em 0.35em !important;border-radius:4px !important;}
hr{border-color:var(--border) !important;}
</style>
""", unsafe_allow_html=True)


def _init_state():
    for k, v in {
        "messages": [], "rag_engine": None,
        "last_sources": [], "followup_questions": [],
        "selected_source_filter": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _get_engine():
    if st.session_state.rag_engine is None:
        with st.spinner("🔧 Initialising AI engine…"):
            st.session_state.rag_engine = RAGEngine()
    return st.session_state.rag_engine


def render_navbar():
    info = get_provider_info()
    st.markdown(f"""
    <div class="navbar">
      <span class="navbar-brand">🔬 AI Research Assistant</span>
      <div style="display:flex;gap:8px;align-items:center;">
        <span class="navbar-badge">{info['icon']} {info['provider']} · {info['model']}</span>
        <span class="navbar-badge">🧩 {settings.embedding_provider} embeddings</span>
      </div>
    </div>""", unsafe_allow_html=True)


def render_upload_panel():
    vs = get_vector_store()
    stats = vs.get_stats()
    is_empty = vs.is_empty()

    with st.expander("📂  Add Documents to Knowledge Base", expanded=is_empty):
        col_file, col_url, col_stats = st.columns([2, 2, 1])

        with col_file:
            st.markdown('<div class="panel-title">📄 Upload Files (PDF / DOCX / TXT)</div>', unsafe_allow_html=True)
            uploaded_files = st.file_uploader(
                "files", type=["pdf","docx","doc","txt","md"],
                accept_multiple_files=True, label_visibility="collapsed",
            )
            if uploaded_files:
                if st.button("⚡ Ingest Files", use_container_width=True):
                    _ingest_files(uploaded_files)

        with col_url:
            st.markdown('<div class="panel-title">🌐 Add Web Page URL</div>', unsafe_allow_html=True)
            url_input = st.text_input("url", placeholder="https://arxiv.org/abs/...",
                                      label_visibility="collapsed")
            if st.button("🌍 Fetch & Ingest URL", use_container_width=True):
                if url_input.strip():
                    _ingest_url(url_input.strip())
                else:
                    st.warning("Please enter a URL first.")

        with col_stats:
            st.markdown('<div class="panel-title">📊 KB Stats</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-card">
              <div class="stat-number">{stats['total_chunks']}</div>
              <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-card">
              <div class="stat-number">{stats['total_sources']}</div>
              <div class="stat-label">Sources</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        s1, s2, s3 = st.columns([2, 2, 1])
        with s1:
            new_k = st.slider("Top-K Results", 1, 12, settings.top_k_results)
            settings.top_k_results = new_k
        with s2:
            src_filter = st.selectbox("Filter by Source",
                                      ["All Sources"] + vs.list_sources())
            st.session_state.selected_source_filter = (
                None if src_filter == "All Sources" else src_filter)
        with s3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🗑️ Clear Chat", use_container_width=True):
                if st.session_state.rag_engine:
                    st.session_state.rag_engine.clear_memory()
                st.session_state.messages = []
                st.session_state.followup_questions = []
                st.rerun()


def _ingest_files(uploaded_files):
    vs = get_vector_store()
    total_added = 0
    progress = st.progress(0, text="Starting…")
    msgs_placeholder = st.empty()
    msgs = []
    for i, f in enumerate(uploaded_files):
        progress.progress((i + 0.5) / len(uploaded_files), text=f"Processing {f.name}…")
        try:
            docs = ingest_file(f.read(), f.name)
            added, skipped = vs.add_documents(docs)
            total_added += added
            msgs.append(f"✅ **{f.name}** — {added} chunks added, {skipped} skipped")
        except Exception as e:
            msgs.append(f"❌ **{f.name}**: {e}")
            logger.error(f"Ingest error: {e}")
        msgs_placeholder.markdown("\n\n".join(msgs))
    progress.progress(1.0, text="Done!")
    time.sleep(0.6)
    progress.empty()
    if total_added > 0:
        st.success(f"🎉 {total_added} new chunks added!")
        time.sleep(0.8)
        st.rerun()


def _ingest_url(url: str):
    url = clean_url(url)
    if not is_valid_url(url):
        st.error("⚠️ Invalid URL. Make sure it starts with https://")
        return
    vs = get_vector_store()
    with st.spinner(f"🌐 Fetching {url}…"):
        try:
            docs = load_url(url)
            added, skipped = vs.add_documents(docs)
            st.success(f"✅ URL ingested — {added} chunks added, {skipped} skipped")
            time.sleep(0.8)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed: {e}")
            logger.error(f"URL error: {e}")


def render_chat():
    vs = get_vector_store()
    if vs.is_empty():
        st.info("📭 **Knowledge base is empty.** Use the panel above to upload documents.", icon="☝️")

    for msg in st.session_state.messages:
        role, content, sources = msg["role"], msg["content"], msg.get("sources", [])
        if role == "user":
            st.markdown(f'<div class="chat-bubble user"><div class="chat-label user">👤 You</div>{content}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="chat-bubble assistant"><div class="chat-label assistant">🤖 Research AI</div></div>',
                        unsafe_allow_html=True)
            st.markdown(content)
            if sources:
                pills = " ".join(
                    f'<span class="source-pill">{source_icon(s["type"])} {truncate(s["source"],40)}</span>'
                    for s in sources)
                st.markdown(f'<div style="margin-top:0.4rem">{pills}</div>', unsafe_allow_html=True)

    if st.session_state.followup_questions:
        st.markdown("**💡 Suggested follow-ups:**")
        for fq in st.session_state.followup_questions:
            if st.button(f"↩ {fq}", key=f"fq_{fq[:30]}", use_container_width=True):
                _handle_query(fq); st.rerun()

    st.markdown("---")
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input("Q", placeholder="e.g. What are the main findings?",
                              label_visibility="collapsed", key="chat_input")
    with col_btn:
        send = st.button("Send 🚀", use_container_width=True)
    if send and query.strip():
        _handle_query(query.strip()); st.rerun()


def _handle_query(query: str):
    engine = _get_engine()
    st.session_state.messages.append({"role": "user", "content": query})
    chunks = []
    with st.spinner("🧠 Thinking…"):
        for chunk in engine.stream_answer(query, source_filter=st.session_state.selected_source_filter):
            chunks.append(chunk)
    full = "".join(chunks)
    st.session_state.messages.append({"role":"assistant","content":full,"sources":engine.last_sources})
    st.session_state.last_sources = engine.last_sources
    try:
        st.session_state.followup_questions = (
            engine.get_follow_up_questions(query, full) if len(full) < 3000 else []
        )
    except Exception:
        st.session_state.followup_questions = []


def render_library():
    st.markdown('<h2 style="font-family:\'DM Serif Display\',serif;color:#E8E8F0;">📚 Knowledge Library</h2>', unsafe_allow_html=True)
    vs = get_vector_store()
    stats = vs.get_stats()
    if stats["total_sources"] == 0:
        st.info("No documents yet. Use the panel above to add documents."); return
    c1,c2,c3 = st.columns(3)
    for col, num, label in [(c1,stats["total_sources"],"Sources"),
                             (c2,stats["total_chunks"],"Total Chunks"),
                             (c3,stats["total_chunks"]//max(stats["total_sources"],1),"Avg Chunks/Src")]:
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    for src in stats["sources"]:
        with st.expander(f"{source_icon(src['type'])}  {src['source']}  ·  {src['chunks']} chunks"):
            ci, ca = st.columns([3,1])
            with ci:
                st.markdown(f"- **Type:** `{src['type']}`\n- **Chunks:** {src['chunks']}\n- **Ingested:** {src.get('ingested_at','')[:19]}")
            with ca:
                if st.button("🗑️ Delete", key=f"del_{src['source']}"):
                    vs.delete_source(src["source"]); st.success("Deleted!"); st.rerun()


def render_summarise():
    st.markdown('<h2 style="font-family:\'DM Serif Display\',serif;color:#E8E8F0;">📝 Document Summariser</h2>', unsafe_allow_html=True)
    vs = get_vector_store()
    sources = vs.list_sources()
    if not sources:
        st.info("No documents yet."); return
    chosen = st.selectbox("Select document:", sources)
    if st.button("🧠 Generate Summary"):
        st.markdown(f"### Summary: `{chosen}`\n---")
        ph = st.empty(); full = ""
        with st.spinner("Generating…"):
            for chunk in summarise_source(chosen):
                full += chunk; ph.markdown(full + "▌")
        ph.markdown(full)
        st.download_button("⬇️ Download", full, f"summary_{Path(chosen).stem}.md", "text/markdown")


def main():
    _init_state()
    issues = settings.validate_keys()
    if issues:
        st.error("⚠️ **Config issues:**\n\n" + "\n".join(f"- {i}" for i in issues))
        st.info("1. Copy `.env.example` → `.env`\n2. Add your API key\n3. Restart")
        st.stop()
    render_navbar()
    render_upload_panel()
    tabs = st.tabs(["💬 Chat", "📚 Library", "📝 Summarise"])
    with tabs[0]: render_chat()
    with tabs[1]: render_library()
    with tabs[2]: render_summarise()


if __name__ == "__main__":
    main()