import os
import uuid
from pathlib import Path
import streamlit as st

from core.config import PROJECTS_DIR, OPENAI_API_KEY
from core.projects import list_projects, create_project, get_project
from core.pipeline import (
    run_ingestion_pipeline,
    load_document_pipeline_data,
    get_project_docs_dir,
    delete_source_document,
)
from core.rag_chain import generate_rag_response
from core.chat_manager import (
    list_chats,
    get_chat,
    save_chat,
    generate_chat_title,
    rename_chat,
    delete_chat,
)

# ----------------- Streamlit Page Configuration -----------------
st.set_page_config(
    page_title="Retrio - Multi-Modal RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------- Sleek Dark Theme Custom CSS -----------------
st.markdown("""
<style>
    /* 1. Eliminate Streamlit top empty gap */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 100% !important;
    }
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0px !important;
    }
    #MainMenu, footer {
        visibility: hidden !important;
        height: 0px !important;
    }
    
    /* 2. Global Dark Background */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }

    /* 3. Retrio Logo */
    .retrio-logo {
        font-size: 26px;
        font-weight: 800;
        background: linear-gradient(135deg, #a78bfa 0%, #818cf8 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
        display: inline-block;
    }

    /* 4. Column constraints to prevent whole-page scrolling */
    [data-testid="column"]:nth-child(1),
    [data-testid="column"]:nth-child(3) {
        max-height: calc(100vh - 2.5rem);
        overflow-y: auto;
        padding-right: 4px;
    }

    /* 5. Custom Card Styling */
    .slate-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 14px;
        margin-bottom: 8px;
        transition: all 0.2s ease;
    }
    .slate-card:hover {
        border-color: #58a6ff;
        background-color: #1c2128;
    }

    /* Success green banner */
    .success-banner {
        background-color: #0d2818;
        border: 1px solid #238636;
        border-radius: 6px;
        padding: 10px 16px;
        color: #3fb950;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 16px;
    }
    
    /* Classification Badges */
    .badge-text {
        background-color: #1f3a2c;
        color: #3fb950;
        border: 1px solid #238636;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-table {
        background-color: #3b234a;
        color: #d2a8ff;
        border: 1px solid #8957e5;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }
    .badge-image {
        background-color: #1f304a;
        color: #79c0ff;
        border: 1px solid #388bfd;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
    }

    /* Metric container for Pipeline tabs */
    .metric-box {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- Session State Initialization -----------------
all_projects = list_projects()
default_pid = all_projects[0]["id"] if all_projects else None

if "active_project_id" not in st.session_state or not st.session_state.active_project_id:
    st.session_state.active_project_id = default_pid

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = str(uuid.uuid4())

if "current_chat_title" not in st.session_state:
    st.session_state.current_chat_title = "New Chat"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "inspecting_document" not in st.session_state:
    st.session_state.inspecting_document = None

if "selected_chunk_id" not in st.session_state:
    st.session_state.selected_chunk_id = None


# ====================================================================
# POPUP DIALOGS (Centered Modals)
# ====================================================================
@st.dialog("📂 Choose Project")
def choose_project_dialog():
    st.caption("Select an existing knowledge base workspace")
    projects = list_projects()
    if not projects:
        st.info("No projects found. Create one first!")
        if st.button("Close"):
            st.rerun()
        return

    search = st.text_input("🔍 Search projects...", placeholder="Filter by name...")
    filtered = [p for p in projects if search.lower() in p.get("name", "").lower()] if search else projects

    for p in filtered:
        is_current = (p.get("id") == st.session_state.active_project_id)
        col_info, col_act = st.columns([7, 3])
        with col_info:
            st.markdown(f"**{p.get('name')}**")
            st.caption(f"{p.get('created_at')} • {p.get('description', '')[:50]}")
        with col_act:
            if is_current:
                st.button("Active", disabled=True, key=f"active_{p.get('id')}", use_container_width=True)
            else:
                if st.button("Select", key=f"sel_{p.get('id')}", use_container_width=True):
                    st.session_state.active_project_id = p.get("id")
                    chats = list_chats(p.get("id"))
                    if chats:
                        first_chat = chats[0]
                        st.session_state.active_chat_id = first_chat["chat_id"]
                        st.session_state.current_chat_title = first_chat["title"]
                        st.session_state.messages = first_chat.get("messages", [])
                    else:
                        st.session_state.active_chat_id = str(uuid.uuid4())
                        st.session_state.current_chat_title = "New Chat"
                        st.session_state.messages = []
                    st.session_state.inspecting_document = None
                    st.rerun()


@st.dialog("✨ Create New Project")
def new_project_dialog():
    st.caption("Start a new knowledge base project")
    p_name = st.text_input("Project Name *", placeholder="e.g., Research Analysis, GenAI Team...")
    p_desc = st.text_area("Description (optional)", placeholder="What will this project be about?")

    col_create, col_cancel = st.columns([1, 1])
    with col_create:
        if st.button("Create Project", type="primary", use_container_width=True):
            if p_name.strip():
                new_proj = create_project(p_name.strip(), p_desc.strip())
                st.session_state.active_project_id = new_proj["id"]
                st.session_state.active_chat_id = str(uuid.uuid4())
                st.session_state.current_chat_title = "New Chat"
                st.session_state.messages = []
                st.session_state.inspecting_document = None
                st.rerun()
            else:
                st.warning("Please provide a project name.")
    with col_cancel:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


# ====================================================================
# SCREEN 2: PIPELINE INSPECTOR (KEPT 100% UNTOUCHED)
# ====================================================================
def render_pipeline_inspector(project_id: str, filename: str):
    # Lock outer window so tabs and header stay fixed on screen
    st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
            height: 100vh !important;
        }
    </style>
    """, unsafe_allow_html=True)

    col_nav1, col_nav2 = st.columns([8, 2])
    with col_nav1:
        st.markdown(f"### 📄 `{filename}`")
        st.caption("Document Processing Pipeline Telemetry")
    with col_nav2:
        if st.button("✕ Close", use_container_width=True, type="primary"):
            st.session_state.inspecting_document = None
            st.rerun()

    pipeline_data = load_document_pipeline_data(project_id, filename)
    if not pipeline_data:
        st.warning(f"No pipeline data found for {filename}. Please re-process the document.")
        return

    stages = pipeline_data.get("stages", {})

    tab_partition, tab_chunk, tab_summary, tab_vector, tab_view = st.tabs([
        "Partitioning",
        "Chunking",
        "Summarisation",
        "Vectorization & Storage",
        "View Chunks"
    ])

    # 1. Partitioning Tab
    with tab_partition:
        st.markdown("#### Partitioning")
        st.caption("Processing and extracting text, images, and tables layout")
        
        part_stats = stages.get("partitioning", {}).get("stats", {})
        st.markdown("##### 📊 Elements Discovered")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{part_stats.get('text_sections', 0)}</div><div class='metric-label'>Text sections</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{part_stats.get('tables', 0)}</div><div class='metric-label'>Tables</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{part_stats.get('images', 0)}</div><div class='metric-label'>Images</div></div>", unsafe_allow_html=True)
        with c4:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{part_stats.get('titles_headers', 0)}</div><div class='metric-label'>Titles/Headers</div></div>", unsafe_allow_html=True)
        with c5:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{part_stats.get('other_elements', 0)}</div><div class='metric-label'>Other elements</div></div>", unsafe_allow_html=True)

        st.markdown("<div class='success-banner'>✔ Step completed successfully</div>", unsafe_allow_html=True)

    # 2. Chunking Tab
    with tab_chunk:
        st.markdown("#### Chunking")
        st.caption("Creating semantic chunks using section boundaries and layout")

        chk_stats = stages.get("chunking", {}).get("stats", {})
        atomic_cnt = chk_stats.get("atomic_elements_count", 0)
        chunks_cnt = chk_stats.get("chunks_created", 0)
        avg_sz = chk_stats.get("avg_chunk_size", 0)

        st.markdown("##### 🧩 Chunking Results")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value' style='color:#3fb950;'>{atomic_cnt} &nbsp;➔&nbsp; {chunks_cnt}</div>
                <div class='metric-label'>atomic elements to chunks created</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-box'>
                <div class='metric-value'>{avg_sz}</div>
                <div class='metric-label'>Average chunk size (characters)</div>
            </div>
            """, unsafe_allow_html=True)

        st.caption(f"{atomic_cnt} atomic elements have been chunked by title to produce {chunks_cnt} chunks.")
        st.markdown("<div class='success-banner'>✔ Step completed successfully</div>", unsafe_allow_html=True)

    # 3. Summarisation Tab
    with tab_summary:
        st.markdown("#### Summarisation")
        st.caption("Enhancing content with AI summaries for images and tables")

        sum_stats = stages.get("summarisation", {}).get("stats", {})
        tot_chunks = sum_stats.get("total_chunks", 0)
        proc_chunks = sum_stats.get("processed_chunks", 0)
        tbl_sum = sum_stats.get("tables_summarized", 0)
        img_sum = sum_stats.get("images_summarized", 0)

        st.markdown("##### 🤖 AI Summarising Progress")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{proc_chunks} / {tot_chunks}</div><div class='metric-label'>Chunks processed</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{tbl_sum}</div><div class='metric-label'>Tables summarized</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{img_sum}</div><div class='metric-label'>Images summarized</div></div>", unsafe_allow_html=True)

        st.caption(f"Model used: `{sum_stats.get('llm_used', 'gpt-4o-mini')}`")
        st.markdown("<div class='success-banner'>✔ Step completed successfully</div>", unsafe_allow_html=True)

    # 4. Vectorization & Storage Tab
    with tab_vector:
        st.markdown("#### Vectorization & Storage")
        st.caption("Generating OpenAI embeddings and indexing vectors into ChromaDB")

        vec_stats = stages.get("vectorization", {}).get("stats", {})
        
        # Query live vector count directly from ChromaDB for real-time project accuracy
        from core.vectorstore import get_project_vectorstore
        try:
            live_total_vectors = get_project_vectorstore(project_id)._collection.count()
        except Exception:
            live_total_vectors = vec_stats.get("total_vectors_stored", 0)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{vec_stats.get('new_vectors_added', 0)}</div><div class='metric-label'>Vectors from this document</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{live_total_vectors}</div><div class='metric-label'>Total collection vector count (live)</div></div>", unsafe_allow_html=True)

        st.markdown(f"**Embedding Model**: `{vec_stats.get('embedding_model', 'text-embedding-3-small')}`")
        st.markdown(f"**Chroma Collection**: `{vec_stats.get('collection_name', 'default')}`")
        st.markdown(f"**Persistent Directory**: `{vec_stats.get('storage_path', '')}`")

        st.markdown("<div class='success-banner'>✔ Step completed successfully</div>", unsafe_allow_html=True)

    # 5. View Chunks Tab + Detail Inspector
    with tab_view:
        chunks = stages.get("view_chunks", {}).get("chunks", [])
        
        col_list, col_inspector = st.columns([1, 1], gap="medium")
        
        with col_list:
            st.markdown(f"#### Content Chunks ({len(chunks)} of {len(chunks)} chunks)")
            
            filter_type = st.radio(
                "Filter by type:",
                ["All", "Text", "Image", "Table"],
                horizontal=True,
                label_visibility="collapsed"
            )
            
            search_query = st.text_input("🔍 Search chunks...", placeholder="Search chunk text...")
            
            filtered_chunks = chunks
            if filter_type != "All":
                filtered_chunks = [c for c in filtered_chunks if c.get("type", "").lower() == filter_type.lower()]
            if search_query.strip():
                q = search_query.lower()
                filtered_chunks = [c for c in filtered_chunks if q in c.get("content", "").lower() or q in (c.get("summary") or "").lower()]

            # Scrollable container for the chunk list
            chunk_list_container = st.container(height=480)
            with chunk_list_container:
                if not filtered_chunks:
                    st.info("No chunks match current filter.")
                else:
                    for idx, c in enumerate(filtered_chunks):
                        cid = c.get("chunk_id", f"c_{idx}")
                        ctype = c.get("type", "text").upper()
                        badge_class = "badge-table" if ctype == "TABLE" else ("badge-image" if ctype == "IMAGE" else "badge-text")
                        
                        is_selected = (st.session_state.selected_chunk_id == cid)
                        border_color = "#58a6ff" if is_selected else "#30363d"
                        
                        with st.container():
                            st.markdown(f"""
                            <div style='background-color:#161b22; border:1px solid {border_color}; border-radius:6px; padding:12px; margin-bottom:8px;'>
                                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                                    <span class='{badge_class}'>{ctype}</span>
                                    <span style='font-size:12px; color:#8b949e;'>Page {c.get('page_number', 1)} • {c.get('char_count', len(c.get('content', '')))} chars</span>
                                </div>
                                <div style='font-size:13px; color:#c9d1d9; line-height:1.4;'>{c.get('content', '')[:120]}...</div>
                            </div>
                            """, unsafe_allow_html=True)
                            if st.button(f"Inspect Chunk {cid}", key=f"btn_chk_{cid}", use_container_width=True):
                                st.session_state.selected_chunk_id = cid
                                st.rerun()

        with col_inspector:
            st.markdown("#### Detail Inspector")
            inspector_container = st.container(height=560)
            with inspector_container:
                selected_chunk = None
                if st.session_state.selected_chunk_id:
                    for c in chunks:
                        if c.get("chunk_id") == st.session_state.selected_chunk_id:
                            selected_chunk = c
                            break
                if not selected_chunk and chunks:
                    selected_chunk = chunks[0]

                if selected_chunk:
                    ctype = selected_chunk.get("type", "text").upper()
                    badge_class = "badge-table" if ctype == "TABLE" else ("badge-image" if ctype == "IMAGE" else "badge-text")
                    
                    st.markdown(f"<span class='{badge_class}'>{ctype}</span> &nbsp; **Chunk ID**: `{selected_chunk.get('chunk_id')}`", unsafe_allow_html=True)
                    st.caption(f"Source: `{selected_chunk.get('filename')}` | Page: `{selected_chunk.get('page_number')}` | Size: `{selected_chunk.get('char_count')} chars`")

                    if selected_chunk.get("summary"):
                        st.markdown("##### 💡 AI Summary")
                        st.info(selected_chunk["summary"])

                    st.markdown("##### 📄 Content")
                    st.code(selected_chunk.get("content", ""), language="markdown")

                    if selected_chunk.get("metadata", {}).get("text_as_html"):
                        with st.expander("View Rendered HTML Table"):
                            st.markdown(selected_chunk["metadata"]["text_as_html"], unsafe_allow_html=True)
                else:
                    st.info("Select a chunk on the left to inspect its details.")
                st.info("Select a chunk on the left to inspect its details.")


# ====================================================================
# SCREEN 1: 3-COLUMN HOME PAGE (RETRIO)
# ====================================================================
def render_retrio_home(project_id: str):
    project = get_project(project_id)
    if not project:
        all_projs = list_projects()
        if all_projs:
            st.session_state.active_project_id = all_projs[0]["id"]
            project = all_projs[0]
        else:
            st.info("No projects available. Please create a new project.")
            if st.button("➕ Create Project", type="primary"):
                new_project_dialog()
            return

    # If the user clicked to inspect a document's pipeline, render Screen 2 (Pipeline Page)
    if st.session_state.inspecting_document:
        render_pipeline_inspector(project_id, st.session_state.inspecting_document)
        return

    # Lock root browser viewport on home screen so only the inner chat messages scroll
    st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            overflow: hidden !important;
            height: 100vh !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 3-Column Layout: Left (Sidebar & Chats), Center (Active Chat), Right (Knowledge Base)
    col_left, col_mid, col_right = st.columns([2.8, 5.4, 3.2], gap="medium")

    # ==========================================
    # LEFT PANEL (Blue Box in drawing)
    # ==========================================
    with col_left:
        st.markdown("<div class='retrio-logo'>⚡ Retrio</div>", unsafe_allow_html=True)

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("📂 Choose", use_container_width=True, help="Choose an existing project"):
                choose_project_dialog()
        with col_btn2:
            if st.button("➕ New", use_container_width=True, help="Create a new project"):
                new_project_dialog()

        st.markdown("---")

        # Current Project Indicator
        st.markdown("<div style='font-size:11px; color:#8b949e; text-transform:uppercase; font-weight:700;'>Current Project</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:16px; font-weight:700; color:#58a6ff; margin-bottom:12px;'>📁 {project.get('name')}</div>", unsafe_allow_html=True)

        # Saved Chats for this project with 3-dots Menu (Rename / Delete)
        st.markdown("<div style='font-size:11px; color:#8b949e; text-transform:uppercase; font-weight:700; margin-bottom:8px;'>Conversations</div>", unsafe_allow_html=True)
        
        project_chats = list_chats(project_id)
        if not project_chats:
            st.caption("No saved chats yet. Start asking questions in the center to create one!")
        else:
            for ch in project_chats:
                cid = ch.get("chat_id")
                ctitle = ch.get("title", "Untitled Chat")
                is_active = (cid == st.session_state.active_chat_id)
                btn_type = "primary" if is_active else "secondary"
                
                col_cbtn, col_cmenu = st.columns([7.8, 2.2])
                with col_cbtn:
                    if st.button(f"💬 {ctitle}", key=f"chat_{cid}", type=btn_type, use_container_width=True):
                        st.session_state.active_chat_id = cid
                        st.session_state.current_chat_title = ctitle
                        st.session_state.messages = ch.get("messages", [])
                        st.session_state.inspecting_document = None
                        st.rerun()
                with col_cmenu:
                    with st.popover("⋮"):
                        st.markdown("**Conversation**")
                        new_title_val = st.text_input("Rename", value=ctitle, key=f"ren_txt_{cid}")
                        if st.button("Save Name", key=f"btn_save_ren_{cid}", use_container_width=True):
                            if new_title_val.strip():
                                rename_chat(project_id, cid, new_title_val.strip())
                                if cid == st.session_state.active_chat_id:
                                    st.session_state.current_chat_title = new_title_val.strip()
                                st.rerun()
                        if st.button("🗑️ Delete", key=f"btn_del_chat_{cid}", type="primary", use_container_width=True):
                            delete_chat(project_id, cid)
                            if cid == st.session_state.active_chat_id:
                                st.session_state.active_chat_id = str(uuid.uuid4())
                                st.session_state.current_chat_title = "New Chat"
                                st.session_state.messages = []
                            st.rerun()

    # ==========================================
    # MIDDLE PANEL: ACTIVE CHAT (Red Box in drawing)
    # ==========================================
    with col_mid:
        # Top Header: ALWAYS displays Project Name + New Chat Button
        col_title, col_new_chat = st.columns([7, 3])
        with col_title:
            st.markdown(f"<div style='font-size:20px; font-weight:800; color:#e6edf3; margin-bottom: 2px;'>📁 {project.get('name')}</div>", unsafe_allow_html=True)
            if project.get("description"):
                st.caption(project.get("description"))
        with col_new_chat:
            if st.button("➕ New Chat", use_container_width=True, type="secondary"):
                if st.session_state.messages:
                    auto_title = generate_chat_title(st.session_state.messages)
                    save_chat(project_id, st.session_state.active_chat_id, auto_title, st.session_state.messages)
                
                st.session_state.active_chat_id = str(uuid.uuid4())
                st.session_state.current_chat_title = "New Chat"
                st.session_state.messages = []
                st.rerun()

        # Fixed height scrollable chat container sized to fit the screen without outer page scrolling
        chat_container = st.container(height=495)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style='text-align: center; padding: 40px 20px; color: #8b949e;'>
                    <div style='font-size: 36px; margin-bottom: 8px;'>⚡</div>
                    <div style='font-size: 16px; font-weight: 600; color: #c9d1d9;'>What can Retrio help you analyze today?</div>
                    <div style='font-size: 13px; margin-top: 4px;'>Ask questions grounded in the knowledge base documents on the right.</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            with st.expander("📚 Retrieved Sources & Citations (BM25 + Dense RRF)"):
                                for i, src in enumerate(msg["sources"]):
                                    st.markdown(f"**[{i+1}] {src.get('filename')} (Page {src.get('page_number')}) - `{src.get('type', 'text').upper()}`**")
                                    st.caption(f"Score: `{round(src.get('score', 0.0), 3)}` | Method: `{src.get('retrieval_method', 'hybrid')}`")
                                    st.text(src.get("content", "")[:250] + "...")

        # Chat Input Box pinned at the bottom
        if user_prompt := st.chat_input("Ask Retrio..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})

            if st.session_state.current_chat_title == "New Chat":
                st.session_state.current_chat_title = generate_chat_title(st.session_state.messages)

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Retrio is searching documents & reasoning..."):
                        response = generate_rag_response(
                            project_id=project_id,
                            query=user_prompt,
                            chat_history=st.session_state.messages[:-1],
                        )
                        answer_text = response.get("answer", "")
                        sources = response.get("sources", [])

                        st.markdown(answer_text)
                        if sources:
                            with st.expander("📚 Retrieved Sources & Citations (BM25 + Dense RRF)"):
                                for i, src in enumerate(sources):
                                    st.markdown(f"**[{i+1}] {src.get('filename')} (Page {src.get('page_number')}) - `{src.get('type', 'text').upper()}`**")
                                    st.caption(f"Score: `{round(src.get('score', 0.0), 3)}` | Method: `{src.get('retrieval_method', 'hybrid')}`")
                                    st.text(src.get("content", "")[:250] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
            })

            save_chat(
                project_id=project_id,
                chat_id=st.session_state.active_chat_id,
                title=st.session_state.current_chat_title,
                messages=st.session_state.messages,
            )
            st.rerun()

    # ==========================================
    # RIGHT PANEL: KNOWLEDGE BASE (Green/Pink Box in drawing)
    # ==========================================
    with col_right:
        st.markdown("<div style='font-size:18px; font-weight:700;'>📚 Knowledge base</div>", unsafe_allow_html=True)
        st.caption("Documents uploaded here belong to this project.")

        # Section: Add Sources
        st.markdown("##### Add sources")
        uploaded_file = st.file_uploader(
            "Drop files or click to upload",
            type=["pdf", "docx", "csv", "txt", "md"],
            label_visibility="collapsed",
            help="PDF, DOCX, CSV, TXT (Max 50MB)"
        )

        if uploaded_file is not None:
            save_dir = get_project_docs_dir(project_id)
            saved_file_path = save_dir / uploaded_file.name

            if not saved_file_path.exists():
                with open(saved_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                progress_bar = st.progress(0, text="Initializing ingestion pipeline...")
                def on_step_progress(stage_name, step, total):
                    pct = int((step / total) * 100)
                    progress_bar.progress(pct, text=f"Executing {stage_name} ({step}/{total})...")

                with st.spinner(f"Ingesting `{uploaded_file.name}`..."):
                    run_ingestion_pipeline(project_id, saved_file_path, progress_callback=on_step_progress)
                    progress_bar.progress(100, text="Ingestion completed successfully!")
                    st.success(f"`{uploaded_file.name}` indexed into ChromaDB!")
                    st.rerun()

        st.markdown("---")

        # Section: Sources List (Click source to open pipeline directly + 3-dots delete menu)
        st.markdown("##### Sources")
        docs_dir = get_project_docs_dir(project_id)
        source_files = [f for f in docs_dir.iterdir() if f.is_file() and not f.name.endswith(".pipeline.json")]

        if not source_files:
            st.info("No sources added yet. Drop a file above to index it!")
        else:
            for sfile in source_files:
                size_mb = round(sfile.stat().st_size / (1024 * 1024), 2)
                size_str = f"{size_mb} MB" if size_mb >= 0.01 else f"{round(sfile.stat().st_size / 1024, 1)} KB"

                col_sbtn, col_smenu = st.columns([8.2, 1.8])
                with col_sbtn:
                    # Clicking the document card directly opens the Pipeline Page
                    doc_label = f"📄 {sfile.name}  ({size_str})"
                    if st.button(doc_label, key=f"src_card_{sfile.name}", use_container_width=True, help="Click to view processing pipeline"):
                        st.session_state.inspecting_document = sfile.name
                        st.rerun()
                with col_smenu:
                    with st.popover("⋮"):
                        st.markdown("**Source Options**")
                        st.caption(f"`{sfile.name}`")
                        if st.button("⚡ Inspect Pipeline", key=f"pop_pipe_{sfile.name}", use_container_width=True):
                            st.session_state.inspecting_document = sfile.name
                            st.rerun()
                        if st.button("🗑️ Delete Source", key=f"del_src_{sfile.name}", type="primary", use_container_width=True):
                            delete_source_document(project_id, sfile.name)
                            st.success(f"Deleted `{sfile.name}`")
                            st.rerun()


# ====================================================================
# MAIN APPLICATION ROUTING
# ====================================================================
def main():
    if not st.session_state.active_project_id:
        all_projs = list_projects()
        if all_projs:
            st.session_state.active_project_id = all_projs[0]["id"]
        else:
            starter = create_project("Research", "Multi-modal RAG documentation & research")
            st.session_state.active_project_id = starter["id"]

    render_retrio_home(st.session_state.active_project_id)


if __name__ == "__main__":
    main()
