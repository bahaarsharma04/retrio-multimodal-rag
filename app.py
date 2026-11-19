import re
import uuid
import streamlit as st

from core.projects import list_projects, create_project, get_project, delete_project
from core.pipeline import (
    run_ingestion_pipeline,
    load_document_pipeline_data,
    get_project_docs_dir,
    get_document_record_path,
    get_interrupted_documents,
    delete_source_document,
    reconcile_project_documents,
)
from core.rag_chain import generate_rag_response
from core.chat_manager import (
    list_chats,
    save_chat,
    generate_chat_title,
    rename_chat,
    delete_chat,
)
from core.users import authenticate_user, create_user
from core.security import create_access_token

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
    /* 1. Eliminate Streamlit top empty gap and lock root viewport */
    html, body, .stApp, [data-testid="stAppViewContainer"], section.main, .main, [data-testid="stMain"] {
        overflow: hidden !important;
        height: 100vh !important;
        max-height: 100vh !important;
    }
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

    /* 3. Eliminate Streamlit Stale-Element Dimming / Screen Darkening on Reruns */
    [data-stale="true"],
    [data-stale="true"] *,
    div[data-stale="true"],
    div[data-stale="true"] > *,
    [data-testid="column"][data-stale="true"],
    [data-testid="column"] [data-stale="true"],
    [data-testid="stVerticalBlock"] [data-stale="true"],
    [data-testid="stVerticalBlock"] > div[data-stale="true"],
    .stElementContainer[data-stale="true"],
    [data-testid="stAppViewContainer"] [data-stale="true"],
    [data-testid="stAppViewBlockContainer"] [data-stale="true"],
    [data-testid="stChatMessageContainer"][data-stale="true"] {
        opacity: 1 !important;
        filter: none !important;
        transition: none !important;
    }

    /* 4. Retrio Logo */
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

    /* 4. Streamlit native container scrollbars styling */
    [data-testid="stVerticalBlockBorderWrapper"] ::-webkit-scrollbar,
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #58a6ff;
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
if "current_user" not in st.session_state:
    st.session_state.current_user = None

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

if "active_project_id" not in st.session_state:
    st.session_state.active_project_id = None

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
    user_id = st.session_state.current_user["id"] if st.session_state.get("current_user") else None

    projects = list_projects(owner_id=user_id)
    if not projects:
        st.info("No projects found.")
        if st.button("Close"):
            st.rerun()
        return

    search = st.text_input("🔍 Search projects...", placeholder="Filter by name...")
    if search:
        s_low = search.lower()
        filtered = [
            p for p in projects
            if s_low in p.get("name", "").lower()
            or s_low in (p.get("description") or "").lower()
        ]
    else:
        filtered = projects

    for p in filtered:
        p_id = p.get("id")
        is_current = (p_id == st.session_state.active_project_id)
        col_info, col_act, col_del = st.columns([6, 2.5, 1.5])
        with col_info:
            st.markdown(f"**{p.get('name')}**")
            st.caption(f"{p.get('created_at')} • {p.get('description', '')[:35]}")
        with col_act:
            if is_current:
                st.button("Active", disabled=True, key=f"active_{p_id}", use_container_width=True)
            else:
                if st.button("Select", key=f"sel_{p_id}", use_container_width=True):
                    st.session_state.active_project_id = p_id
                    chats = list_chats(p_id, user_id=user_id)
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
        with col_del:
            if st.button("🗑️", key=f"del_{p_id}", help=f"Delete project '{p.get('name')}'", use_container_width=True):
                delete_project(p_id, owner_id=user_id)
                remaining = [proj for proj in list_projects(owner_id=user_id) if proj.get("id") != p_id]
                if st.session_state.active_project_id == p_id:
                    if remaining:
                        new_active = remaining[0]["id"]
                        st.session_state.active_project_id = new_active
                        chats = list_chats(new_active, user_id=user_id)
                        if chats:
                            st.session_state.active_chat_id = chats[0]["chat_id"]
                            st.session_state.current_chat_title = chats[0]["title"]
                            st.session_state.messages = chats[0].get("messages", [])
                        else:
                            st.session_state.active_chat_id = str(uuid.uuid4())
                            st.session_state.current_chat_title = "New Chat"
                            st.session_state.messages = []
                    else:
                        st.session_state.active_project_id = None
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
    user_id = st.session_state.current_user["id"] if st.session_state.get("current_user") else None

    col_create, col_cancel = st.columns([1, 1])
    with col_create:
        if st.button("Create Project", type="primary", use_container_width=True):
            if p_name.strip():
                new_proj = create_project(p_name.strip(), p_desc.strip(), owner_id=user_id)
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
# SCREEN 2: PIPELINE INSPECTOR 
# ====================================================================
def render_pipeline_inspector(project_id: str, filename: str):
    # Lock outer window so tabs and header stay fixed on screen
    st.markdown("""  
    <style>
        html, body, .stApp, [data-testid="stAppViewContainer"], section.main, .main, [data-testid="stMain"] {
            overflow: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }
    </style>
    """, unsafe_allow_html=True)

    col_nav1, col_nav2, col_nav3 = st.columns([7, 2.5, 1.5])
    with col_nav1:
        st.markdown(f"### 📄 `{filename}`")
        st.caption("Document Processing Pipeline Telemetry")
    with col_nav2:
        if st.button("🔄 Re-process", key=f"reingest_top_{filename}", use_container_width=True, help="Force re-run pipeline from scratch"):
            doc_fpath = get_project_docs_dir(project_id) / filename
            if doc_fpath.exists():
                with st.spinner(f"Re-processing `{filename}` with latest models & vision..."):
                    run_ingestion_pipeline(project_id, doc_fpath, restart=True)
                    st.success("Re-processing complete!")
                    st.rerun()
    with col_nav3:
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

        total_atomic = part_stats.get('total_elements', 0)
        st.caption(f"💡 Total atomic elements discovered: **{total_atomic}** ({part_stats.get('text_sections', 0)} text + {part_stats.get('titles_headers', 0)} titles/headers + {part_stats.get('tables', 0)} tables + {part_stats.get('images', 0)} images).")

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
        st.caption("Generating OpenAI embeddings and indexing vectors into PostgreSQL (pgvector)")

        vec_stats = stages.get("vectorization", {}).get("stats", {})
        
        # Query live vector count directly from PostgreSQL for real-time project accuracy
        from core.vectorstore import get_live_vector_count
        try:
            live_total_vectors = get_live_vector_count(project_id) or vec_stats.get("total_vectors_stored", 0)
        except Exception:
            live_total_vectors = vec_stats.get("total_vectors_stored", 0)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{vec_stats.get('new_vectors_added', 0)}</div><div class='metric-label'>Vectors from this document</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='metric-box'><div class='metric-value'>{live_total_vectors}</div><div class='metric-label'>Total collection vector count (live)</div></div>", unsafe_allow_html=True)

        db_engine_name = vec_stats.get("database_engine", "PostgreSQL (pgvector)")
        st.markdown(f"**Embedding Model**: `{vec_stats.get('embedding_model', 'text-embedding-3-small')}`")
        st.markdown(f"**Database Engine**: `{db_engine_name}`")
        st.markdown(f"**Storage Target**: `{vec_stats.get('storage_path', 'PostgreSQL database (retrio_db.document_chunks)')}`")

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

                    # If an extracted figure image file exists on disk, display it
                    img_path = selected_chunk.get("image_path") or selected_chunk.get("metadata", {}).get("image_path")
                    if img_path and os.path.isfile(img_path):
                        st.markdown("##### 🖼️ Extracted Visual Figure")
                        st.image(img_path, caption=selected_chunk.get("caption") or "Extracted Document Figure", use_container_width=True)

                    st.markdown("##### 📄 Content")
                    st.code(selected_chunk.get("content", ""), language="markdown")

                    if selected_chunk.get("metadata", {}).get("text_as_html"):
                        with st.expander("View Rendered HTML Table"):
                            raw_html = selected_chunk["metadata"]["text_as_html"]
                            # XSS sanitize: strip scripts and event handlers
                            clean_html = re.sub(r"<(script|iframe|object|embed)[^>]*>.*?</\1>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
                            clean_html = re.sub(r"on\w+\s*=\s*['\"][^'\"]*['\"]", "", clean_html, flags=re.IGNORECASE)
                            clean_html = re.sub(r"javascript:\s*", "", clean_html, flags=re.IGNORECASE)
                            st.markdown(clean_html, unsafe_allow_html=True)
                else:
                    st.info("Select a chunk on the left to inspect its details.")


# ====================================================================
# SCREEN 1: 3-COLUMN HOME PAGE (RETRIO)
# ====================================================================
def render_retrio_home(project_id: str):
    if project_id:
        try:
            reconcile_project_documents(project_id)
        except Exception:
            pass
    user_id = st.session_state.current_user["id"] if st.session_state.get("current_user") else None
    project = get_project(project_id, owner_id=user_id) if project_id else None
    if not project:
        all_projs = list_projects(owner_id=user_id)
        if all_projs:
            st.session_state.active_project_id = all_projs[0]["id"]
            project = all_projs[0]
        else:
            col_l, col_m, col_r = st.columns([2.8, 5.4, 3.2], gap="medium")
            with col_l:
                st.markdown("<div class='retrio-logo'>⚡ Retrio</div>", unsafe_allow_html=True)
                if st.session_state.get("current_user"):
                    u_email = st.session_state.current_user.get("email", "")
                    st.caption(f"👤 Signed in as **{u_email}**")
                    if st.button("Sign Out", key="logout_empty"):
                        st.session_state.current_user = None
                        st.session_state.auth_token = None
                        st.session_state.active_project_id = None
                        st.session_state.messages = []
                        st.rerun()
            with col_m:
                st.info("No projects available in your workspace. Please create a project to get started!")
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
        html, body, .stApp, [data-testid="stAppViewContainer"], section.main, .main, [data-testid="stMain"] {
            overflow: hidden !important;
            height: 100vh !important;
            max-height: 100vh !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # 3-Column Layout: Left (Sidebar & Chats), Center (Active Chat), Right (Knowledge Base)
    col_left, col_mid, col_right = st.columns([2.8, 5.4, 3.2], gap="medium")

    # ==========================================
    # LEFT PANEL (Blue Box in drawing)
    # ==========================================
    with col_left:
        col_logo, col_logout = st.columns([6.8, 3.2])
        with col_logo:
            st.markdown("<div class='retrio-logo'>⚡ Retrio</div>", unsafe_allow_html=True)
            if st.session_state.get("current_user"):
                u_name = st.session_state.current_user.get("full_name") or st.session_state.current_user.get("email", "").split("@")[0]
                st.caption(f"👤 **{u_name}**")
        with col_logout:
            if st.button("Sign Out", key="btn_logout", help="Sign out of your session", use_container_width=True):
                st.session_state.current_user = None
                st.session_state.auth_token = None
                st.session_state.active_project_id = None
                st.session_state.messages = []
                st.rerun()

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
        st.markdown(f"<div style='font-size:16px; font-weight:700; color:#58a6ff; margin-bottom:4px;'>📁 {project.get('name')}</div>", unsafe_allow_html=True)

        # Saved Chats for this project with 3-dots Menu (Rename / Delete)
        st.markdown("<div style='font-size:11px; color:#8b949e; text-transform:uppercase; font-weight:700; margin-bottom:8px;'>Conversations</div>", unsafe_allow_html=True)
        
        project_chats = list_chats(project_id, user_id=user_id)
        chats_container = st.container(height=420)
        with chats_container:
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
                                    rename_chat(project_id, cid, new_title_val.strip(), user_id=user_id)
                                    if cid == st.session_state.active_chat_id:
                                        st.session_state.current_chat_title = new_title_val.strip()
                                    st.rerun()
                            if st.button("🗑️ Delete", key=f"btn_del_chat_{cid}", type="primary", use_container_width=True):
                                delete_chat(project_id, cid, user_id=user_id)
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
                    save_chat(project_id, st.session_state.active_chat_id, auto_title, st.session_state.messages, user_id=user_id)
                
                st.session_state.active_chat_id = str(uuid.uuid4())
                st.session_state.current_chat_title = "New Chat"
                st.session_state.messages = []
                st.rerun()

        # Fixed height scrollable chat container sized to fit the screen without outer page scrolling
        chat_container = st.container(height=450)
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
                                if msg.get("search_query"):
                                    st.caption(f"🔍 Reformulated Query: `{msg.get('search_query')}`")
                                if msg.get("target_document"):
                                    st.caption(f"🎯 Document Routing: `{msg.get('target_document')}`")
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
                                if response.get("search_query"):
                                    st.caption(f"🔍 Reformulated Query: `{response.get('search_query')}`")
                                if response.get("target_document"):
                                    st.caption(f"🎯 Document Routing: `{response.get('target_document')}`")
                                for i, src in enumerate(sources):
                                    st.markdown(f"**[{i+1}] {src.get('filename')} (Page {src.get('page_number')}) - `{src.get('type', 'text').upper()}`**")
                                    st.caption(f"Score: `{round(src.get('score', 0.0), 3)}` | Method: `{src.get('retrieval_method', 'hybrid')}`")
                                    st.text(src.get("content", "")[:250] + "...")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
                "search_query": response.get("search_query"),
                "target_document": response.get("target_document"),
            })

            save_chat(
                project_id=project_id,
                chat_id=st.session_state.active_chat_id,
                title=st.session_state.current_chat_title,
                messages=st.session_state.messages,
                user_id=user_id,
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
            upload_token = f"{project_id}_{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.get("last_uploaded_token") != upload_token:
                save_dir = get_project_docs_dir(project_id)
                saved_file_path = save_dir / uploaded_file.name

                with open(saved_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                progress_bar = st.progress(0, text="Initializing ingestion pipeline...")
                def on_step_progress(stage_name, step, total):
                    pct = int((step / total) * 100)
                    progress_bar.progress(pct, text=f"Executing {stage_name} ({step}/{total})...")

                with st.spinner(f"Ingesting `{uploaded_file.name}`..."):
                    run_ingestion_pipeline(project_id, saved_file_path, progress_callback=on_step_progress, restart=True)
                    st.session_state.last_uploaded_token = upload_token
                    progress_bar.progress(100, text="Ingestion completed successfully!")
                    st.success(f"`{uploaded_file.name}` indexed into PostgreSQL (pgvector)!")
                    st.session_state.inspecting_document = uploaded_file.name
                    st.rerun()

        st.markdown("---")

        # Knowledge Base Sources Header
        st.markdown("<div style='font-size:11px; color:#8b949e; text-transform:uppercase; font-weight:700; margin-bottom:8px;'>Knowledge Base Sources</div>", unsafe_allow_html=True)

        # Scrollable container for Paused Ingestions and Knowledge Base Sources
        sources_container = st.container(height=310)
        with sources_container:
            # Section: Paused / Interrupted Ingestions
            interrupted_docs = get_interrupted_documents(project_id)
            if interrupted_docs:
                st.markdown("##### ⚠️ Paused Ingestions")
                for item in interrupted_docs:
                    fname = item["filename"]
                    lstage = item["last_stage"].capitalize()
                    fpath = get_project_docs_dir(project_id) / fname

                    st.warning(f"**`{fname}`** paused after: **{lstage}**")
                    col_res, col_disc = st.columns([1, 1])
                    with col_res:
                        if st.button(f"▶ Resume", key=f"res_{fname}", type="primary", use_container_width=True):
                            progress_bar = st.progress(0, text="Resuming ingestion from checkpoint...")
                            def on_resume_progress(stage_name, step, total):
                                pct = int((step / total) * 100)
                                progress_bar.progress(pct, text=f"Executing {stage_name} ({step}/{total})...")

                            with st.spinner(f"Resuming `{fname}`..."):
                                run_ingestion_pipeline(project_id, fpath, progress_callback=on_resume_progress)
                                st.success(f"`{fname}` completed and indexed into PostgreSQL (pgvector)!")
                                st.rerun()
                    with col_disc:
                        if st.button(f"🗑️ Discard", key=f"disc_{fname}", use_container_width=True):
                            delete_source_document(project_id, fname)
                            st.info(f"Discarded `{fname}`")
                            st.rerun()

                st.markdown("---")

            # Section: Sources List (Click source to open pipeline directly + 3-dots delete menu)
            docs_dir = get_project_docs_dir(project_id)
            interrupted_filenames = {item["filename"] for item in interrupted_docs}
            source_files = [
                f for f in docs_dir.iterdir()
                if f.is_file()
                and not f.name.endswith(".pipeline.json")
                and not f.name.endswith(".checkpoint.json")
                and f.name not in interrupted_filenames
            ]

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
                            if st.button("🔄 Re-process Document", key=f"reingest_{sfile.name}", use_container_width=True):
                                with st.spinner(f"Re-processing `{sfile.name}`..."):
                                    run_ingestion_pipeline(project_id, sfile, restart=True)
                                    st.success(f"Re-processed `{sfile.name}`!")
                                    st.session_state.inspecting_document = sfile.name
                                    st.rerun()
                            if st.button("🗑️ Delete Source", key=f"del_src_{sfile.name}", type="primary", use_container_width=True):
                                delete_source_document(project_id, sfile.name)
                                st.success(f"Deleted `{sfile.name}`")
                                st.rerun()


# ====================================================================
# AUTHENTICATION & LOGIN SCREEN
# ====================================================================
def render_auth_screen():
    st.markdown("""
    <div style='text-align: center; margin-top: 50px; margin-bottom: 25px;'>
        <div class='retrio-logo' style='font-size: 38px;'>⚡ Retrio</div>
        <div style='color: #8b949e; font-size: 14px; margin-top: 4px;'>Enterprise Multi-Modal RAG & Knowledge Base</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        auth_tab1, auth_tab2 = st.tabs(["🔒 Sign In", "✨ Register Account"])

        with auth_tab1:
            with st.form("login_form"):
                email = st.text_input("Email", placeholder="name@company.com")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                if submitted:
                    if not email.strip() or not password:
                        st.error("Please enter email and password.")
                    else:
                        user = authenticate_user(email.strip(), password)
                        if user:
                            token = create_access_token({"sub": user["id"], "email": user["email"], "role": user.get("role", "user")})
                            st.session_state.current_user = user
                            st.session_state.auth_token = token
                            st.session_state.active_project_id = None
                            st.session_state.messages = []
                            st.rerun()
                        else:
                            st.error("Invalid email or password.")

        with auth_tab2:
            with st.form("register_form"):
                reg_name = st.text_input("Full Name", placeholder="e.g. Alice Smith")
                reg_email = st.text_input("Email", placeholder="name@company.com")
                reg_password = st.text_input("Password (min 6 characters)", type="password", placeholder="••••••••")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="••••••••")
                reg_submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                if reg_submitted:
                    if not reg_email.strip() or not reg_password:
                        st.error("Please enter email and password.")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    else:
                        try:
                            user = create_user(reg_email.strip(), reg_password, reg_name.strip())
                            token = create_access_token({"sub": user["id"], "email": user["email"], "role": user.get("role", "user")})
                            st.session_state.current_user = user
                            st.session_state.auth_token = token
                            st.session_state.active_project_id = None
                            st.session_state.messages = []
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))


# ====================================================================
# MAIN APPLICATION ROUTING
# ====================================================================
def main():
    if not st.session_state.get("current_user"):
        render_auth_screen()
        return

    user_id = st.session_state.current_user["id"]
    if not st.session_state.active_project_id:
        user_projs = list_projects(owner_id=user_id)
        if user_projs:
            st.session_state.active_project_id = user_projs[0]["id"]

    render_retrio_home(st.session_state.active_project_id)


if __name__ == "__main__":
    main()
