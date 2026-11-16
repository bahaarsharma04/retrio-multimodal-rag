import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime

# Set project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import PROJECTS_DIR
from db.session import get_db_session, is_db_connected, init_db
from db.models import (
    Project,
    Document,
    PipelineRun,
    PipelineStage,
    DocumentChunk,
    Conversation,
    Message,
)


def clean_pg_text(val):
    """PostgreSQL TEXT fields cannot contain NUL (0x00) bytes."""
    if isinstance(val, str):
        return val.replace("\x00", "").strip()
    return val


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def migrate_all():
    print("=" * 60)
    print("Retrio: JSON & Chroma to PostgreSQL Migration Tool")
    print("=" * 60)

    if not is_db_connected(force_check=True):
        print("ERROR: Cannot connect to PostgreSQL database.")
        print("Please ensure PostgreSQL is running (e.g. 'docker compose up -d')")
        print("and DATABASE_URL is configured in your .env file.")
        sys.exit(1)

    print("Initializing database schema...")
    if not init_db():
        print("ERROR: Failed to initialize database tables.")
        sys.exit(1)

    stats = {
        "projects": 0,
        "documents": 0,
        "chunks": 0,
        "embeddings_transferred": 0,
        "conversations": 0,
        "messages": 0,
    }

    if not PROJECTS_DIR.exists():
        print(f"No projects directory found at: {PROJECTS_DIR}")
        return

    with get_db_session() as session:
        for pdir in PROJECTS_DIR.iterdir():
            if not pdir.is_dir():
                continue

            project_id = pdir.name
            pfile = pdir / "project.json"
            if not pfile.exists():
                continue

            # 1. Migrate Project
            with open(pfile, "r", encoding="utf-8") as f:
                pdata = json.load(f)

            db_proj = session.query(Project).filter(Project.id == project_id).first()
            if not db_proj:
                db_proj = Project(
                    id=project_id,
                    name=pdata.get("name", "Untitled Project"),
                    description=pdata.get("description", ""),
                )
                session.add(db_proj)
                stats["projects"] += 1
                print(f"-> Project migrated: {db_proj.name} ({project_id})")
            session.flush()

            # 2. Extract ChromaDB vectors if available
            chroma_vectors = {}
            chroma_dir = pdir / "chroma"
            if chroma_dir.exists():
                try:
                    import chromadb
                    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
                    cols = chroma_client.list_collections()
                    for col in cols:
                        col_data = col.get(include=["embeddings"])
                        ids = col_data.get("ids", [])
                        embeddings = col_data.get("embeddings", [])
                        for cid, emb in zip(ids, embeddings):
                            if emb is not None:
                                chroma_vectors[cid] = emb
                    if chroma_vectors:
                        print(f"   Loaded {len(chroma_vectors)} pre-computed embeddings from ChromaDB")
                except Exception as e:
                    print(f"   Notice: Could not read Chroma vectors ({e}); will migrate chunks without vectors")

            # 3. Migrate Documents & Pipeline Data
            docs_dir = pdir / "documents"
            if docs_dir.exists():
                for doc_file in docs_dir.iterdir():
                    if doc_file.is_file() and not doc_file.name.endswith((".pipeline.json", ".checkpoint.json")):
                        filename = doc_file.name
                        file_hash = compute_file_sha256(doc_file)
                        ext = doc_file.suffix.lower()

                        db_doc = (
                            session.query(Document)
                            .filter(Document.project_id == project_id, Document.filename == filename)
                            .first()
                        )
                        if not db_doc:
                            db_doc = Document(
                                project_id=project_id,
                                filename=filename,
                                file_path=str(doc_file),
                                file_hash=file_hash,
                                mime_type="application/pdf" if ext == ".pdf" else "text/plain",
                                ingestion_status="completed",
                            )
                            session.add(db_doc)
                            session.flush()
                            stats["documents"] += 1
                            print(f"   -> Document registered: {filename}")

                        # Check for pipeline json
                        pipeline_file = docs_dir / f"{filename}.pipeline.json"
                        if pipeline_file.exists():
                            try:
                                with open(pipeline_file, "r", encoding="utf-8") as pf:
                                    pip_data = json.load(pf)

                                stages = pip_data.get("stages", {})
                                chunks_list = (
                                    stages.get("view_chunks", {}).get("chunks")
                                    or stages.get("chunking", {}).get("chunks", [])
                                    or stages.get("summarisation", {}).get("enriched_chunks", [])
                                )

                                for idx, c in enumerate(chunks_list):
                                    cid = c.get("chunk_id") or c.get("id") or f"chunk_{idx}"
                                    existing_chunk = session.query(DocumentChunk).filter(DocumentChunk.id == cid).first()
                                    if not existing_chunk:
                                        p_raw = c.get("page_numbers") or c.get("page_range") or c.get("page_number") or 1
                                        if isinstance(p_raw, list):
                                            pages = [int(p) for p in p_raw if str(p).isdigit()] or [1]
                                        elif isinstance(p_raw, str) and "-" in p_raw:
                                            pages = [int(p) for p in p_raw.split("-") if p.isdigit()] or [1]
                                        else:
                                            pages = [int(p_raw)] if str(p_raw).isdigit() else [1]

                                        vec = chroma_vectors.get(cid)
                                        if vec is not None:
                                            vec = list(vec)
                                            stats["embeddings_transferred"] += 1

                                        meta = c.get("metadata") or {}
                                        html_val = clean_pg_text(c.get("text_as_html") or meta.get("text_as_html"))

                                        db_chunk = DocumentChunk(
                                            id=cid,
                                            document_id=db_doc.id,
                                            project_id=project_id,
                                            chunk_index=idx,
                                            chunk_type=c.get("type", "text"),
                                            section=clean_pg_text(c.get("section") or meta.get("section") or "General"),
                                            page_numbers=pages,
                                            content=clean_pg_text(c.get("content", "")),
                                            text_as_html=html_val,
                                            summary=clean_pg_text(c.get("summary")),
                                            indexed_text=clean_pg_text(c.get("indexed_text")),
                                            content_hash=c.get("content_hash", ""),
                                            char_count=c.get("char_count", len(c.get("content", ""))),
                                            embedding=vec,
                                        )
                                        session.add(db_chunk)
                                        stats["chunks"] += 1
                            except Exception as e:
                                print(f"   Warning: Failed to import pipeline chunks for {filename}: {e}")

            session.flush()

            # 4. Migrate Chat Conversations
            chats_dir = pdir / "chats"
            if chats_dir.exists():
                for chat_file in chats_dir.glob("*.json"):
                    try:
                        with open(chat_file, "r", encoding="utf-8") as cf:
                            cdata = json.load(cf)

                        chat_id = cdata.get("chat_id") or chat_file.stem
                        clean_title = clean_pg_text(cdata.get("title")) or "Untitled Chat"
                        db_conv = session.query(Conversation).filter(Conversation.id == chat_id).first()
                        if not db_conv:
                            db_conv = Conversation(
                                id=chat_id,
                                project_id=project_id,
                                title=clean_title,
                            )
                            session.add(db_conv)
                            session.flush()
                            stats["conversations"] += 1

                            # Add messages
                            messages = cdata.get("messages", [])
                            for m in messages:
                                msg = Message(
                                    conversation_id=chat_id,
                                    role=m.get("role", "user"),
                                    content=clean_pg_text(m.get("content", "")),
                                    sources=m.get("sources"),
                                    rewritten_query=clean_pg_text(m.get("rewritten_query")),
                                    latency_ms=m.get("latency_ms", 0.0),
                                )
                                session.add(msg)
                                stats["messages"] += 1
                    except Exception as e:
                        print(f"   Warning: Failed to migrate chat file {chat_file.name}: {e}")

            session.commit()

        session.commit()

    print("\n" + "=" * 60)
    print("Migration Summary to PostgreSQL:")
    print(f"  • Projects imported:             {stats['projects']}")
    print(f"  • Documents registered:          {stats['documents']}")
    print(f"  • Document chunks saved:         {stats['chunks']}")
    print(f"  • Chroma vectors migrated:       {stats['embeddings_transferred']} (0 OpenAI API calls!)")
    print(f"  • Chat conversations imported:   {stats['conversations']}")
    print(f"  • Messages saved:                {stats['messages']}")
    print("=" * 60)
    print("Migration completed successfully!")


if __name__ == "__main__":
    migrate_all()
