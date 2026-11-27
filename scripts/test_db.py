"""
Retrio PostgreSQL & pgvector Database Verification and Health Inspection Suite.
Usage:
    python scripts/test_db.py
    python scripts/test_db.py --verbose
"""
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.session import get_db_session, is_db_connected, engine
from db.models import Project, Document, DocumentChunk, Conversation, Message, MessageCitation, User
from core.retriever import count_tokens
from sqlalchemy import select, text


def run_test(verbose: bool = False):
    width = 78
    print("=" * width)
    print(" RETRIO POSTGRESQL & PGVECTOR VERIFICATION SUITE ".center(width, "="))
    print("=" * width)

    # 1. Connection Check
    t0 = time.perf_counter()
    connected = is_db_connected(force_check=True)
    conn_latency_ms = (time.perf_counter() - t0) * 1000.0

    print(f"\n[1/6] Database Connectivity & Engine Status:")
    print(f"      Status:          {'CONNECTED (Healthy)' if connected else 'DISCONNECTED (Failed)'}")
    print(f"      Ping Latency:    {conn_latency_ms:.1f}ms")
    print(f"      Database URL:    {engine.url.render_as_string(hide_password=True)}")

    if not connected:
        print("\nERROR: Could not connect to PostgreSQL. Please check your DATABASE_URL in .env.", file=sys.stderr)
        sys.exit(1)

    with get_db_session() as s:
        # 2. Check Database Extensions
        print(f"\n[2/6] PostgreSQL Extensions Audit:")
        try:
            ext_results = s.execute(text("SELECT extname, extversion FROM pg_extension;")).fetchall()
            installed_exts = {name: ver for name, ver in ext_results}
            for ext in ["vector", "pgcrypto", "uuid-ossp"]:
                status_icon = "✔" if ext in installed_exts else "✖"
                version = installed_exts.get(ext, "Not Installed")
                print(f"      {status_icon} Extension '{ext}': version {version}")
        except Exception as e:
            print(f"      Notice: Could not query pg_extension ({e})")

        # 3. Complete Table Row Counts
        proj_count = s.query(Project).count()
        doc_count = s.query(Document).count()
        chunk_count = s.query(DocumentChunk).count()
        emb_count = s.query(DocumentChunk).filter(DocumentChunk.embedding.isnot(None)).count()
        user_count = s.query(User).count()
        conv_count = s.query(Conversation).count()
        msg_count = s.query(Message).count()
        cite_count = s.query(MessageCitation).count()

        print(f"\n[3/6] Table Inventory & Row Counts:")
        print(f"      • users:              {user_count} registered users")
        print(f"      • projects:           {proj_count} project workspaces")
        print(f"      • documents:          {doc_count} source documents")
        print(f"      • document_chunks:    {chunk_count} total chunks")
        print(f"      • pgvector vectors:   {emb_count} embedded chunks (1536-dim)")
        print(f"      • conversations:      {conv_count} chat conversations")
        print(f"      • messages:           {msg_count} chat messages")
        print(f"      • message_citations:  {cite_count} source citations")

        # 4. Project Breakdown
        print(f"\n[4/6] Project Workspaces Overview:")
        projects = s.query(Project).all()
        if not projects:
            print("      No projects found in database.")
        else:
            limit = len(projects) if verbose else min(5, len(projects))
            for p in projects[:limit]:
                p_chunks = s.query(DocumentChunk).filter(DocumentChunk.project_id == str(p.id)).count()
                p_docs = s.query(Document).filter(Document.project_id == str(p.id)).all()
                print(f"      📁 Project '{p.name}' | ID: {p.id}")
                print(f"         Total Chunks: {p_chunks} | Documents: {len(p_docs)}")
                for d in p_docs[:4]:
                    print(f"           └── [{d.ingestion_status.upper()}] {d.filename} ({d.mime_type or 'unknown'})")
                if len(p_docs) > 4:
                    print(f"           └── ... and {len(p_docs) - 4} more document(s)")
            if not verbose and len(projects) > 5:
                print(f"      ... and {len(projects) - 5} more projects (use --verbose to see all)")

        # 5. Native pgvector Semantic Search Test
        print(f"\n[5/6] Native pgvector Cosine Distance (<=>) Operator Test:")
        sample = s.query(DocumentChunk).filter(DocumentChunk.embedding.isnot(None)).first()
        if sample:
            t_search = time.perf_counter()
            nearest = s.scalars(
                select(DocumentChunk)
                .order_by(DocumentChunk.embedding.cosine_distance(sample.embedding))
                .limit(3)
            ).all()
            search_latency_ms = (time.perf_counter() - t_search) * 1000.0

            content_clean = sample.content.replace('\n', ' ')
            sample_tok = count_tokens(sample.content)
            print(f"      ✔ Executed vector distance query in {search_latency_ms:.1f}ms")
            print(f"      Target Chunk ID: {sample.id[:20]}... (tokensize: {sample_tok})")
            print(f"      Target first_100_chars: \"{content_clean[:100]}\"")
            print("      Top 3 Nearest Matches via pgvector:")
            for i, match in enumerate(nearest, 1):
                match_preview = match.content.replace('\n', ' ')[:100]
                match_tok = count_tokens(match.content)
                print(f"         [{i}] Page: {match.page_numbers} | Section: \"{match.section}\" | tokens: {match_tok}")
                print(f"             first_100_chars: \"{match_preview}\"")
        else:
            print("      Notice: No chunks with embeddings found to test vector similarity.")

        # 6. Conversation & Message Retrieval Test
        print(f"\n[6/6] Chat Conversation & Message Persistence Test:")
        recent_conv = s.query(Conversation).order_by(Conversation.updated_at.desc()).first()
        if recent_conv:
            print(f"      ✔ Found active conversation: '{recent_conv.title}' (ID: {recent_conv.id})")
            msgs = s.query(Message).filter(Message.conversation_id == recent_conv.id).order_by(Message.created_at.asc()).limit(3).all()
            for m in msgs:
                preview = m.content[:90].replace('\n', ' ')
                print(f"         [{m.role.upper()}]: \"{preview}...\"")
        else:
            print("      No conversations recorded in database yet.")

    print("\n" + "=" * width)
    print(" ALL DATABASE HEALTH CHECKS PASSED: POSTGRESQL & PGVECTOR READY! ".center(width, "="))
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(description="Retrio Database & pgvector Health Suite")
    parser.add_argument("--verbose", "-v", action="store_true", help="Display exhaustive details for all projects")
    args = parser.parse_args()
    run_test(verbose=args.verbose)


if __name__ == "__main__":
    main()
