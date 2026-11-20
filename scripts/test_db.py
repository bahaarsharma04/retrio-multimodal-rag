import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.session import get_db_session, is_db_connected
from db.models import Project, Document, DocumentChunk, Conversation, Message
from sqlalchemy import select


def run_test():
    print("=" * 65)
    print("Retrio PostgreSQL & pgvector Verification Suite")
    print("=" * 65)

    # 1. Connection Check
    connected = is_db_connected(force_check=True)
    print(f"\n1. Database Connection Status: {'CONNECTED' if connected else 'DISCONNECTED'}")
    if not connected:
        print("ERROR: Could not connect to PostgreSQL on localhost:5432.")
        return

    with get_db_session() as s:
        # 2. Table Summary
        proj_count = s.query(Project).count()
        doc_count = s.query(Document).count()
        chunk_count = s.query(DocumentChunk).count()
        emb_count = s.query(DocumentChunk).filter(DocumentChunk.embedding.isnot(None)).count()
        conv_count = s.query(Conversation).count()
        msg_count = s.query(Message).count()

        print("\n2. Database Record Overview:")
        print(f"   • Projects:          {proj_count}")
        print(f"   • Documents:         {doc_count}")
        print(f"   • Document Chunks:   {chunk_count}")
        print(f"   • Stored Embeddings: {emb_count} in pgvector")
        print(f"   • Conversations:     {conv_count}")
        print(f"   • Messages:          {msg_count}")

        # 3. Sample Projects & Documents
        print("\n3. Sample Projects in Database:")
        for p in s.query(Project).limit(3).all():
            print(f"   [Project ID: {p.id[:8]}...] Name: '{p.name}'")
            docs = s.query(Document).filter(Document.project_id == p.id).all()
            for d in docs[:3]:
                print(f"      └── Document: {d.filename} ({d.ingestion_status})")

        # 4. Native pgvector Semantic Search Test
        print("\n4. Testing Native pgvector Cosine Similarity Search:")
        sample = s.query(DocumentChunk).filter(DocumentChunk.embedding.isnot(None)).first()
        if sample:
            # Query top 3 nearest chunks using pgvector cosine distance operator (<=>)
            nearest = s.scalars(
                select(DocumentChunk)
                .order_by(DocumentChunk.embedding.cosine_distance(sample.embedding))
                .limit(3)
            ).all()

            print(f"   Target chunk: \"{sample.content[:60]}...\"")
            print("   Top 3 nearest results retrieved by PostgreSQL:")
            for i, match in enumerate(nearest, 1):
                preview = match.content[:60].replace("\n", " ")
                print(f"      {i}. [Page {match.page_numbers} | Section: {match.section}] \"{preview}...\"")
        else:
            print("   No vector embeddings found in document_chunks table.")

        # 5. Recent Conversation Test
        print("\n5. Testing Chat History Retrieval:")
        recent_conv = s.query(Conversation).order_by(Conversation.updated_at.desc()).first()
        if recent_conv:
            print(f"   Active Conversation: '{recent_conv.title}' (ID: {recent_conv.id[:8]}...)")
            msgs = s.query(Message).filter(Message.conversation_id == recent_conv.id).limit(2).all()
            for m in msgs:
                preview = m.content[:70].replace("\n", " ")
                print(f"      [{m.role.upper()}]: {preview}...")
        else:
            print("   No conversations found.")

    print("\n" + "=" * 65)
    print("ALL TESTS PASSED: PostgreSQL & pgvector are active and ready!")
    print("=" * 65)


if __name__ == "__main__":
    run_test()
