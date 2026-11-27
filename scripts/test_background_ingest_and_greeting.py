"""
Verification script for Background Document Ingestion and Greeting Interception.
"""
import sys
import io
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from api.main import app
from core.rag_chain import generate_rag_response, is_greeting_query
from db.session import get_db_session
from db.models import User, Project
from core.security import create_access_token


def test_greeting_interception():
    print("=" * 70)
    print("1. TESTING GREETING INTERCEPTION")
    print("=" * 70)

    # Unit tests for is_greeting_query
    assert is_greeting_query("hi") is True
    assert is_greeting_query("Hello!") is True
    assert is_greeting_query("hey there") is True
    assert is_greeting_query("good morning") is True
    assert is_greeting_query("hi what does page 2 say?") is False
    assert is_greeting_query("hello what is in table 1 of the document?") is False
    print("✔ is_greeting_query() pattern and guardrail checks passed.")

    # Greeting in project with documents
    res_with_docs = generate_rag_response(
        project_id="ade1ae6a-b265-4090-8d43-5875909edbf9",
        query="hi",
    )
    print(f"✔ Greeting (Project with documents):\n  \"{res_with_docs['answer']}\"")
    assert "ready" in res_with_docs["answer"].lower()
    assert res_with_docs["retrieval_method"] == "greeting"
    assert res_with_docs["sources"] == []

    # Greeting in project without documents
    res_empty = generate_rag_response(
        project_id="00000000-0000-0000-0000-000000000000",
        query="hello",
    )
    print(f"✔ Greeting (Empty Project):\n  \"{res_empty['answer']}\"")
    assert "upload your documents" in res_empty["answer"].lower()
    assert res_empty["retrieval_method"] == "greeting"
    assert res_empty["sources"] == []
    print("\n")


def test_background_ingestion():
    print("=" * 70)
    print("2. TESTING ASYNCHRONOUS BACKGROUND DOCUMENT INGESTION")
    print("=" * 70)

    client = TestClient(app)

    # 1. Get or create test user and project
    with get_db_session() as s:
        user = s.query(User).filter(User.email == "test_bg_user@retrio.com").first()
        if not user:
            user = User(
                email="test_bg_user@retrio.com",
                full_name="Background Test User",
                hashed_password="hashed_pw_test",
                role="user",
            )
            s.add(user)
            s.commit()
            s.refresh(user)

        proj = s.query(Project).filter(
            Project.owner_id == user.id,
            Project.name == "Background Ingest Test Proj"
        ).first()
        if not proj:
            proj = Project(
                name="Background Ingest Test Proj",
                owner_id=user.id,
                description="Testing background upload",
            )
            s.add(proj)
            s.commit()
            s.refresh(proj)

        user_id = str(user.id)
        project_id = str(proj.id)

    token = create_access_token({"sub": user_id, "email": "test_bg_user@retrio.com", "role": "user"})
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload document via POST /api/v1/projects/{project_id}/documents
    test_content = (
        "Project Architecture Overview\n"
        "Retrio uses PostgreSQL pgvector as its single unified data store.\n"
        "All chunking, embedding, and retrieval operations are decoupled and asynchronous.\n"
    )
    test_filename = f"bg_test_doc_{int(time.time())}.txt"
    files = {
        "file": (test_filename, io.BytesIO(test_content.encode("utf-8")), "text/plain")
    }

    t0 = time.perf_counter()
    response = client.post(
        f"/api/v1/projects/{project_id}/documents",
        files=files,
        headers=headers,
    )
    upload_duration_ms = (time.perf_counter() - t0) * 1000

    print(f"✔ Upload HTTP Response status: {response.status_code} in {upload_duration_ms:.1f}ms")
    assert response.status_code == 202, f"Expected 202, got {response.status_code}: {response.text}"

    data = response.json()
    print(f"✔ Upload Response payload: {data}")
    assert data["status"] == "processing"
    assert data["filename"] == test_filename

    # 3. Poll GET /status endpoint
    print(f"\nPolling status for '{test_filename}'...")
    max_wait = 25
    start_poll = time.time()
    final_status = None

    while time.time() - start_poll < max_wait:
        status_resp = client.get(
            f"/api/v1/projects/{project_id}/documents/{test_filename}/status",
            headers=headers,
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        final_status = status_data["status"]
        print(f"  [{time.time() - start_poll:.1f}s] status={final_status} | total_chunks={status_data.get('total_chunks')}")
        if final_status in ("completed", "failed"):
            break
        time.sleep(1.0)

    assert final_status == "completed", f"Expected completed, got {final_status}"
    print(f"✔ Ingestion successfully completed in background!")

    # 4. Check Pipeline Audit Record
    pipeline_resp = client.get(
        f"/api/v1/projects/{project_id}/documents/{test_filename}/pipeline",
        headers=headers,
    )
    if pipeline_resp.status_code == 200:
        p_data = pipeline_resp.json()
        print("\n✔ Pipeline Execution Audit Record:")
        print(f"   • Status:      {p_data.get('status')}")
        print(f"   • Stages Run:  {list(p_data.get('stages', {}).keys())}")
        for s_name, s_info in p_data.get("stages", {}).items():
            print(f"      - {s_name:<15s}: status={s_info.get('status')} | details={s_info}")

    # 5. Cleanup test document
    del_resp = client.delete(
        f"/api/v1/projects/{project_id}/documents/{test_filename}",
        headers=headers,
    )
    print(f"\n✔ Cleaned up test document ({del_resp.status_code}).")


if __name__ == "__main__":
    test_greeting_interception()
    test_background_ingestion()
    print("=" * 70)
    print("ALL TESTS PASSED: GREETINGS & BACKGROUND INGESTION VERIFIED!")
    print("=" * 70)
