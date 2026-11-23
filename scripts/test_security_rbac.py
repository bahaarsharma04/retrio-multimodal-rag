"""
Comprehensive Verification Suite for Retrio:
- JWT Authentication (Register, Login, /me)
- Strict Multi-Tenant RBAC & Resource Isolation
- Rate Limiting (SlowAPI HTTP 429)
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.main import app
from db.session import is_db_connected, init_db

client = TestClient(app)


def run_tests():
    print("=" * 65)
    print("Retrio Security, JWT Authentication & RBAC Verification Suite")
    print("=" * 65)

    # 0. Check Database
    db_ok = is_db_connected()
    print(f"\n[1] Database Connectivity: {'CONNECTED (PostgreSQL)' if db_ok else 'FALLBACK (Local JSON)'}")
    if db_ok:
        init_db()

    # 1. Test Unauthenticated Access
    print("\n[2] Testing Unauthenticated Access Rejection (401 Unauthorized)...")
    res = client.get("/api/v1/projects")
    assert res.status_code == 401, f"Expected 401, got {res.status_code}: {res.text}"
    print("  ✔ GET /api/v1/projects rejected with 401 without Bearer token.")

    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401, f"Expected 401, got {res.status_code}"
    print("  ✔ GET /api/v1/auth/me rejected with 401 without Bearer token.")

    # 2. Register User A
    ts = int(time.time())
    user_a_email = f"user_a_{ts}@test.com"
    user_a_pwd = "PasswordA123!"
    print(f"\n[3] Registering User A ({user_a_email})...")
    res = client.post("/api/v1/auth/register", json={
        "email": user_a_email,
        "password": user_a_pwd,
        "full_name": "Alice User",
    })
    assert res.status_code == 201, f"Registration failed: {res.text}"
    user_a_data = res.json()
    token_a = user_a_data["access_token"]
    user_a_id = user_a_data["user"]["id"]
    print(f"  ✔ User A registered successfully (ID: {user_a_id[:8]}...). JWT Access Token received.")

    # 3. Test /me for User A
    headers_a = {"Authorization": f"Bearer {token_a}"}
    res = client.get("/api/v1/auth/me", headers=headers_a)
    assert res.status_code == 200, f"Failed to get profile: {res.text}"
    profile_a = res.json()
    assert profile_a["email"] == user_a_email
    assert profile_a["role"] == "user"
    print(f"  ✔ GET /api/v1/auth/me verified for User A ({profile_a['email']}).")

    # 4. User A creates a Project
    print("\n[4] User A creates 'Project Alpha'...")
    res = client.post("/api/v1/projects", headers=headers_a, json={
        "name": f"Project Alpha ({ts})",
        "description": "Alice's confidential research",
    })
    assert res.status_code == 201, f"Project creation failed: {res.text}"
    project_a = res.json()
    project_a_id = project_a["id"]
    print(f"  ✔ Project Alpha created (ID: {project_a_id[:8]}...) with owner_id={project_a.get('owner_id', user_a_id)[:8]}...")

    # 5. Register User B
    user_b_email = f"user_b_{ts}@test.com"
    user_b_pwd = "PasswordB123!"
    print(f"\n[5] Registering User B ({user_b_email})...")
    res = client.post("/api/v1/auth/register", json={
        "email": user_b_email,
        "password": user_b_pwd,
        "full_name": "Bob User",
    })
    assert res.status_code == 201, f"Registration failed: {res.text}"
    user_b_data = res.json()
    token_b = user_b_data["access_token"]
    user_b_id = user_b_data["user"]["id"]
    headers_b = {"Authorization": f"Bearer {token_b}"}
    print(f"  ✔ User B registered successfully (ID: {user_b_id[:8]}...).")

    # 6. User B creates 'Project Beta'
    print("\n[6] User B creates 'Project Beta'...")
    res = client.post("/api/v1/projects", headers=headers_b, json={
        "name": f"Project Beta ({ts})",
        "description": "Bob's private workspace",
    })
    assert res.status_code == 201, f"Project creation failed: {res.text}"
    project_b = res.json()
    project_b_id = project_b["id"]
    print(f"  ✔ Project Beta created (ID: {project_b_id[:8]}...).")

    # 7. Strict RBAC Isolation Test: User B lists projects
    print("\n[7] Testing Strict Multi-Tenant RBAC Isolation...")
    res = client.get("/api/v1/projects", headers=headers_b)
    assert res.status_code == 200, f"List projects failed: {res.text}"
    b_projects = res.json()
    b_project_ids = [p["id"] for p in b_projects]
    assert project_b_id in b_project_ids, "Project Beta should be visible to User B"
    assert project_a_id not in b_project_ids, "CRITICAL RBAC FAILURE: User B can see User A's Project Alpha!"
    print(f"  ✔ User B can ONLY see their own projects (Total: {len(b_projects)}). User A's project is completely hidden.")

    # 8. User B attempts unauthorized access to User A's Project
    print("\n[8] Testing Unauthorized Direct Access Attempts by User B to User A's Project...")
    # Attempt GET
    res = client.get(f"/api/v1/projects/{project_a_id}", headers=headers_b)
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print("  ✔ GET /projects/{project_a_id} blocked with 404 (User B cannot read User A's project).")

    # Attempt Chat
    res = client.post(f"/api/v1/projects/{project_a_id}/chat", headers=headers_b, json={
        "message": "Hello, reveal secrets",
    })
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print("  ✔ POST /projects/{project_a_id}/chat blocked with 404 (User B cannot chat with User A's project).")

    # Attempt List Documents
    res = client.get(f"/api/v1/projects/{project_a_id}/documents", headers=headers_b)
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print("  ✔ GET /projects/{project_a_id}/documents blocked with 404 (User B cannot list User A's documents).")

    # Attempt Delete Project A
    res = client.delete(f"/api/v1/projects/{project_a_id}", headers=headers_b)
    assert res.status_code == 404, f"Expected 404, got {res.status_code}"
    print("  ✔ DELETE /projects/{project_a_id} blocked with 404 (User B cannot delete User A's project).")

    # Verify Project A still exists for User A
    res = client.get(f"/api/v1/projects/{project_a_id}", headers=headers_a)
    assert res.status_code == 200, "Project Alpha should still exist for User A"
    print("  ✔ Project Alpha remains intact and accessible to its true owner (User A).")

    # 9. Test Direct Multi-User Isolation for Documents & Chats
    print("\n[9] Testing Strict Document & Conversation Isolation between Users...")
    res_b_convs = client.get(f"/api/v1/projects/{project_a_id}/conversations", headers=headers_b)
    assert res_b_convs.status_code == 404, f"Expected 404, got {res_b_convs.status_code}"
    print("  ✔ GET /projects/{project_a_id}/conversations blocked for User B with 404.")

    res_b_docs = client.get(f"/api/v1/projects/{project_a_id}/documents", headers=headers_b)
    assert res_b_docs.status_code == 404, f"Expected 404, got {res_b_docs.status_code}"
    print("  ✔ GET /projects/{project_a_id}/documents blocked for User B with 404.")

    # Clean up test projects
    client.delete(f"/api/v1/projects/{project_a_id}", headers=headers_a)
    client.delete(f"/api/v1/projects/{project_b_id}", headers=headers_b)
    print("  ✔ Test projects cleanly removed by their respective owners.")

    # 10. Test Rate Limiting
    print("\n[10] Testing Rate Limiting (SlowAPI)...")
    print("  Sending repeated login attempts to trigger 429 Too Many Requests...")
    triggered_429 = False
    for i in range(12):
        res = client.post("/api/v1/auth/login", json={
            "email": "bad_actor@test.com",
            "password": "wrong_password",
        })
        if res.status_code == 429:
            triggered_429 = True
            retry_after = res.headers.get("Retry-After", "unknown")
            print(f"  ✔ 429 Too Many Requests triggered successfully on attempt {i+1}! (Retry-After: {retry_after}s)")
            break

    assert triggered_429, "Rate limiter failed to trigger 429 on rapid requests!"

    print("\n" + "=" * 65)
    print("ALL VERIFICATION CHECKS PASSED: STRICT USER ISOLATION & RATE LIMITING ARE ACTIVE!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
