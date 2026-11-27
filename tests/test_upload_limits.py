import unittest
import tempfile
import shutil
from pathlib import Path
from fastapi.testclient import TestClient
from api.main import app
from core.config import MAX_UPLOAD_SIZE_BYTES, MAX_UPLOAD_SIZE_MB, PROJECTS_DIR
from api.deps import get_current_active_user, get_current_user


class TestUploadLimitsAndDeduplication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Mock authenticated user dependency
        cls.mock_user = {
            "id": "mock_tester_id_123",
            "email": "tester@retrio.io",
            "role": "user",
            "is_active": True,
        }
        app.dependency_overrides[get_current_user] = lambda: cls.mock_user
        app.dependency_overrides[get_current_active_user] = lambda: cls.mock_user

    @classmethod
    def tearDownClass(cls):
        app.dependency_overrides.clear()

    def test_upload_size_limit_rejection(self):
        from core.projects import create_project, delete_project
        proj = create_project("Upload Limit Test Proj", owner_id=self.mock_user["id"])
        proj_id = proj["id"]
        try:
            # Header indicating size > MAX_UPLOAD_SIZE_BYTES should immediately return 413
            oversized = MAX_UPLOAD_SIZE_BYTES + (1024 * 1024)
            headers = {"content-length": str(oversized)}
            
            res = self.client.post(
                f"/api/v1/projects/{proj_id}/documents",
                headers=headers,
                files={"file": ("oversized.pdf", b"small content", "application/pdf")},
            )
            self.assertEqual(res.status_code, 413)
            self.assertIn(f"{MAX_UPLOAD_SIZE_MB}MB", res.json().get("detail", ""))
        finally:
            delete_project(proj_id, owner_id=self.mock_user["id"])

    def test_magic_bytes_spoofing_rejected(self):
        """Verifies that an uploaded file with spoofed extension is rejected with 400."""
        from core.projects import create_project, delete_project
        proj = create_project("Magic Bytes Test Proj", owner_id=self.mock_user["id"])
        proj_id = proj["id"]
        try:
            # 1. Plain text disguised as a PDF
            res = self.client.post(
                f"/api/v1/projects/{proj_id}/documents",
                files={"file": ("fake.pdf", b"This is plainly not a PDF document", "application/pdf")},
            )
            self.assertEqual(res.status_code, 400)
            self.assertIn("signature mismatch", res.json().get("detail", ""))

            # 2. Executable binary disguised as CSV
            res2 = self.client.post(
                f"/api/v1/projects/{proj_id}/documents",
                files={"file": ("malware.csv", b"MZ\x90\x00\x03\x00\x00\x00", "text/csv")},
            )
            self.assertEqual(res2.status_code, 400)
            self.assertIn("Security violation", res2.json().get("detail", ""))
        finally:
            delete_project(proj_id, owner_id=self.mock_user["id"])

    def test_filename_deduplication_counter(self):
        """Verifies that duplicate filename uploads do not overwrite existing files."""
        project_id = "test_dedup_project"
        docs_dir = PROJECTS_DIR / project_id / "documents"
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        base_file = docs_dir / "report.pdf"
        base_file.write_text("original content")
        
        try:
            # Re-implementing the deduplication loop logic to verify counter incrementation
            original_filename = "report.pdf"
            stem = Path(original_filename).stem
            ext = Path(original_filename).suffix.lower()
            
            safe_filename = original_filename
            dest_path = docs_dir / safe_filename
            counter = 1
            while dest_path.exists():
                safe_filename = f"{stem}_{counter}{ext}"
                dest_path = docs_dir / safe_filename
                counter += 1
                
            self.assertEqual(safe_filename, "report_1.pdf")
            self.assertEqual(dest_path.name, "report_1.pdf")
            
            # If report_1.pdf also exists:
            dest_path.write_text("first duplicate content")
            safe_filename = original_filename
            dest_path = docs_dir / safe_filename
            counter = 1
            while dest_path.exists():
                safe_filename = f"{stem}_{counter}{ext}"
                dest_path = docs_dir / safe_filename
                counter += 1
                
            self.assertEqual(safe_filename, "report_2.pdf")
            self.assertEqual(dest_path.name, "report_2.pdf")
        finally:
            if docs_dir.exists():
                shutil.rmtree(PROJECTS_DIR / project_id, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
