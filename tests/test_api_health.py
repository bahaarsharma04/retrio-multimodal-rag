import unittest
from fastapi.testclient import TestClient
from api.main import app
from core.config import ALLOWED_ORIGINS


class TestApiHealthAndCORS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_endpoint(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "online")
        self.assertEqual(data.get("title"), "Retrio Multi-Modal RAG API")

    def test_health_check_endpoint(self):
        res = self.client.get("/api/v1/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("database_connected", data)

    def test_cors_authorized_origin(self):
        target_origin = ALLOWED_ORIGINS[0] if ALLOWED_ORIGINS else "http://localhost:3000"
        headers = {
            "Origin": target_origin,
            "Access-Control-Request-Method": "GET",
        }
        res = self.client.options("/api/v1/health", headers=headers)
        # CORS middleware returns 200 for allowed origin OPTIONS
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), target_origin)

    def test_cors_unauthorized_origin_rejected(self):
        unauthorized_origin = "http://malicious-attacker-site.com"
        headers = {
            "Origin": unauthorized_origin,
            "Access-Control-Request-Method": "GET",
        }
        res = self.client.options("/api/v1/health", headers=headers)
        # Unauthorized origin should not receive access-control-allow-origin header
        self.assertNotEqual(res.headers.get("access-control-allow-origin"), unauthorized_origin)


if __name__ == "__main__":
    unittest.main()
