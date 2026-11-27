import unittest
from fastapi.testclient import TestClient
from api.main import app
from core.security import (
    create_access_token,
    decode_access_token,
    create_refresh_token,
    decode_refresh_token,
    revoke_token,
    is_token_revoked,
)


class TestAuthAndRBAC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_unauthenticated_requests_rejected(self):
        # Accessing projects list without bearer token must return 401
        res = self.client.get("/api/v1/projects")
        self.assertEqual(res.status_code, 401)

        # Accessing /auth/me without bearer token must return 401
        res = self.client.get("/api/v1/auth/me")
        self.assertEqual(res.status_code, 401)

    def test_jwt_token_creation_and_decoding(self):
        payload = {
            "sub": "user_123456",
            "email": "engineer@retrio.io",
            "role": "user",
        }
        token = create_access_token(payload)
        self.assertIsInstance(token, str)
        self.assertTrue(len(token) > 20)

        decoded = decode_access_token(token)
        self.assertEqual(decoded.get("sub"), "user_123456")
        self.assertEqual(decoded.get("email"), "engineer@retrio.io")
        self.assertEqual(decoded.get("role"), "user")

    def test_invalid_jwt_rejected(self):
        headers = {"Authorization": "Bearer invalid.jwt.token"}
        res = self.client.get("/api/v1/auth/me", headers=headers)
        self.assertEqual(res.status_code, 401)

    def test_refresh_token_creation_and_decoding(self):
        payload = {
            "sub": "user_78910",
            "email": "refresh_tester@retrio.io",
            "role": "user",
        }
        ref_token = create_refresh_token(payload)
        self.assertIsInstance(ref_token, str)
        decoded = decode_refresh_token(ref_token)
        self.assertEqual(decoded.get("sub"), "user_78910")
        self.assertEqual(decoded.get("type"), "refresh")

    def test_logout_and_revocation(self):
        # Create user token
        payload = {"sub": "user_logout_test", "email": "logout@retrio.io", "role": "user"}
        acc_token = create_access_token(payload)
        ref_token = create_refresh_token(payload)

        # Before logout: not revoked
        self.assertFalse(is_token_revoked(acc_token))
        self.assertFalse(is_token_revoked(ref_token))

        # Perform logout
        headers = {"Authorization": f"Bearer {acc_token}"}
        res = self.client.post("/api/v1/auth/logout", headers=headers, json={"refresh_token": ref_token})
        self.assertEqual(res.status_code, 200)

        # After logout: both tokens must be revoked
        self.assertTrue(is_token_revoked(acc_token))
        self.assertTrue(is_token_revoked(ref_token))

        # Using revoked access token returns 401
        res2 = self.client.get("/api/v1/auth/me", headers=headers)
        self.assertEqual(res2.status_code, 401)

        # Using revoked refresh token on /refresh returns 401
        res3 = self.client.post("/api/v1/auth/refresh", json={"refresh_token": ref_token})
        self.assertEqual(res3.status_code, 401)


if __name__ == "__main__":
    unittest.main()
