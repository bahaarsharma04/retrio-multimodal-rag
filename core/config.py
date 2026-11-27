import os
from pathlib import Path
from typing import List, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
load_dotenv()
os.environ["UNSTRUCTURED_DISABLE_NOTICE"] = "1"

# Project directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
LOGS_DIR = BASE_DIR / "logs"

# Ensure data & logs directories exist
DATA_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Explicitly load .env from absolute BASE_DIR so it works from any working directory
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """
    Centralized, strongly-typed application configuration managed via pydantic-settings.
    Automatically loads from environment variables and .env file with validation.
    """
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & Database
    OPENAI_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/retrio_db"

    # JWT & Security Configuration
    JWT_SECRET_KEY: str = "" 
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30  # Shortened to 30 minutes for security
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7     # 7-day refresh token rotation

    # CORS Configuration - Restrict to trusted frontend origins
    ALLOWED_ORIGINS: Union[List[str], str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
    ]

    # Upload Limits
    MAX_UPLOAD_SIZE_MB: int = 5

    # Model Configurations
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Ingestion Pipeline Defaults
    MAX_CHUNK_CHARACTERS: int = 1500
    NEW_AFTER_N_CHARS: int = 1000
    COMBINE_UNDER_N_CHARS: int = 200
    CHUNK_OVERLAP_CHARACTERS: int = 250

    # Retrieval Defaults
    HYBRID_SEARCH_TOP_K: int = 15
    RERANK_TOP_N: int = 8

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def validate_jwt_secret_key(cls, v: str) -> str:
        if not v or v.strip() == "" or v == "retrio-secret-key-32-bytes-minimum-for-security-2026":
            raise ValueError(
                "CRITICAL SECURITY CONFIGURATION ERROR: JWT_SECRET_KEY environment variable is missing or insecure. "
                "The application refuses to start without a dedicated, cryptographically secure secret key. "
                "Please generate a secure key (e.g. via `python -c 'import secrets; print(secrets.token_hex(32))'`) "
                "and set JWT_SECRET_KEY in your .env file."
            )
        return v

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @property
    def MAX_UPLOAD_SIZE_BYTES(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


# Initialize settings instance
settings = Settings()

# Module-level exports for full backwards compatibility
OPENAI_API_KEY = settings.OPENAI_API_KEY
COHERE_API_KEY = settings.COHERE_API_KEY
DATABASE_URL = settings.DATABASE_URL
JWT_SECRET_KEY = settings.JWT_SECRET_KEY
JWT_ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_DAYS = settings.REFRESH_TOKEN_EXPIRE_DAYS
ALLOWED_ORIGINS = settings.ALLOWED_ORIGINS
MAX_UPLOAD_SIZE_MB = settings.MAX_UPLOAD_SIZE_MB
MAX_UPLOAD_SIZE_BYTES = settings.MAX_UPLOAD_SIZE_BYTES
DEFAULT_LLM_MODEL = settings.DEFAULT_LLM_MODEL
DEFAULT_EMBEDDING_MODEL = settings.DEFAULT_EMBEDDING_MODEL
MAX_CHUNK_CHARACTERS = settings.MAX_CHUNK_CHARACTERS
NEW_AFTER_N_CHARS = settings.NEW_AFTER_N_CHARS
COMBINE_UNDER_N_CHARS = settings.COMBINE_UNDER_N_CHARS
CHUNK_OVERLAP_CHARACTERS = settings.CHUNK_OVERLAP_CHARACTERS
HYBRID_SEARCH_TOP_K = settings.HYBRID_SEARCH_TOP_K
RERANK_TOP_N = settings.RERANK_TOP_N
