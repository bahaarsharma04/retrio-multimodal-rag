import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
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

# API Keys & Database
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/retrio_db")

# JWT & Security Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "retrio-secret-key-32-bytes-minimum-for-security-2026")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# Model Configurations
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Ingestion Pipeline Defaults
MAX_CHUNK_CHARACTERS = 1500
NEW_AFTER_N_CHARS = 1000
COMBINE_UNDER_N_CHARS = 200
CHUNK_OVERLAP_CHARACTERS = 250

# Retrieval Defaults
HYBRID_SEARCH_TOP_K = 15
RERANK_TOP_N = 8


