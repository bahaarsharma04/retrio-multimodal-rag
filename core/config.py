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

# Ensure data directories exist
DATA_DIR.mkdir(exist_ok=True)
PROJECTS_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# Model Configurations
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Ingestion Pipeline Defaults
MAX_CHUNK_CHARACTERS = 1500
NEW_AFTER_N_CHARS = 1000
COMBINE_UNDER_N_CHARS = 200

# Retrieval Defaults
HYBRID_SEARCH_TOP_K = 8
RERANK_TOP_N = 4
