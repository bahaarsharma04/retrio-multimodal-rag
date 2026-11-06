import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse
from api.routers import chat, projects, documents
from db.session import init_db, is_db_connected
from core.logger import app_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure DB tables exist if PostgreSQL is available
    if is_db_connected():
        init_db()
        app_logger.info("Connected to PostgreSQL database successfully.")
    else:
        app_logger.info("PostgreSQL database not reachable; operating in local JSON fallback mode.")
    yield


app = FastAPI(
    title="Retrio - Multi-Modal Hybrid RAG API",
    description="Production-grade Retrieval-Augmented Generation API with Unstructured layout parsing, BM25 + ChromaDB Hybrid Search, Cohere Reranking, and PostgreSQL conversation memory.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for external frontends / Postman / web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    app_logger.info(f"{request.method} {request.url.path} - Status: {response.status_code} ({elapsed_ms:.1f}ms)")
    return response

# Include API Routers
app.include_router(chat.router)
app.include_router(projects.router)
app.include_router(documents.router)


@app.get("/", tags=["Root"])
def root():
    return {
        "title": "Retrio Multi-Modal RAG API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "status": "online",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    db_ok = is_db_connected()
    return HealthResponse(
        status="healthy",
        database_connected=db_ok,
        version="1.0.0",
    )
