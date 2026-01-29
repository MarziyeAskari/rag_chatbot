from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Union
import logging
import time

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config_loader import get_setting
from src.documents_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_chain import RagChain
from src.session_manager import SessionManager
from src.db_session_manager import DatabaseSessionManager

logger = logging.getLogger(__name__)
settings = get_setting()

document_processor: Optional[DocumentProcessor] = None
vector_store: Optional[VectorStore] = None
rag_chain: Optional[RagChain] = None
session_manager: Optional[Union[SessionManager, DatabaseSessionManager]] = None


# ----------------------------
# Lifespan (startup / shutdown)
# ----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global document_processor, vector_store, rag_chain, session_manager
    logger.info("Starting RAG API...")

    # Sessions
    if settings.use_database_session:
        session_manager = DatabaseSessionManager(
            database_url=settings.session_database_url,
            database_path=settings.session_database_path,
            session_timeout=settings.session_timeout,
            max_history_per_session=settings.max_history_per_session,
        )
    else:
        session_manager = SessionManager(
            session_timeout=settings.session_timeout,
            max_history_per_session=settings.max_history_per_session,
        )

    # Core pipeline
    document_processor = DocumentProcessor(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    vector_store = VectorStore(
        persist_directory=settings.vector_store_path,
        collection_name=settings.collection_name,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        api_key=settings.openai_api_key if settings.embedding_provider == "openai" else None,
    )

    rag_chain = RagChain(
        vector_store=vector_store,
        llm_provider=settings.llm_provider,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        api_key=settings.openai_api_key if settings.llm_provider == "openai" else None,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
    )

    yield
    logger.info("Shutting down RAG API")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Frontend
# ----------------------------
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def root():
    index = frontend_path / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"status": "ok", "docs": "/docs"}


# ----------------------------
# Schemas
# ----------------------------
class QueryRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    top_k: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[dict]
    session_id: str


class UploadResponse(BaseModel):
    message: str
    total_chunks: int


# ----------------------------
# API
# ----------------------------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag_chain or not session_manager:
        raise HTTPException(503, "System not ready")

    start = time.time()
    session = session_manager.get_or_create_session(req.session_id)
    history = session_manager.get_conversation_history(session.session_id, max_messages=10)

    session_manager.add_message(session.session_id, "user", req.question)

    result = rag_chain.query(
        question=req.question,
        top_k=req.top_k,
        conversation_history=history,
    )

    session_manager.add_message(session.session_id, "assistant", result["answer"])

    logger.info("Query took %.2fs", time.time() - start)

    return QueryResponse(
        answer=result["answer"],
        source_documents=result["source_documents"],
        session_id=session.session_id,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not document_processor or not vector_store:
        raise HTTPException(503, "System not ready")

    path = Path(settings.uploads_path) / file.filename
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        f.write(await file.read())

    chunks = document_processor.process_files(str(path))
    vector_store.add_documents(chunks)

    return UploadResponse(
        message="File processed",
        total_chunks=vector_store.get_collection_size(),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "vector_store_size": vector_store.get_collection_size() if vector_store else 0,
    }

@app.post("/sessions")
async def create_session():
    if not session_manager:
        raise HTTPException(status_code=503, detail="System not ready")
    s = session_manager.create_session()
    return {
        "session_id": s.session_id,
        "created_at": s.created_at,
    }


@app.get("/vectorstore/stats")
async def vectorstore_stats():
    if not vector_store:
        raise HTTPException(status_code=503, detail="System not ready")

    return {
        "collection_name": vector_store.collection_name,
        "total_documents": vector_store.get_collection_size(),
        "vector_store_path": str(vector_store.persist_directory),
    }

