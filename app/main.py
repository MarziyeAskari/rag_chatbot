"""
FastAPI application for RAG Chatbot
"""
from contextlib import asynccontextmanager
from csv import excel
from pathlib import Path
from typing import List

from black.lines import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from openai import vector_stores
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from src.config_loader import get_setting
import logging

from src.db_session_manager import DatabaseSessionManager
from src.documents_processor import DocumentProcessor
from src.rag_chain import RagChain
from src.session_manager import SessionManager
from src.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
document_processor: Optional[DocumentProcessor] = None
vector_store: Optional[VectorStore] = None
rag_chain: Optional[RagChain] = None
session_manager: Optional[SessionManager] = None

setting = get_setting()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, session_manager, document_processor, vector_store, rag_chain
    try:
        logger.info(f"Initializing RAG Chatbot components...")
        if setting.use_database_session:
            logger.info("Using database-baked session manager for persistent storage")
            session_manager = DatabaseSessionManager(
                database_url=setting.database_url if setting.session_database_url else None,
                database_path=setting.database_path if setting.session_database_url else None,
                session_timeout=setting.session_timeout,
                max_history_per_session=setting.max_history_per_session,
            )
        else:
            logger.info("Using in-memory session manager")
            session_manager = SessionManager(
                session_timeout=setting.session_timeout,
                max_history_per_session=setting.max_history_per_session,
            )

        document_processor = DocumentProcessor(
            chunk_size=setting.document_chunk_size,
            chunk_overlap=setting.document_chunk_overlap,
        )
        vector_store = VectorStore(
            persist_directory=setting.vector_store_path,
            collection_name=setting.collection_name,
            embedding_provider=setting.embedding_provider,
            embedding_model=setting.embedding_model,
            api_key=setting.api_key if setting.embedding_provider == "openai" else None,
        )
        rag_chain = RagChain(
            vector_store=vector_store,
            llm_provider=setting.llm_provider,
            model=setting.rag_model,
            max_tokens=setting.max_tokens,
            temperature=setting.temperature,
            api_key=setting.api_key if setting.llm_provider == "openai" else None,
        )
        logger.info(f"RAG Chatbot initialized successfully")
    except Exception as ex:
        logger.error(f"RAG Chatbot initialization failed: {ex}")
        raise
    yield
    logger.info(f"Shutting down RAG Chatbot ...")
    if session_manager:
        session_manager.cleanup_expired_sessions()
        logger.info(f"Cleaned up sessions. Active sessions{session_manager.get_session_count()}")


app = FastAPI(
    title=setting.app_name,
    version=setting.version,
    description="A production RAG (Retrieval-Augmented Generation) Chatbot API",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fronted_path = Path(__file__).parent.parent / "fronted"
if fronted_path.exists():
    app.mount("/static", StaticFiles(directory=fronted_path), name="static")
    logger.info(f"Fronted static files from {fronted_path}")


# Requests
class QueryRequest(BaseModel):
    Question: str
    top_k: Optional[int] = 3
    session_id: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[dict]
    session_id: int


class UploadResponse(BaseModel):
    message: str
    chunk_size: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    vector_store_size: int


@app.get("/")
async def root():
    index_path = fronted_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Welcome to RAG Chatbot API",
        "version": setting.app_version,
        "docs": "/docs",
    }


@app.get("/api", response_model=dict)
async def api_info():
    return {
        "message": "Welcome to RAG Chatbot API",
        "version": setting.app_version,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        size = vector_store.get_collection_size() if vector_store else 0
        return HealthResponse(
            status="OK",
            vector_store_size=size,
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=List[QueryResponse])
async def query(request: QueryRequest):
    if not rag_chain or not session_manager:
        raise HTTPException(status_code=404, detail="RAG Chain or session not initialized")
    import time
    start_time = time.time()
    try:
        session = session_manager.get_session(request.session_id)
        session_id = session.session_id
        conversation_history = session_manager.get_conversation_history(
            session_id,
            max_message=10
        )
        session_manager.add_message(
            session_id, "user", request.question)
        result = rag_chain.query(
            question=request.question,
            conversation_history=conversation_history,
            top_key=request.top_k or setting.top_k,
        )
        session_manager.add_message(session_id, "assistant", result["answer"])
        duration = time.time() - start_time
    except Exception as ex:
        duration = time.time() - start_time
        logger.error(f"Error processing query : {str(ex)}")
