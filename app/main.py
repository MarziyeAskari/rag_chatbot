import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List, Union
import logging
import time
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from src.config_loader import get_setting
from src.documents_processor import DocumentProcessor
from src.queue import SQSClient
from src.vector_store import VectorStore
from src.rag_chain import RagChain
from src.session_manager import SessionManager
from src.db_session_manager import DatabaseSessionManager
from src.upload_storage import S3UploadStorage, LocalUploadStorage, UploadStorage

logger = logging.getLogger(__name__)
settings = get_setting()

document_processor: Optional[DocumentProcessor] = None
vector_store: Optional[VectorStore] = None
rag_chain: Optional[RagChain] = None
session_manager: Optional[Union[SessionManager, DatabaseSessionManager]] = None
upload_storage: Optional[UploadStorage] = None
sqs_client: SQSClient | None = None  # init at startup


# ----------------------------
# Lifespan (startup / shutdown)
# ----------------------------
import asyncio

ready_event = asyncio.Event()   # becomes set when init finished
init_error: Exception | None = None

async def _init_pipeline():
    global document_processor, vector_store, rag_chain, session_manager, upload_storage, sqs_client, init_error
    try:
        logger.info("Initializing pipeline...")

        # SQS
        if settings.upload_async:
            sqs_client = SQSClient(settings.sqs_queue_url, settings.aws_region)

        # Upload storage
        if settings.upload_storage == "s3":
            upload_storage = S3UploadStorage(
                bucket=settings.s3_bucket_name,
                prefix=settings.s3_prefix,
                region=settings.aws_region,
            )
        else:
            upload_storage = LocalUploadStorage(base_dir=str(Path(settings.uploads_path)))

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

        # Document processing
        document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        # Vector store (can block if DB not reachable)
        vector_store = VectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.collection_name,
            embedding_model=settings.embedding_model,
            embedding_provider=settings.embedding_provider,
            api_key=settings.openai_api_key if settings.embedding_provider == "openai" else None,
            vector_store_type=settings.vector_store_type,
            db_url=settings.vector_store_db_url,
        )

        # RAG chain
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

        logger.info("Pipeline ready ✅")
        ready_event.set()

    except Exception as e:
        init_error = e
        logger.exception("Pipeline init failed ❌")
        # do NOT raise; keep API up for /health so ALB doesn't kill it

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG API...")
    # Start init in background; do not block startup
    asyncio.create_task(_init_pipeline())
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

def _require_ready():
    if init_error:
        raise HTTPException(500, f"Init failed: {type(init_error).__name__}")
    if not ready_event.is_set():
        raise HTTPException(503, "System not ready")
# ----------------------------
# API
# ----------------------------
@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if not rag_chain or not session_manager:
        raise HTTPException(503, "System not ready")
    _require_ready()
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

def _safe_name(name:str) -> str:
    safe="".join(c for c in (name or "file") if c.isalnum() or c in "._-")
    return safe[:80] or "file"

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    _require_ready()
    if not document_processor or not vector_store or not upload_storage:
        raise HTTPException(503, "System not ready")

    job_id = str(uuid.uuid4())
    tmp_path = None

    try:
        # stream to temp file (avoids RAM spikes)
        safe_name = _safe_name(file.filename)
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, dir="/tmp", suffix=f"_{safe_name}") as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)

        # store original in local or S3
        def _save():
            with open(tmp_path, "rb") as f:
                return upload_storage.save_bytes(file.filename, f.read())

        saved = await run_in_threadpool(_save)

        # --- AWS async mode ---
        if settings.upload_async:
            if not sqs_client:
                raise HTTPException(503, "SQS not configured")

            if not getattr(saved, "bucket", "") or not getattr(saved, "key", ""):
                raise HTTPException(500, "S3 storage must return bucket/key for worker")

            payload = {"job_id": job_id, "bucket": saved.bucket, "key": saved.key, "filename": file.filename}
            await run_in_threadpool(sqs_client.send, payload)

            # return immediately
            return {"status": "accepted", "job_id": job_id, "uri": saved.uri}

        # --- Local sync mode ---
        chunks = await run_in_threadpool(document_processor.process_files, tmp_path)
        await run_in_threadpool(vector_store.add_documents, chunks)
        total_chunks = vector_store.get_collection_size()

        return {"status": "done", "job_id": job_id, "uri": saved.uri, "total_chunks": total_chunks}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(500, f"Upload failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": ready_event.is_set(),
        "init_error": None if not init_error else type(init_error).__name__,
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

