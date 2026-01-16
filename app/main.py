"""
FastAPI application for RAG Chatbot
"""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional, Union
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
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
session_manager: Optional[Union[SessionManager,DatabaseSessionManager]] = None

settings = get_setting()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, session_manager, document_processor, vector_store
    try:
        logger.info(f"Initializing RAG Chatbot components...")
        if settings.use_database_session:
            logger.info("Using database-backed session manager for persistent storage")
            database_url = settings.session_database_url or None
            database_path = settings.session_database_path or None
            session_manager = DatabaseSessionManager(
                database_url=database_url,
                database_path=database_path,
                session_timeout=settings.session_timeout,
                max_history_per_session=settings.max_history_per_session,
            )
        else:
            logger.info("Using in-memory session manager")
            session_manager = SessionManager(
                session_timeout=settings.session_timeout,
                max_history_per_session=settings.max_history_per_session,
            )

        document_processor = DocumentProcessor(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        vector_store = VectorStore(
            persist_directory=settings.vector_store_path,
            collection_name=settings.collection_name,
            embedding_provider=settings.embedding_provider,
            embedding_model=settings.embedding_model,
            api_key=settings.openai_api_key if settings.embedding_provider.lower() == "openai" else None,
        )
        rag_chain = RagChain(
            vector_store=vector_store,
            llm_provider=settings.llm_provider,
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key if settings.llm_provider.lower() == "openai" else None,
            top_k=settings.top_k,
            similarity_threshold=settings.similarity_threshold,
        )
        logger.info(f"RAG Chatbot initialized successfully")
    except Exception as ex:
        logger.error(f"RAG Chatbot initialization failed: {ex}")
        raise
    yield
    logger.info(f"Shutting down RAG Chatbot ...")
    if session_manager:
        cleaned = session_manager.cleanup_expired_sessions()
        count = session_manager.get_session_count() if hasattr(session_manager, "get_session_count") else None
        logger.info(f"Cleaned up {cleaned} sessions. Active sessions: {count}")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
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

frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    logger.info(f"Frontend static files from {frontend_path}")


# Requests
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    source_documents: List[dict]
    session_id: str


class UploadResponse(BaseModel):
    message: str
    chunk_size: int
    total_chunks: int


class HealthResponse(BaseModel):
    status: str
    vector_store_size: int


@app.get("/")
async def root():
    index_path = frontend_path / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Welcome to RAG Chatbot API",
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/api", response_model=dict)
async def api_info():
    return {
        "message": "Welcome to RAG Chatbot API",
        "version": settings.app_version,
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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not rag_chain or not session_manager:
        raise HTTPException(status_code=503, detail="RAG Chain or session not initialized")
    import time
    start_time = time.time()
    try:
        session = session_manager.get_or_create_session(request.session_id)
        session_id = session.session_id
        conversation_history = session_manager.get_conversation_history(
            session_id,
            max_messages=10
        )
        session_manager.add_message(
            session_id, "user", request.question)
        result = rag_chain.query(
            question=request.question,
            conversation_history=conversation_history,
            top_k=request.top_k or settings.top_k,
        )
        session_manager.add_message(session_id, "assistant", result["answer"])
        duration = time.time() - start_time
        logger.info(f"Query handled in {duration:.3f}s (session={session_id})")
        return QueryResponse(
            answer=result["answer"],
            session_id=session_id,
            source_documents=result.get("source_documents",[]),
        )
    except Exception as ex:
        logger.error(f"Error processing query: {str(ex)}")
        raise HTTPException(status_code=500, detail=str(ex))

@app.post("/upload", response_model=UploadResponse)
async def upload_file (file: UploadFile = File(...)):
    if not document_processor or not vector_store:
        raise HTTPException(status_code= 503,detail="some components are not initialized")
    try:
        upload_dir = Path(settings.uploads_path)
        upload_dir.mkdir(parents=True, exist_ok=True)

        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        chunks = document_processor.process_files(str(file_path))
        vector_store.add_documents(chunks)
        total_chunks = vector_store.get_collection_size() if vector_store else 0
        logger.info(f"Upload and Processed {file.filename}: {total_chunks} chunks ")

        return UploadResponse(
            message=f"Successfully processed{file.filename}",
            chunk_size=settings.chunk_size,
            total_chunks=total_chunks,
        )
    except Exception as e:
        logger.error(f"Error processing file : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-directly")
async def upload_directly(directory_path:str):
    if not document_processor or not vector_store:
        raise HTTPException(status_code= 503,detail="some components are not initialized")
    try:
        chunks = document_processor.process_directory(directory_path)
        if chunks:
            vector_store.add_documents(chunks)
        return {"message": f"Processed {directory_path}", "total_chunks": vector_store.get_collection_size()}
    except Exception as e:
        logger.error(f"Error processing directory : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/vectorstore")
async def clear_vector_store():
    if not vector_store:
        raise HTTPException(status_code=503,detail="some components are not initialized")
    try:
        vector_store.delete_collection()
        return {"message": f"Vector store cleared successfully"}
    except Exception as e:
        logger.error(f"Error deleting vector store : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/vectorstore/stats")
async def get_vector_store_stats():
    if not vector_store:
        raise HTTPException(status_code=503,detail="some components are not initialized")
    try:
        size = vector_store.get_collection_size() if vector_store else 0
        return {
            "collection_name": vector_store.collection_name,
            "total_documents": size,
            "vector_store_path":settings.vector_store_path,
        }
    except Exception as e:
        logger.error(f"Error getting vector store stats : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/health")
async def get_monitoring_health():
    return {"status": "ok"}

@app.get("/monitoring/metrics")
async def get_monitoring_metrics():
    return {"status": "ok"}

@app.post("/sessions")
async def create_sessions():
    if not session_manager:
        raise HTTPException(status_code=503, detail="some components are not initialized")
    try:
        session = session_manager.create_session()
        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "message": "session created successfully",
        }
    except Exception as e:
        logger.error(f"Error creating session : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}")
async def get_session_info(session_id: str):
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    try:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        return {
            "session_id": session.session_id,
            "created_at": session.created_at,
            "last_accessed": session.last_accessed,
            "message_count": len(session.messages),
            "messages": [
                {"role": msg.role, "content": msg.content, "timestamp": msg.timestamp}
                for msg in session.messages
            ],
            "metadata": session.metadata,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, max_messages: int):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        history = session_manager.get_conversation_history(session_id ,max_messages = max_messages)
        return [
            {
                "role":role,
                "content": content
            }
            for role, content in history
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"Error getting session history : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_manager:
        raise HTTPException(status_code=503, detail="some components are not initialized")
    try:
        deleted = session_manager.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=500, detail="session was not deleted")
        return {
            "message": f"Session {session_id} was deleted successfully",
        }
    except Exception as e:
        logger.error(f"Error deleting session : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
async def list_sessions():
    if not session_manager:
        raise HTTPException(status_code=503, detail="some components are not initialized")
    try:
        sessions = session_manager.get_all_sessions()
        return{
            "total_sessions": len(sessions),
            "sessions": [
                {
                "session_id": session.session_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed":session.last_accessed.isoformat(),
                "message_count": len(session.messages),
                }
             for session in sessions
            ]
        }
    except Exception as e:
        logger.error(f"Error getting session : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sessions/cleanup")
async def cleanup_sessions():
    if not session_manager:
        raise HTTPException(status_code=503, detail="some components are not initialized")
    try:
        cleaned_sessions = session_manager.cleanup_expired_sessions()
        return{"message": f"Successfully cleaned up {cleaned_sessions} sessions"}
    except Exception as e:
        logger.error(f"Error cleaning up session : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
