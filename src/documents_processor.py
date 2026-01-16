"""
Document processing and chunking module
"""
from pathlib import Path
from typing import List
import logging

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            elif ext in (".txt", ".md"):
                loader = TextLoader(str(path), encoding="utf-8")
            elif ext in (".doc", ".docx"):
                loader = UnstructuredWordDocumentLoader(str(path))
            else:
                raise ValueError(f"Unsupported file format: {ext} ({path})")

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} pages from {path}")
            return documents

        except Exception:
            logger.exception(f"Failed to load document: {path}")
            raise

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def process_files(self, file_path: str) -> List[Document]:
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)

        source = str(Path(file_path))
        for i, chunk in enumerate(chunks):
            chunk.metadata = chunk.metadata or {}
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source_file"] = source

        return chunks

    def process_directory(self, directory_path: str) -> List[Document]:
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        all_chunks: List[Document] = []
        supported_extensions = {".pdf", ".txt", ".docx", ".doc", ".md"}

        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    chunks = self.process_files(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {str(e)}")
                    continue

        logger.info(f"Processed {len(all_chunks)} chunks from directory {directory}")
        return all_chunks
