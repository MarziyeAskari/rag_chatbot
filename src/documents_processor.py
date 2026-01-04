"""
Document processing and chunking module
"""
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters  import RecursiveCharacterTextSplitter
import logging
from langchain_community.document_loaders import (
PyPDFLoader,
TextLoader,
UnstructuredWordDocumentLoader,
)

logger = logging.getLogger(__name__)
class DocumentProcessor:
    def __init__(self,chunk_size: int =1000, chunk_overlap: int =200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function = len
        )

    def load_document(self, file_path: str) -> List[Document]:
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_extension == [".txt", ".md"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            elif file_extension == [".doc",".docx"]:
                loader = UnstructuredWordDocumentLoader(str(file_path))
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            documents=loader.load()
            logger.info(f"Loaded {len(documents)} pages from {file_path}")
            return documents
        except Exception as e:
               logger.error(e)
               raise
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into  {len(chunks)} chunks")
        return chunks
    def process_files(self, file_path:str) -> List[Document]:
        documents = self.load_document(file_path)
        chunks = self.chunk_documents(documents)
        for i,chunk in enumerate(chunks):
            chunk.metadata["chunk_id"]=i
            chunk.metadata["source_file"]=str(file_path)

        return chunks
    def process_directory(self, directory_path:str) -> List[Document]:
        directory = Path(directory_path)
        all_chunks = []
        supported_extensions = ["pdf", "txt", "docx","doc","md"]
        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    chunks = self.process_files(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning(f"Fiealed to process {file_path}: {str(e)}")
                    continue
        logger.info(f"Processed {len(all_chunks)} chunks from directory {directory_path}")
        return all_chunks