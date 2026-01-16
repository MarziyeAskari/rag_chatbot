from pathlib import Path
from typing import Optional, List, Any
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self,
                 persist_directory: str = "./data/vectorstore",
                 collection_name: str = "rag_documents",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_provider: str = "sentence-transformers",
                 api_key: Optional[str] = None, ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        if embedding_provider == "openai" and api_key:
            self.embeddings = OpenAIEmbeddings(api_key=api_key)
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self.embeddings = HuggingFaceEmbeddings(
                embedding_model=embedding_model,
                model_kwargs={"device": "cpu"}
            )
        self.vectorstore = None
        self._load_or_create_vectorstore()

    def _load_or_create_vectorstore(self):
        try:
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            logger.info(f"Loaded vector store from {self.persist_directory}")
        except Exception as e:
            logger.warning(f"Could not load existing vector store: {str(e)}")
            self.vectorstore = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )
            logger.info(f"Created new vector store ")

    def add_documents(self, documents: List[Document]):
        try:
            ids = self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            logger.info(f"Added {len(documents)} documents to vectorstore")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.7):
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Filter by similarity score
        filtered = [
            doc for doc, score in results
            if score <= threshold  # lower score = more similar in Chroma
        ]

        return filtered

    def similarity_search_with_score(self, query: str, k: int = 2) -> List[tuple]:
        try:
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            logger.info(f"Found {len(results)} similar documents with score for query: {query}")
            return results
        except Exception as e:
            logger.error(f"Error searching similar documents whit score: {str(e)}")
            raise

    def delete_collection(self):
        try:
            self.vectorstore.delete_collection()
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {str(e)}")
            raise

    def get_collection_size(self) -> int:
        try:
            collection = self.vectorstore._collection
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection size: {str(e)}")
            raise
