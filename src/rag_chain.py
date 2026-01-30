
from __future__ import annotations

from typing import Optional, List, Tuple, Dict, Any

import logging


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_openai import ChatOpenAI
from langchain_community.llms.ollama import Ollama

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

logger = logging.getLogger(__name__)


class RagChain:
    """
    Thread-safe RAG chain wrapper.

    Key design:
    - Build the document-combine chain once in __init__ (safe, reusable).
    - For each request, build a NEW retriever + retrieval chain locally in query()
      so concurrent requests can't overwrite shared state (top_k, retriever, rag_chain).
    """

    def __init__(
        self,
        vector_store,
        llm_provider: str = "openai",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        api_key: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # ---- LLM ----
        if llm_provider.lower() == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            self.llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        else:
            # Ollama typically does not need api_key
            self.llm = Ollama(model=model, temperature=temperature)

        # ---- Prompt ----
        # NOTE: create_retrieval_chain usually expects keys:
        # - input: question
        # - context: retrieved docs
        # So prompt includes {context} and {input}.
        self.prompt = PromptTemplate(
            template=(
                "You are a helpful assistant.\n"
                "Use the provided CONTEXT to answer the USER QUESTION.\n"
                "If you don't know, say you don't know.\n\n"
                "CONVERSATION HISTORY:\n{conversation_history}\n\n"
                "CONTEXT:\n{context}\n\n"
                "USER QUESTION:\n{input}\n\n"
                "ANSWER:"
            ),
            input_variables=["conversation_history", "context", "input"],
        )

        # Build the document combine chain ONCE (safe, shared)
        self.doc_chain = create_stuff_documents_chain(self.llm, self.prompt)

    def _format_history(self, conversation_history: Optional[List[Tuple[str, str]]]) -> str:
        if not conversation_history:
            return "None"
        lines: List[str] = []
        for role, content in conversation_history:
            role_label = "User" if role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def _build_retriever(self, k: int):
        # If you later want to apply similarity_threshold at retrieval time,
        # you can switch to a retriever/search type that supports score filtering.
        return self.vector_store.vectorstore.as_retriever(search_kwargs={"k": k})

    def _extract_sources(self, source_docs: Any) -> List[Dict[str, Any]]:
        """
        Robustly convert whatever the chain returns into your stable output schema:
        [{"content": "...", "metadata": {...}}, ...]
        """
        if not source_docs:
            return []

        out: List[Dict[str, Any]] = []
        for d in source_docs:
            page_content = getattr(d, "page_content", None)
            metadata = getattr(d, "metadata", None)

            if page_content is None:
                # skip unknown objects
                continue

            out.append(
                {
                    "content": page_content,
                    "metadata": metadata or {},
                }
            )
        return out

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        conversation_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Run a single RAG query.

        Returns:
          {
            "answer": str,
            "source_documents": [{"content": str, "metadata": dict}, ...]
          }
        """
        k = top_k if top_k is not None else self.top_k
        history_text = self._format_history(conversation_history)

        try:
            # Build retriever + retrieval chain LOCALLY (thread-safe)
            retriever = self._build_retriever(k)
            rag_chain = create_retrieval_chain(retriever, self.doc_chain)

            result = rag_chain.invoke(
                {
                    "input": question,
                    "conversation_history": history_text,
                }
            )

            # Depending on langchain version, answer key can vary.
            # Most modern versions: "answer"
            # Some variants: "output"
            answer = result.get("answer")
            if answer is None:
                answer = result.get("output", "")

            source_docs = result.get("context", []) or result.get("source_documents", []) or []
            sources = self._extract_sources(source_docs)

            logger.info("Generated answer for question preview: %s", question[:80])
            return {"answer": answer, "source_documents": sources}

        except Exception:
            logger.exception("Error in RAG query")
            raise

    def get_content(self, question: str, top_k: int = 5) -> List[Document]:
        """
        Convenience method for fetching relevant documents without generating an answer.
        Uses your VectorStore's similarity_search API.
        """
        return self.vector_store.similarity_search(
            query=question,
            k=top_k,
            threshold=self.similarity_threshold,
        )
