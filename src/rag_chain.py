from typing import Optional, List, Tuple, Dict, Any

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import logging

logger = logging.getLogger(__name__)


class RagChain:
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
            self.llm = Ollama(model=model, temperature=temperature)

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
        self.retriever = self.vector_store.vectorstore.as_retriever(
            search_kwargs={"k": self.top_k}
        )

        # ---- Build chains ONCE ----
        self.doc_chain = create_stuff_documents_chain(self.llm, self.prompt)
        self.rag_chain = create_retrieval_chain(self.retriever, self.doc_chain)

    def _format_history(self, conversation_history: Optional[List[Tuple[str, str]]]) -> str:
        if not conversation_history:
            return "None"
        lines = []
        for role, content in conversation_history:
            role_label = "User" if role == "user" else "Assistant"
            lines.append(f"{role_label}: {content}")
        return "\n".join(lines)

    def query(
            self,
            question: str,
            top_k: Optional[int] = None,
            conversation_history: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[str, Any]:
        # Update retriever k if requested
        if top_k is not None and top_k != self.top_k:
            self.top_k = top_k
            self.retriever = self.vector_store.vectorstore.as_retriever(
                search_kwargs={"k": self.top_k}
            )
            self.rag_chain = create_retrieval_chain(self.retriever, self.doc_chain)

        history_text = self._format_history(conversation_history)

        try:
            result = self.rag_chain.invoke(
                {
                    "input": question,
                    "conversation_history": history_text,
                }
            )
            answer = result.get("answer", "")
            source_docs = result.get("context", [])

            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    }
                    for doc in source_docs
                    if isinstance(doc, Document)
                ],
            }

        except Exception:
            logger.exception("Error in RAG query")
            raise

    def get_content(self, question: str, top_k: int = 5) -> List[Document]:
        # Use your VectorStore's similarity_search API
        return self.vector_store.similarity_search(
            query=question,
            k=top_k,
            threshold=self.similarity_threshold,
        )
