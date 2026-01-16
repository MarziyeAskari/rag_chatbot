from typing import Optional, List, Tuple


from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms.ollama import Ollama
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import logging




logger = logging.getLogger(__name__)




class RagChain:
    def __init__(self,
                 vector_store,
                 llm_provider:str ="openai",
                 model:str ="gpt-4o-mini",
                 temperature: float=0.7,
                 max_tokens: int=100,
                 api_key:Optional[str]=None,
                 ):
        self.vector_store = vector_store
        if llm_provider == "openai":
            if not api_key:
                raise ValueError("Please provide your OpenAI API key")
            self.llm=ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key)
        else:
            self.llm=Ollama(model=model,temperature=temperature)

        self.prompt_template = PromptTemplate(
            template="""
            Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make an answer.
            {conversation_history}
            Context: {context}
            Question: {question}
            Answer: 
            """,
            input_variables=["context", "question","conversation_history"])
        self.retriever =vector_store.vectorstore.as_retriever(
            search_kwargs={"k":2})
        # QA chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt_template,
        )
        self.retrieval_qa = create_retrieval_chain(
            retriever=self.retriever,
            combine_docs_chain=document_chain,
        )

    def query(self,
              question:str,
              top_key: int =5,
              conversation_history:Optional[List[Tuple[str,str]]]=None,) -> dict:
        try:
            conversation_context = ""
            if conversation_history:
                history_parts =[]
                for role, content in conversation_history:
                    role_label = "User" if role == "user" else "Assistant"
                    history_parts.append(f"{role_label}: {content}")
                conversation_context = ",".join(history_parts)
                if conversation_context:
                    conversation_context = f"Previous conversation context:\n {conversation_context}\n\n"
            self.retriever=self.vector_store.vectorstore.as_retriever(
                search_kwargs={"k":top_key}
            )
            base_template = """
              Use the following pieces of context to answer the question at the end.
            If you don't know the answer, just say that you don't know, don't try to make an answer.
            """
            if conversation_context:
                template_str = base_template + conversation_context + "context {context}\n\n Question: {question}\n\n Answer:"
            else:
                template_str = base_template + "context {context}\n\n Question: {question}\n\n Answer: "
            prompt_with_history = PromptTemplate(
                template=template_str,
                input_variables=["context", "question"]
            )
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt_with_history,
            )
            self.retrieval_qa = create_retrieval_chain(
                retriever=self.retriever,
                combine_docs_chain=document_chain,
            )
            result = self.retrieval_qa.invoke({"input": question})
            answer = result.get("answer","")
            source_docs = result.get("context",[])
            logger.info(f"Generated answer for question: {question[:50]}")
            return {
                "answer": answer,
                "source_documents": [
                    {
                        "content": doc.page_content if hasattr(doc, "page_content") else str(doc),
                        "metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                    }
                    for doc in source_docs
                ]
            }

        except Exception as e:
            logger.error(f"Error in RAG query{str(e)}")
            raise


    def get_content(self,question: str,top_key:int =2,
                    ) -> List[Document]:
        return self.vector_store.similarity_search(
            question,2,0.7)



