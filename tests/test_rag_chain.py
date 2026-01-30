from src.rag_chain import RagChain
from src.vector_store import VectorStore

def test_rag_init():
    store = VectorStore(persist_directory="./data/test", collection_name="test")
    rag = RagChain(vector_store=store, llm_provider="ollama", model="llama2")
    assert rag is not None
