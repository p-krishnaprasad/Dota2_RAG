import os

from chromadb import EmbeddingFunction, Embeddings
from langchain_ollama import OllamaEmbeddings

class OllamaEmbeddingFunction(EmbeddingFunction):
    """
    Generic reusable Ollama embedding wrapper.
    Works with any Ollama model — not Dota 2 specific.
    """
    def __init__(self, model: str = None):
        # Use passed model → fallback to env var → fallback to default
        resolved_model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
        self.embedder = OllamaEmbeddings(model=resolved_model)

    def name(self) -> str:
        return self.embedder.model

    def __call__(self, input: list[str]) -> Embeddings:
        return self.embedder.embed_documents(input)