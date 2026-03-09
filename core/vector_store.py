import chromadb
from core.embedder import OllamaEmbeddingFunction

class VectorStore:
    """
    Generic reusable Chroma vector store wrapper.
    Not tied to any specific domain.
    """
    def __init__(self, collection_name: str, persist_directory: str = "./chroma_db"):
        # PersistentClient saves data to disk
        self.client = chromadb.PersistentClient(path=persist_directory)

        # collection_name makes this reusable
        # dota2 uses "dota2_heroes"
        # medical would use "medical_docs" etc
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=OllamaEmbeddingFunction()
        )

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict]):
        """Store chunks into Chroma — insert or update if exists."""
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def search(self, query: str, n_results: int = 5, where: dict = None) -> list[str]:
        """
        Search for similar chunks.
        where = optional metadata filter
        Falls back to pure vector search if filtered search fails.
        """
        query_params = {
            "query_texts": [query],
            "n_results": n_results
        }
        if where:
            query_params["where"] = where

        try:
            results = self.collection.query(**query_params)
            return results["documents"][0]
        except Exception:
            # Fallback to pure vector search if filter returns nothing
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results["documents"][0]

    def count(self) -> int:
        """Returns total number of chunks stored."""
        return self.collection.count()