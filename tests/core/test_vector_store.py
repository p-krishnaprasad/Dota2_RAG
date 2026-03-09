import pytest
from unittest.mock import MagicMock, patch

class TestVectorStore:
    """
    Unit tests for VectorStore.
    We mock chromadb so we don't need a real Chroma instance.
    """

    @patch("core.vector_store.OllamaEmbeddingFunction")
    @patch("core.vector_store.chromadb.PersistentClient")
    def test_upsert_calls_collection(self, mock_client, mock_embedder):
        """upsert() should call collection.upsert with correct params"""
        from core.vector_store import VectorStore
        store = VectorStore(collection_name="test")
        store.upsert(
            ids=["id1"],
            documents=["text1"],
            metadatas=[{"hero": "Anti-Mage"}]
        )
        store.collection.upsert.assert_called_once_with(
            ids=["id1"],
            documents=["text1"],
            metadatas=[{"hero": "Anti-Mage"}]
        )

    @patch("core.vector_store.OllamaEmbeddingFunction")
    @patch("core.vector_store.chromadb.PersistentClient")
    def test_search_returns_documents(self, mock_client, mock_embedder):
        """search() should return list of document strings"""
        from core.vector_store import VectorStore
        store = VectorStore(collection_name="test")
        # Simulate Chroma returning search results
        store.collection.query.return_value = {
            "documents": [["chunk 1", "chunk 2"]]
        }
        results = store.search("test query", n_results=2)
        assert results == ["chunk 1", "chunk 2"]

    @patch("core.vector_store.OllamaEmbeddingFunction")
    @patch("core.vector_store.chromadb.PersistentClient")
    def test_search_with_filter(self, mock_client, mock_embedder):
        """search() should pass where filter to Chroma"""
        from core.vector_store import VectorStore
        store = VectorStore(collection_name="test")
        store.collection.query.return_value = {
            "documents": [["chunk 1"]]
        }
        store.search("query", where={"hero": "Axe"})
        # Verify where filter was passed
        call_kwargs = store.collection.query.call_args[1]
        assert call_kwargs["where"] == {"hero": "Axe"}

    @patch("core.vector_store.OllamaEmbeddingFunction")
    @patch("core.vector_store.chromadb.PersistentClient")
    def test_search_fallback_on_exception(self, mock_client, mock_embedder):
        """search() should fall back to no filter if exception occurs"""
        from core.vector_store import VectorStore
        store = VectorStore(collection_name="test")
        # First call raises exception, second call succeeds
        store.collection.query.side_effect = [
            Exception("filter error"),
            {"documents": [["fallback chunk"]]}
        ]
        results = store.search("query", where={"hero": "Unknown"})
        assert results == ["fallback chunk"]

    @patch("core.vector_store.OllamaEmbeddingFunction")
    @patch("core.vector_store.chromadb.PersistentClient")
    def test_count(self, mock_client, mock_embedder):
        """count() should return number of stored chunks"""
        from core.vector_store import VectorStore
        store = VectorStore(collection_name="test")
        store.collection.count.return_value = 1270
        assert store.count() == 1270