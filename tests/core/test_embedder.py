import pytest
from unittest.mock import MagicMock, patch

class TestOllamaEmbeddingFunction:
    """
    Unit tests for OllamaEmbeddingFunction.
    We mock OllamaEmbeddings so we don't need Ollama running.
    """

    @patch("core.embedder.OllamaEmbeddings")
    def test_name_returns_model_name(self, mock_ollama):
        """name() should return the model name"""
        from core.embedder import OllamaEmbeddingFunction
        # mock_ollama.return_value.model simulates the model attribute
        mock_ollama.return_value.model = "nomic-embed-text"
        embedder = OllamaEmbeddingFunction()
        assert embedder.name() == "nomic-embed-text"

    @patch("core.embedder.OllamaEmbeddings")
    def test_call_returns_embeddings(self, mock_ollama):
        """__call__() should return a list of vectors"""
        from core.embedder import OllamaEmbeddingFunction
        # Simulate Ollama returning a list of vectors
        mock_ollama.return_value.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]
        embedder = OllamaEmbeddingFunction()
        result = embedder(["text one", "text two"])
        assert len(result) == 2
        assert len(result[0]) == 3

    @patch("core.embedder.OllamaEmbeddings")
    def test_default_model_is_nomic(self, mock_ollama):
        """Default model should be nomic-embed-text"""
        from core.embedder import OllamaEmbeddingFunction
        OllamaEmbeddingFunction()
        # Verify OllamaEmbeddings was called with correct model
        mock_ollama.assert_called_once_with(model="nomic-embed-text")

    @patch("core.embedder.OllamaEmbeddings")
    def test_custom_model(self, mock_ollama):
        """Should accept a custom model name"""
        from core.embedder import OllamaEmbeddingFunction
        mock_ollama.return_value.model = "mxbai-embed-large"
        embedder = OllamaEmbeddingFunction(model="mxbai-embed-large")
        mock_ollama.assert_called_once_with(model="mxbai-embed-large")