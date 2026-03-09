import pytest
from unittest.mock import MagicMock, patch

class TestLLMClient:
    """
    Unit tests for LLMClient.
    We mock ChatGroq so we don't use real API quota.
    """

    @patch("core.llm.ChatGroq")
    def test_ask_returns_string(self, mock_groq):
        """ask() should return a plain string"""
        from core.llm import LLMClient
        # Simulate Groq returning a response object with .content
        mock_groq.return_value.invoke.return_value.content = "Hello!"
        llm = LLMClient()
        result = llm.ask("Say hello")
        assert isinstance(result, str)
        assert result == "Hello!"

    @patch("core.llm.ChatGroq")
    def test_ask_calls_invoke(self, mock_groq):
        """ask() should call llm.invoke() with the prompt"""
        from core.llm import LLMClient
        mock_groq.return_value.invoke.return_value.content = "response"
        llm = LLMClient()
        llm.ask("test prompt")
        # Verify invoke was called with our prompt
        mock_groq.return_value.invoke.assert_called_once_with("test prompt")

    @patch("core.llm.ChatGroq")
    def test_default_model(self, mock_groq):
        """Default model should be llama-3.3-70b-versatile"""
        from core.llm import LLMClient
        LLMClient()
        # Only check the model name, ignore api_key value
        call_kwargs = mock_groq.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert "api_key" in call_kwargs

    @patch("core.llm.ChatGroq")
    def test_empty_prompt(self, mock_groq):
        """ask() should handle empty prompt without crashing"""
        from core.llm import LLMClient
        mock_groq.return_value.invoke.return_value.content = ""
        llm = LLMClient()
        result = llm.ask("")
        assert result == ""