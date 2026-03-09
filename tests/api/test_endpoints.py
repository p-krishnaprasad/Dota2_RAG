import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

# TestClient simulates HTTP requests to FastAPI
# without actually starting a real server
from main import app
client = TestClient(app)

class TestHealthEndpoint:
    """Tests for GET /health"""

    def test_health_returns_200(self):
        """Health endpoint should return 200"""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_correct_body(self):
        """Health endpoint should return correct JSON"""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "Dota 2 RAG API"


class TestAskEndpoint:
    """Tests for POST /ask"""

    @patch("dota2.query.extract_entities")
    @patch("dota2.query.retrieve_chunks")
    @patch("dota2.query.generate_answer")
    def test_ask_returns_200(self, mock_answer, mock_chunks, mock_entities):
        """Should return 200 for valid question"""
        # Mock all pipeline steps so test runs instantly
        mock_entities.return_value = {"hero_name": "Anti-Mage", "ability_name": None}
        mock_chunks.return_value = ["some chunk text"]
        mock_answer.return_value = "Anti-Mage is a carry hero."
        response = client.post("/ask", json={"question": "Tell me about Anti-Mage"})
        assert response.status_code == 200

    @patch("dota2.query.extract_entities")
    @patch("dota2.query.retrieve_chunks")
    @patch("dota2.query.generate_answer")
    def test_ask_returns_answer(self, mock_answer, mock_chunks, mock_entities):
        """Should return answer in response body"""
        mock_entities.return_value = {"hero_name": None, "ability_name": "Mana Break"}
        mock_chunks.return_value = ["Mana Break burns mana"]
        mock_answer.return_value = "Mana Break burns mana on attack."
        response = client.post("/ask", json={"question": "What does Mana Break do?"})
        data = response.json()
        assert data["answer"] == "Mana Break burns mana on attack."
        assert data["question"] == "What does Mana Break do?"

    def test_empty_question_returns_400(self):
        """Empty question should return 400"""
        response = client.post("/ask", json={"question": ""})
        assert response.status_code == 400
        assert response.json()["detail"] == "Question cannot be empty"

    def test_missing_question_field_returns_422(self):
        """Missing question field should return 422 validation error"""
        response = client.post("/ask", json={})
        assert response.status_code == 422

    def test_whitespace_question_returns_400(self):
        """Whitespace only question should return 400"""
        response = client.post("/ask", json={"question": "   "})
        assert response.status_code == 400

    @patch("dota2.query.extract_entities")
    @patch("dota2.query.retrieve_chunks")
    @patch("dota2.query.generate_answer")
    def test_response_has_all_fields(self, mock_answer, mock_chunks, mock_entities):
        """Response should have all required fields"""
        mock_entities.return_value = {"hero_name": "Axe", "ability_name": None}
        mock_chunks.return_value = ["chunk 1", "chunk 2"]
        mock_answer.return_value = "Axe is a strength hero."
        response = client.post("/ask", json={"question": "Tell me about Axe"})
        data = response.json()
        assert "question" in data
        assert "answer" in data
        assert "entities" in data
        assert "chunks_used" in data

    @patch("main.ask")
    def test_internal_error_returns_500(self, mock_ask):
        """Internal error should return 500"""
        mock_ask.side_effect = Exception("something went wrong")
        response = client.post("/ask", json={"question": "What does Axe do?"})
        assert response.status_code == 500