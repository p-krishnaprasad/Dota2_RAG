import pytest
from unittest.mock import MagicMock, patch

class TestExtractEntities:
    """Unit tests for extract_entities function."""

    @patch("dota2.query.llm")
    def test_extracts_hero_name(self, mock_llm):
        """Should extract hero name from question"""
        from dota2.query import extract_entities
        mock_llm.ask.return_value = '{"hero_name": "Anti-Mage", "ability_name": null}'
        result = extract_entities("Who counters Anti-Mage?")
        assert result["hero_name"] == "Anti-Mage"

    @patch("dota2.query.llm")
    def test_extracts_ability_name(self, mock_llm):
        """Should extract ability name from question"""
        from dota2.query import extract_entities
        mock_llm.ask.return_value = '{"hero_name": null, "ability_name": "Mana Break"}'
        result = extract_entities("What does Mana Break do?")
        assert result["ability_name"] == "Mana Break"

    @patch("dota2.query.llm")
    def test_returns_null_when_nothing_found(self, mock_llm):
        """Should return null for both fields when nothing found"""
        from dota2.query import extract_entities
        mock_llm.ask.return_value = '{"hero_name": null, "ability_name": null}'
        result = extract_entities("What is the capital of France?")
        assert result["hero_name"] is None
        assert result["ability_name"] is None

    @patch("dota2.query.llm")
    def test_handles_invalid_json(self, mock_llm):
        """Should handle invalid JSON from Groq gracefully"""
        from dota2.query import extract_entities
        mock_llm.ask.return_value = "not valid json"
        result = extract_entities("some question")
        assert result == {"hero_name": None, "ability_name": None}


class TestBuildFilter:
    """Unit tests for build_filter function."""

    def test_hero_and_ability(self):
        """Should build AND filter when both hero and ability found"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": "Anti-Mage", "ability_name": "Mana Break"})
        assert result == {"$and": [{"hero": "Anti-Mage"}, {"ability": "Mana Break"}]}

    def test_hero_only(self):
        """Should build simple filter when only hero found"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": "Anti-Mage", "ability_name": None})
        assert result == {"hero": "Anti-Mage"}

    def test_ability_only(self):
        """Should build simple filter when only ability found"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": None, "ability_name": "Mana Break"})
        assert result == {"ability": "Mana Break"}

    def test_nothing_found_returns_none(self):
        """Should return None when nothing found"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": None, "ability_name": None})
        assert result is None

    def test_null_string_treated_as_none(self):
        """Should treat 'null' string as None"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": "null", "ability_name": "null"})
        assert result is None

    def test_multiple_heroes_comma_separated(self):
        """Should handle comma separated hero names"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": "Axe, Anti-Mage", "ability_name": None})
        assert result == {"$or": [{"hero": "Axe"}, {"hero": "Anti-Mage"}]}

    def test_multiple_heroes_as_list(self):
        """Should handle list of hero names"""
        from dota2.query import build_filter
        result = build_filter({"hero_name": ["Axe", "Anti-Mage"], "ability_name": None})
        assert result == {"$or": [{"hero": "Axe"}, {"hero": "Anti-Mage"}]}