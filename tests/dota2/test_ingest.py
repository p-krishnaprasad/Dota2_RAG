import pytest
from dota2.ingest import convert_hero_to_chunks

# Sample hero data — same structure as our JSON files
SAMPLE_HERO = {
    "id": 1,
    "name": "Anti-Mage",
    "internal_name": "npc_dota_hero_antimage",
    "primary_attr": "agi",
    "attack_type": "Melee",
    "roles": ["Carry", "Escape"],
    "base_stats": {
        "move_speed": 310,
        "armor": 1,
        "attack_range": 150,
        "base_str": 21,
        "base_agi": 24,
        "base_int": 12
    },
    "abilities": [
        {
            "name": "antimage_mana_break",
            "display_name": "Mana Break",
            "description": "Burns mana on attack.",
            "lore": "Some lore text.",
            "damage_type": "Physical",
            "pierces_bkb": "No",
            "behavior": "Passive",
            "target_team": "",
            "attribs": [
                {"key": "mana_per_hit", "header": "MANA BURNED:", "value": "40"}
            ]
        }
    ],
    "facets": [
        {"title": "Magebane's Mirror", "description": "Reflects spells."}
    ],
    "aghanims": {
        "has_scepter": True,
        "scepter_desc": "Upgrades Blink.",
        "scepter_skill": "Blink",
        "has_shard": True,
        "shard_desc": "Upgrades Counterspell.",
        "shard_skill": "Counterspell"
    },
    "counters": [
        {"hero_name": "Ogre Magi", "games_played": 120, "our_win_rate": 0.392}
    ]
}

class TestConvertHeroToChunks:
    """Unit tests for convert_hero_to_chunks function."""

    def test_returns_list(self):
        """Should return a list of chunks"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        assert isinstance(chunks, list)

    def test_has_overview_chunk(self):
        """Should always have an overview chunk"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        ids = [c["id"] for c in chunks]
        assert "Anti-Mage_overview" in ids

    def test_overview_contains_hero_name(self):
        """Overview chunk text should mention hero name"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        overview = next(c for c in chunks if c["id"] == "Anti-Mage_overview")
        assert "Anti-Mage" in overview["text"]

    def test_overview_contains_role(self):
        """Overview chunk should mention hero roles"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        overview = next(c for c in chunks if c["id"] == "Anti-Mage_overview")
        assert "Carry" in overview["text"]

    def test_ability_chunk_created(self):
        """Should create one chunk per ability"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        ids = [c["id"] for c in chunks]
        assert "Anti-Mage_antimage_mana_break" in ids

    def test_ability_chunk_contains_description(self):
        """Ability chunk should contain the description"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        ability_chunk = next(
            c for c in chunks if "mana_break" in c["id"]
        )
        assert "Burns mana on attack" in ability_chunk["text"]

    def test_counter_chunk_created(self):
        """Should create a counter chunk when counters exist"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        ids = [c["id"] for c in chunks]
        assert "Anti-Mage_counters" in ids

    def test_counter_chunk_mentions_counter_hero(self):
        """Counter chunk should mention counter hero names"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        counter_chunk = next(c for c in chunks if c["id"] == "Anti-Mage_counters")
        assert "Ogre Magi" in counter_chunk["text"]

    def test_aghanims_chunk_created(self):
        """Should create aghanims chunk when scepter/shard exists"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        ids = [c["id"] for c in chunks]
        assert "Anti-Mage_aghanims" in ids

    def test_facets_chunk_created(self):
        """Should create facets chunk when facets exist"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        ids = [c["id"] for c in chunks]
        assert "Anti-Mage_facets" in ids

    def test_chunk_metadata_has_hero(self):
        """Every chunk should have hero in metadata"""
        chunks = convert_hero_to_chunks(SAMPLE_HERO)
        for chunk in chunks:
            assert chunk["metadata"]["hero"] == "Anti-Mage"

    def test_hero_with_no_counters(self):
        """Should handle hero with no counters gracefully"""
        hero = {**SAMPLE_HERO, "counters": []}
        chunks = convert_hero_to_chunks(hero)
        ids = [c["id"] for c in chunks]
        assert "Anti-Mage_counters" not in ids