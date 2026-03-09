import os
import json
from core.vector_store import VectorStore

# Use core VectorStore 
store = VectorStore(collection_name="dota2_heroes")


def convert_hero_to_chunks(hero: dict) -> list[dict]:
    """
    Takes a hero dictionary and returns a list of chunks.
    Each chunk has:
      - text: the plain text string to embed
      - id: unique identifier for this chunk
      - metadata: extra info stored alongside (not embedded)
    """
    chunks = []
    hero_name = hero["name"]  # e.g. "Anti-Mage"

    # ── Chunk 1: Hero overview
    # We build a plain English sentence describing the hero
    roles = ", ".join(hero.get("roles", []))
    stats = hero.get("base_stats", {})
    overview_text = (
        f"{hero_name} is a {hero['attack_type']} {hero['primary_attr'].upper()} hero. "
        f"Roles: {roles}. "
        f"Move speed: {stats.get('move_speed')}. "
        f"Attack range: {stats.get('attack_range')}. "
        f"Base armor: {stats.get('armor')}. "
        f"Base STR: {stats.get('base_str')}, "
        f"AGI: {stats.get('base_agi')}, "
        f"INT: {stats.get('base_int')}."
    )
    chunks.append({
        # id must be unique across ALL chunks in the collection
        "id": f"{hero_name}_overview",
        "text": overview_text,
        # metadata is stored but not embedded — useful for filtering later
        "metadata": {"hero": hero_name, "type": "overview"}
    })

    # ── Chunk per ability
    for ability in hero.get("abilities", []):
        display_name = ability.get("display_name", ability["name"])
        description = ability.get("description", "")
        lore = ability.get("lore", "")
        damage_type = ability.get("damage_type", "")
        behavior = ability.get("behavior", "")
        pierces_bkb = ability.get("pierces_bkb", "")

        # Build attribs string — join all stat headers + values
        # e.g. "MANA BURNED PER HIT: 25/30/35/40"
        attrib_parts = []
        for attrib in ability.get("attribs", []):
            # Skip generated internal values — not useful for RAG
            if attrib.get("generated"):
                continue
            header = attrib.get("header", "")
            value = attrib.get("value", "")
            # value can be a list (per level) or a single string
            if isinstance(value, list):
                value = "/".join(value)  # "25/30/35/40"
            attrib_parts.append(f"{header} {value}")

        attribs_str = ". ".join(attrib_parts)

        ability_text = (
            f"{hero_name} has an ability called {display_name}. "
            f"What {display_name} does: {description} "
            f"Behavior: {behavior}. "
            f"Damage type: {damage_type}. "
            f"Pierces BKB: {pierces_bkb}. "
            f"{attribs_str}. "
            f"Lore: {lore}"
        )

        chunks.append({
            "id": f"{hero_name}_{ability['name']}",
            "text": ability_text,
            "metadata": {"hero": hero_name, "type": "ability", "ability": display_name}
        })

    # ── Chunk: Counters
    if hero.get("counters"):
        counter_parts = []
        for c in hero["counters"]:
            counter_win_pct = round((1 - c["our_win_rate"]) * 100, 1)
            counter_parts.append(
                f"{c['hero_name']} (wins {counter_win_pct}% of games against {hero_name})"
            )
        
        counters_text = (
            f"Heroes that counter {hero_name} and are strong against {hero_name}: "
            f"{', '.join(counter_parts)}. "
            f"If you are playing against {hero_name}, consider picking these heroes."
        )
        
        chunks.append({
            "id": f"{hero_name}_counters",
            "text": counters_text,
            "metadata": {"hero": hero_name, "type": "counters"}
        })

    # ── Chunk: Aghanims
    aghs = hero.get("aghanims", {})
    if aghs.get("scepter_desc") or aghs.get("shard_desc"):
        aghs_text = f"{hero_name} Aghanims upgrades. "
        if aghs.get("has_scepter") and aghs.get("scepter_desc"):
            aghs_text += (
                f"Scepter upgrades {aghs['scepter_skill']}: "
                f"{aghs['scepter_desc']} "
            )
        if aghs.get("has_shard") and aghs.get("shard_desc"):
            aghs_text += (
                f"Shard upgrades {aghs['shard_skill']}: "
                f"{aghs['shard_desc']}"
            )
        chunks.append({
            "id": f"{hero_name}_aghanims",
            "text": aghs_text,
            "metadata": {"hero": hero_name, "type": "aghanims"}
        })

    # ── Chunk: Facets
    if hero.get("facets"):
        facet_parts = []
        for facet in hero["facets"]:
            facet_parts.append(
                f"{facet['title']}: {facet['description']}"
            )
        facets_text = (
            f"{hero_name} facets (gameplay variants): "
            f"{'. '.join(facet_parts)}"
        )
        chunks.append({
            "id": f"{hero_name}_facets",
            "text": facets_text,
            "metadata": {"hero": hero_name, "type": "facets"}
        })

    return chunks


def ingest_all_heroes():
    """
    Reads all hero JSON files from data/ folder,
    converts them to chunks, and stores in Chroma.
    """
    data_folder = "./data"
    hero_files = [f for f in os.listdir(data_folder) if f.endswith(".json")]
    print(f"Found {len(hero_files)} hero files")

    all_ids = []
    all_texts = []
    all_metadatas = []

    # ── Loop through every hero file
    for filename in hero_files:
        filepath = os.path.join(data_folder, filename)

        # Open and parse the JSON file
        with open(filepath, "r", encoding="utf-8") as f:
            hero = json.load(f)

        # Convert hero dict → list of chunks
        chunks = convert_hero_to_chunks(hero)

        # Collect all chunks across all heroes
        for chunk in chunks:
            all_ids.append(chunk["id"])
            all_texts.append(chunk["text"])
            all_metadatas.append(chunk["metadata"])

    print(f"Total chunks to embed: {len(all_texts)}")
    print("Embedding and storing in Chroma (this may take a few minutes)...")

    # ── Store everything in Chroma in one batch
    # upsert = insert if new, update if id already exists
    # Chroma automatically calls OllamaEmbeddingFunction on all_texts

    store.upsert(
        ids=all_ids,
        documents=all_texts,
        metadatas=all_metadatas
    )

    print(f"✅ Successfully ingested {len(all_texts)} chunks into Chroma!")


# ── Run ingestion when this file is executed directly
if __name__ == "__main__":
    ingest_all_heroes()