import requests
import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Step 1: Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# ── Step 2: Fetch all 4 APIs
print("Fetching data from OpenDota API...")

# All heroes basic info — keyed by hero id ("1", "2" etc)
heroes_raw = requests.get(os.getenv("DOTA2_HEROES_URL")).json()
print(f"✅ Heroes: {len(heroes_raw)}")

# Which abilities each hero has — keyed by internal hero name
hero_abilities_raw = requests.get(os.getenv("DOTA2_HERO_ABILITIES_URL")).json()
print(f"✅ Hero abilities map: {len(hero_abilities_raw)}")

# Full ability details — keyed by ability name
abilities_raw = requests.get(os.getenv("DOTA2_ABILITIES_URL")).json()
print(f"✅ Abilities: {len(abilities_raw)}")

# Aghanims scepter + shard descriptions — this is a LIST not a dict
aghs_raw = requests.get(os.getenv("DOTA2_AGHS_URL")).json()
print(f"✅ Aghanims descriptions: {len(aghs_raw)}")

# ── Step 3: Convert aghs_raw list → dict keyed by hero_id
# aghs_raw is a list like [{"hero_id": 1, ...}, {"hero_id": 2, ...}]
# We convert it to {1: {...}, 2: {...}} so we can look up by hero_id easily
aghs_by_hero_id = {}
for entry in aghs_raw:
    aghs_by_hero_id[entry["hero_id"]] = entry

# ── Step 4: Fetch matchups for every hero
# /heroes/{hero_id}/matchups returns win/loss data vs every other hero
# We use this to calculate who counters each hero
print("\nFetching matchups for all heroes (this takes ~2 mins)...")
matchups_by_hero_id = {}

for hero_id_str, hero in heroes_raw.items():
    hero_id = hero["id"]

    # Fetch matchup data for this hero
    url = f"https://api.opendota.com/api/heroes/{hero_id}/matchups"
    response = requests.get(url)
    matchups = response.json()

    # Calculate win rate for each opponent
    # matchup = {"hero_id": 2, "games_played": 1000, "wins": 600}
    # wins here means OUR hero won against that opponent
    # so low win rate = that opponent beats us = counter
    counters = []
    for matchup in matchups:
        games = matchup.get("games_played", 0)
        wins = matchup.get("wins", 0)
        if games > 100:  # ignore heroes with too few games — unreliable data
            win_rate = wins / games
            counters.append({
                "hero_id": matchup["hero_id"],
                "games_played": games,
                "win_rate_against": round(win_rate, 3)
            })

    # Sort ascending = lowest win rate first = hardest counters at top
    counters.sort(key=lambda x: x["win_rate_against"])

    # Keep top 5 counters only
    matchups_by_hero_id[hero_id] = counters[:5]

    print(f"  ✅ {hero['localized_name']}")

    # Wait 1 second between calls to respect OpenDota rate limits
    time.sleep(1)

print("✅ All matchups fetched!")

# ── Step 5: Build a hero_id → localized_name lookup
# We need this to convert counter hero_ids back to readable names
# e.g. hero_id 2 → "Axe"
id_to_name = {}
for hero_id_str, hero in heroes_raw.items():
    id_to_name[hero["id"]] = hero["localized_name"]

# ── Step 6: Loop through every hero and build + save their JSON file
print("\nBuilding hero files...")
for hero_id_str, hero in heroes_raw.items():

    hero_id = hero["id"]
    internal_name = hero["name"]  # e.g. "npc_dota_hero_antimage"

    # Extract clean hero name for filename
    # "npc_dota_hero_antimage" → "antimage"
    short_name = internal_name.replace("npc_dota_hero_", "")

    # ── Get this hero's abilities
    hero_ability_data = hero_abilities_raw.get(internal_name, {})
    ability_names = hero_ability_data.get("abilities", [])

    # ── Look up full details for each ability
    abilities = []
    for ability_name in ability_names:
        # Skip hidden/generic abilities — not real hero abilities
        if "hidden" in ability_name or "generic" in ability_name:
            continue

        ability_detail = abilities_raw.get(ability_name, {})

        abilities.append({
            "name": ability_name,
            # dname = human readable ability name e.g. "Mana Break"
            "display_name": ability_detail.get("dname", ability_name),
            # desc = plain English description of what the ability does
            "description": ability_detail.get("desc", ""),
            # lore = flavour lore text
            "lore": ability_detail.get("lore", ""),
            # dmg_type = Physical, Magical or Pure
            "damage_type": ability_detail.get("dmg_type", ""),
            # bkbpierce = does it pierce Black King Bar
            "pierces_bkb": ability_detail.get("bkbpierce", ""),
            "behavior": ability_detail.get("behavior", []),
            "target_team": ability_detail.get("target_team", ""),
            "attribs": ability_detail.get("attrib", [])
        })

    # ── Get facets
    facets = []
    for facet in hero_ability_data.get("facets", []):
        facets.append({
            "title": facet.get("title", ""),
            "description": facet.get("description", "")
        })

    # ── Get aghanims scepter + shard info
    aghs = aghs_by_hero_id.get(hero_id, {})
    aghanims = {
        "has_scepter": aghs.get("has_scepter", False),
        "scepter_desc": aghs.get("scepter_desc", ""),
        "scepter_skill": aghs.get("scepter_skill_name", ""),
        "has_shard": aghs.get("has_shard", False),
        "shard_desc": aghs.get("shard_desc", ""),
        "shard_skill": aghs.get("shard_skill_name", "")
    }

    # ── Resolve counter hero_ids to readable names
    raw_counters = matchups_by_hero_id.get(hero_id, [])
    counters = []
    for c in raw_counters:
        counter_name = id_to_name.get(c["hero_id"], "Unknown")
        counters.append({
            "hero_name": counter_name,
            "games_played": c["games_played"],
            # win_rate_against = how often OUR hero wins against this counter
            # lower = harder counter e.g. 0.42 means we only win 42% of the time
            "our_win_rate": c["win_rate_against"]
        })

    # ── Build the final hero object
    hero_data = {
        "id": hero_id,
        "name": hero.get("localized_name", ""),
        "internal_name": internal_name,
        "primary_attr": hero.get("primary_attr", ""),
        "attack_type": hero.get("attack_type", ""),
        "roles": hero.get("roles", []),
        "base_stats": {
            "move_speed": hero.get("move_speed", 0),
            "armor": hero.get("base_armor", 0),
            "attack_range": hero.get("attack_range", 0),
            "base_str": hero.get("base_str", 0),
            "base_agi": hero.get("base_agi", 0),
            "base_int": hero.get("base_int", 0)
        },
        "abilities": abilities,
        "facets": facets,
        "aghanims": aghanims,
        "counters": counters
    }

    # ── Save to data/antimage.json etc
    filepath = f"data/{short_name}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(hero_data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved {len(heroes_raw)} hero files to data/")
print("🎉 Done!")