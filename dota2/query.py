import json
from core.vector_store import VectorStore
from core.llm import LLMClient

# Use core components — no Chroma or Groq details here
store = VectorStore(collection_name="dota2_heroes")
llm = LLMClient()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1 — Entity Extraction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def extract_entities(question: str) -> dict:
    """
    Asks Groq to extract hero and ability names
    from the user's question.
    Returns: {"hero_name": "Anti-Mage", "ability_name": "Mana Break"}
    """
    # We give Groq a very specific prompt
    # telling it to ONLY return JSON — nothing else
    prompt = f"""You are a Dota 2 expert. 
    Extract any Dota 2 hero name and ability name from the question below.

    Rules:
    - If a hero name is mentioned, return it exactly as it appears in the game
    - If an ability name is mentioned, return it exactly as it appears in the game
    - If nothing is mentioned, return null for that field
    - Return ONLY a JSON object, no explanation, no markdown

    Question: {question}

    Return format:
    {{"hero_name": "hero name or null", "ability_name": "ability name or null"}}"""

    response = llm.ask(prompt)
    
    # response is a string like: '{"hero_name": "Anti-Mage", "ability_name": "Mana Break"}'
    # json.loads() converts that string into a Python dictionary
    try:
        # strip() removes any leading/trailing whitespace
        entities = json.loads(response.strip())
    except json.JSONDecodeError:
        # If Groq returns something unexpected, default to no entities
        entities = {"hero_name": None, "ability_name": None}

    return entities


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2 — Build Chroma Filter
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def build_filter(entities: dict) -> dict | None:
    """
    Builds a Chroma metadata filter based on extracted entities.
    Handles single hero, multiple heroes, and ability names.
    """
    hero = entities.get("hero_name")
    ability = entities.get("ability_name")

    # Clean up "null" strings Groq sometimes returns
    if hero == "null":
        hero = None
    if ability == "null":
        ability = None

    # Handle comma separated string "Axe, Anti-Mage" → ["Axe", "Anti-Mage"]
    if isinstance(hero, str) and "," in hero:
        hero = [h.strip() for h in hero.split(",") if h.strip() and h.strip() != "null"]


    # Normalize hero name — fix common variations
    # e.g. "Anti Mage" → "Anti-Mage", "antimage" → "Anti-Mage"
    if isinstance(hero, str):
        hero = hero.strip()
    
    # If Groq returned multiple heroes as a list
    # e.g. ["Axe", "Anti-Mage"] → use $or filter to match either
    if isinstance(hero, list):
        # Clean each hero name in the list
        hero = [h.strip() for h in hero if h and h != "null"]
        if len(hero) == 0:
            hero = None
        elif len(hero) == 1:
            hero = hero[0]  # single hero — treat normally

    if isinstance(hero, list) and ability:
        # Multiple heroes + ability — search by ability only
        return {"ability": ability}

    elif isinstance(hero, list):
        # Multiple heroes — use $or to match any of them
        return {
            "$or": [{"hero": h} for h in hero]
        }

    elif hero and ability:
        return {
            "$and": [
                {"hero": hero},
                {"ability": ability}
            ]
        }

    elif hero:
        return {"hero": hero}

    elif ability:
        return {"ability": ability}

    else:
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3 — Query Chroma
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def retrieve_chunks(question: str, chroma_filter: dict | None, entities: dict) -> list[str]:
    """
    Queries Chroma with the question and optional metadata filter.
    For multiple heroes, fetches chunks for each hero separately
    then combines them to ensure balanced representation.
    """
    hero = entities.get("hero_name")

    # Handle comma separated string
    if isinstance(hero, str) and "," in hero:
        hero = [h.strip() for h in hero.split(",") if h.strip()]

    # Multiple heroes — fetch separately and combine
    if isinstance(hero, list) and len(hero) > 1:
        all_chunks = []
        # Fetch top 3 chunks per hero separately
        # This guarantees equal representation of each hero
        for hero_name in hero:
            try:
                chunks = store.search(question, n_results=3, where={"hero": hero_name})
                all_chunks.extend(chunks)
            except Exception:
                pass
        return all_chunks if all_chunks else []

    # Single hero or no hero — normal flow
    try:
        # search() handles the query_texts and n_results internally
        # pass chroma_filter as where= if we have one
        return store.search(question, n_results=5, where=chroma_filter)
    except Exception:
        # Fallback — pure vector search with no filter
        return store.search(question, n_results=5)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4 + 5 — Build Prompt + Ask Groq
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def generate_answer(question: str, chunks: list[str]) -> str:
    """
    Sends retrieved chunks + question to Groq.
    Returns Groq's generated answer.
    """
    # Join all chunks into one context string
    # Each chunk separated by a blank line for readability
    context = "\n\n".join(chunks)

    # This is called a "prompt template"
    # We give Groq the context first, then the question
    # "Answer ONLY from context" prevents hallucination
    prompt = f"""You are a helpful Dota 2 expert assistant. 
    Use the following context retrieved from the Dota 2 knowledge base to answer the question accurately.

    Guidelines:
    - Answer ONLY based on the provided context
    - Be concise but complete
    - For abilities: mention what it does, damage type, and key stats
    - For counters: mention win rates to support your answer
    - For aghanims: clearly state what ability it upgrades and what it does
    - If the answer is not in the context, say "I don't have enough information about that"
    - Never make up information not present in the context

    Context:
    {context}

    Question: {question}

    Answer:"""

    response = llm.ask(prompt)
    return response


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN — Full Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def ask(question: str) -> dict:
    """
    Full RAG pipeline:
    question → extract entities → build filter
    → retrieve chunks → generate answer → return
    """
    print(f"\n🔍 Question: {question}")

    # Step 1 — Extract hero/ability names from question
    entities = extract_entities(question)
    print(f"📌 Entities extracted: {entities}")

    # Step 2 — Build Chroma filter from entities
    chroma_filter = build_filter(entities)
    print(f"🔧 Chroma filter: {chroma_filter}")

    # Step 3 — Retrieve relevant chunks from Chroma
    chunks = retrieve_chunks(question, chroma_filter, entities)
    print(f"📚 Retrieved {len(chunks)} chunks")
    # for i, chunk in enumerate(chunks):
    #     print(f"  Chunk {i+1}: {chunk[:150]}")

    # Step 4 + 5 — Generate answer using Groq
    answer = generate_answer(question, chunks)

    return {
        "question": question,
        "entities": entities,
        "chunks_used": len(chunks),
        "answer": answer
    }