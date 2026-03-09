# Dota 2 RAG API

A production-ready Retrieval-Augmented Generation (RAG) API for answering questions about Dota 2 heroes and abilities. Built with FastAPI, ChromaDB, Ollama, and Groq.

---

## What It Does

Ask natural language questions about Dota 2 and get accurate, context-grounded answers:

- _"What does Mana Break do?"_
- _"Who counters Anti-Mage?"_
- _"What is Enigma's Black Hole ability?"_
- _"What is Anti-Mage's Aghanim's Scepter upgrade?"_
- _"What is the difference between Axe and Anti-Mage?"_

---

## Architecture

```
Question
   │
   ▼
Entity Extraction (Groq)        ← extracts hero/ability names
   │
   ▼
Metadata Filter Builder         ← builds Chroma query filter
   │
   ▼
Vector Search (ChromaDB)        ← retrieves relevant chunks
   │
   ▼
Answer Generation (Groq)        ← generates grounded answer
   │
   ▼
JSON Response
```

### Project Structure

```
dota2-rag/
├── core/                        # Generic reusable RAG framework
│   ├── embedder.py              # Ollama embedding wrapper
│   ├── vector_store.py          # ChromaDB wrapper
│   └── llm.py                   # Groq LLM wrapper
│
├── dota2/                       # Dota 2 specific logic
│   ├── fetch_data.py            # Fetches data from OpenDota API
│   ├── ingest.py                # Chunks heroes and ingests to ChromaDB
│   └── query.py                 # Entity extraction + RAG pipeline
│
├── tests/                       # Full test suite (45 tests)
│   ├── core/                    # Unit tests for core framework
│   ├── dota2/                   # Unit tests for Dota 2 logic
│   └── api/                     # Integration tests for API endpoints
│
├── data/                        # 127 hero JSON files (auto-generated)
├── chroma_db/                   # ChromaDB vector store (auto-generated)
├── main.py                      # FastAPI application
├── pytest.ini                   # Pytest configuration
└── requirements.txt             # Python dependencies
```

---

## Tech Stack

| Component       | Technology                                 |
| --------------- | ------------------------------------------ |
| Web Framework   | FastAPI                                    |
| Vector Database | ChromaDB (local, persistent)               |
| Embeddings      | Ollama + nomic-embed-text (free, local)    |
| LLM             | Groq — llama-3.3-70b-versatile (free tier) |
| Orchestration   | LangChain                                  |
| Testing         | pytest                                     |
| Language        | Python 3.11                                |

---

## Prerequisites

- Python 3.11
- Ollama (see installation below)
- [Groq API key](https://console.groq.com) (free tier)

---

## Installing Ollama

Ollama runs AI models locally on your machine. It is used here for generating text embeddings.

### Windows

Download and run the installer from [ollama.com/download](https://ollama.com/download/windows).

Ollama will install as a background service and start automatically.

### macOS

```bash
brew install ollama
```

Or download the app from [ollama.com/download](https://ollama.com/download/mac).

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Verify Ollama is running

```bash
ollama --version
```

### Pull the embedding model

```bash
ollama pull nomic-embed-text
```

This downloads the `nomic-embed-text` model (~270MB) used to generate vector embeddings.

> **Note:** Ollama must be running in the background whenever you use the API or run ingestion. On Windows and macOS it starts automatically. On Linux, start it with `ollama serve`.

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd dota2-rag
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

---

## Data Preparation

This is a one-time setup. You only need to repeat these steps if you want to refresh the hero data.

### Step 1 — Fetch hero data from OpenDota API

```bash
python dota2/fetch_data.py
```

This calls the [OpenDota API](https://api.opendota.com) — no API key required — and saves one JSON file per hero into `data/`. It fetches hero stats, abilities, counters, Aghanim's upgrades, and facets.

```
✅ Expected output:
Fetching heroes...
Fetching abilities...
Fetching aghanims...
Fetching matchups for 127 heroes... (this takes ~3 minutes)
Saved 127 hero files to data/
```

> **Note:** The matchup step makes 127 individual API calls with a 1 second delay between each to avoid rate limiting. This is why it takes ~3 minutes.

### Step 2 — Ingest data into ChromaDB

```bash
python dota2/ingest.py
```

This reads all hero JSON files from `data/`, splits each hero into chunks, generates vector embeddings using Ollama, and stores everything in ChromaDB.

```
✅ Expected output:
Ingesting 127 heroes...
Total chunks ingested: 1270
Done!
```

> **Note:** Ollama must be running before you run this step. Ingestion takes ~2 minutes depending on your machine.

---

## Running the API

Once data preparation is complete, start the server:

```bash
uvicorn main:app --reload
```

```
✅ Expected output:
🚀 Dota 2 RAG API starting up...
✅ Chroma DB connected
✅ Ready to accept questions!
INFO: Uvicorn running on http://127.0.0.1:8000
```

The API is now available at `http://localhost:8000`.

> **Note:** `--reload` restarts the server automatically when you change any source file. Remove it in production.

---

## API Reference

### `POST /ask`

Ask a question about Dota 2.

**Request:**

```json
{
  "question": "What does Mana Break do?"
}
```

**Response:**

```json
{
  "question": "What does Mana Break do?",
  "answer": "Mana Break burns an opponent's mana on each attack and deals damage equal to 50% of the mana burned.",
  "entities": {
    "hero_name": null,
    "ability_name": "Mana Break"
  },
  "chunks_used": 1
}
```

**Error responses:**

| Status | Reason                                   |
| ------ | ---------------------------------------- |
| 400    | Empty or whitespace-only question        |
| 422    | Missing `question` field in request body |
| 500    | Internal server error                    |

---

### `GET /health`

Check if the API is running.

**Response:**

```json
{
  "status": "healthy",
  "service": "Dota 2 RAG API"
}
```

---

### Interactive Docs

FastAPI auto-generates interactive API documentation. With the server running, visit:

```
http://localhost:8000/docs
```

---

## Data

Each hero file in `data/` contains:

- **Overview** — roles, attack type, primary attribute, base stats
- **Abilities** — display name, description, lore, damage type, pierces BKB, behavior, attributes
- **Counters** — top 5 heroes by win rate against this hero
- **Aghanim's** — scepter and shard upgrade descriptions
- **Facets** — gameplay variant descriptions

Data is sourced from the [OpenDota API](https://docs.opendota.com) and is free to use.

---

## Chunking Strategy

Each hero is split into the following chunks stored in ChromaDB:

| Chunk Type | Metadata          | Content                             |
| ---------- | ----------------- | ----------------------------------- |
| `overview` | `hero`            | Roles, stats, attack type           |
| `ability`  | `hero`, `ability` | One chunk per ability               |
| `counters` | `hero`            | Top 5 counter heroes with win rates |
| `aghanims` | `hero`            | Scepter + shard descriptions        |
| `facets`   | `hero`            | Gameplay variant descriptions       |

**Total: 1270 chunks across 127 heroes**

---

## Running Tests

```bash
pytest -v
```

**Test coverage:**

```
tests/core/test_embedder.py       4 tests  — OllamaEmbeddingFunction
tests/core/test_llm.py            4 tests  — LLMClient
tests/core/test_vector_store.py   5 tests  — VectorStore
tests/dota2/test_ingest.py       12 tests  — convert_hero_to_chunks
tests/dota2/test_query.py        11 tests  — extract_entities, build_filter
tests/api/test_endpoints.py       9 tests  — /ask and /health endpoints

Total: 45 tests
```

All external services (Groq, Ollama, ChromaDB) are mocked in unit tests — no internet connection required to run the test suite.

---

## Reusing the Core Framework

The `core/` directory is intentionally decoupled from Dota 2. To build a RAG API for a different domain (e.g. medical, legal, customer support):

1. Keep `core/` unchanged
2. Create a new domain folder (e.g. `medical/`)
3. Add your own `fetch_data.py`, `ingest.py`, and `query.py`
4. Update `main.py` to import from your new domain folder

```
my-rag-project/
├── core/          ← unchanged, reusable
├── medical/       ← your new domain
│   ├── fetch_data.py
│   ├── ingest.py
│   └── query.py
└── main.py
```

---

## Known Limitations

- Counter data is based on match win rates from OpenDota and reflects the current patch at the time `fetch_data.py` was last run
- The API uses Groq's free tier which has rate limits — heavy usage may result in 429 errors
- Multi-hero comparison questions retrieve up to 3 chunks per hero (6 total) which may miss some context for heroes with many abilities

---

## License

MIT
