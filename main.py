from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from dota2.query import ask
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

# ── Lifespan event handler
# This runs ONCE when the app starts up
# We use it to verify Chroma DB is accessible before accepting requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    # STARTUP
    print("🚀 Dota 2 RAG API starting up...")

    # Actually verify Chroma is accessible by counting chunks
    # If chroma_db/ is missing or corrupted this will raise an exception
    # and the server will refuse to start
    try:
        from dota2.query import store
        count = store.count()
        if count == 0:
            raise RuntimeError("ChromaDB is empty — run python dota2/ingest.py first")
        print(f"✅ Chroma DB connected — {count} chunks loaded")
        print("✅ Ready to accept questions!")
    except Exception as e:
        print(f"❌ Startup failed: {e}")
        raise  # re-raise so the server refuses to start

    yield

    # SHUTDOWN
    print("👋 Shutting down...")

# ── Create FastAPI app
# lifespan= tells FastAPI to use our startup/shutdown handler
app = FastAPI(
    title="Dota 2 RAG API",
    description="Ask questions about Dota 2 heroes and abilities",
    version="1.0.0",
    lifespan=lifespan
)

# CORSMiddleware allows frontend apps to call our API
# allow_origins=["*"] means allow ALL origins — fine for development
# In production you'd restrict this to your frontend's domain only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],       # allow GET, POST, PUT etc
    allow_headers=["*"],       # allow all headers
)

# ── Request model
# Pydantic validates incoming request body automatically
# If "question" is missing → FastAPI returns 422 error automatically
class QuestionRequest(BaseModel):
    question: str

# ── Response model
# Defines the shape of our response
# This also auto-generates API documentation
class QuestionResponse(BaseModel):
    question: str
    answer: str
    entities: dict
    chunks_used: int

# ── Health check endpoint
# GET /health → tells us if the API is running
# Useful for monitoring and Docker health checks
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Dota 2 RAG API"}

# ── Main endpoint
# POST /ask → accepts a question, returns an answer
@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):

    # Validate question is not empty
    if not request.question.strip():
        # HTTPException sends a proper HTTP error response
        # 400 = Bad Request
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        # Call our full RAG pipeline from query.py
        result = ask(request.question)
        return QuestionResponse(
            question=result["question"],
            answer=result["answer"],
            entities=result["entities"],
            chunks_used=result["chunks_used"]
        )
    except Exception as e:
        # 500 = Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )