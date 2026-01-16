import os
import logging
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from rag_core import RagEngine
app = FastAPI(title="osTicket RAG API", version="1.0")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("osticket-rag")

# Create once at startup (keeps Milvus loaded, avoids reconnect per request)
ENGINE = RagEngine()

# Optional: protect endpoint with an API key
# If RAG_API_KEY is set, requests must send header: X-API-Key: <value>
API_KEY = (os.getenv("RAG_API_KEY") or "").strip()


class AskRequest(BaseModel):
    query: str


class RelatedDoc(BaseModel):
    doc_key: str
    source_type: str
    ticket_id: int | None = None
    ticket_number: str
    subject: str
    top_score: float | None = None
    url: str | None = None


class AskResponse(BaseModel):
    results: list[RelatedDoc]


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        results = ENGINE.retrieve_related(q)
        return AskResponse(results=results)
    except Exception as exc:
        logger.exception("/ask failed")
        if os.getenv("RAG_API_DEBUG", "0") == "1":
            raise HTTPException(status_code=500, detail=f"RAG backend error: {exc!r}")
        raise HTTPException(status_code=500, detail="RAG backend error")
