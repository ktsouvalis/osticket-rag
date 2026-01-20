import os
import logging
from fastapi import FastAPI, Header, HTTPException, Query
from pydantic import BaseModel
from rag_core import RagEngine
from translation_service import send_to_translation_service
app = FastAPI(title="osTicket RAG API", version="1.0")

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("osticket-rag")

# Create once at startup (keeps Milvus loaded, avoids reconnect per request)
ENGINE = RagEngine()

# Optional: protect endpoint with an API key
# If RAG_API_KEY is set, requests must send header: X-API-Key: <value>
API_KEY = (os.getenv("RAG_API_KEY") or "").strip()


class AskResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/ask", response_model=AskResponse)
def ask(
    query: str = Query(..., min_length=1, description="Search query"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    translate: bool = False,
):
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    q = (query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty query")

    try:
        answer_text = ENGINE.answer(q)
        if translate:
            answer_text = send_to_translation_service(answer_text)
        return AskResponse(answer=answer_text)
    except Exception as exc:
        logger.exception("/ask failed")
        if os.getenv("RAG_API_DEBUG", "0") == "1":
            raise HTTPException(status_code=500, detail=f"RAG backend error: {exc!r}")
        raise HTTPException(status_code=500, detail="RAG backend error")
