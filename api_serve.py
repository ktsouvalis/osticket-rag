"""FastAPI wrapper for the OSTicket retriever (RAG retrieval only).

Run:
    uvicorn api_serve:app --host 0.0.0.0 --port 8000

Notes:
- `include_answer` is accepted for backwards compatibility but ignored.
- The response always contains `answer: null`.
"""

from __future__ import annotations

from fastapi import FastAPI, Query

from rag_core import RagEngine

app = FastAPI(title="osticket-rag (retrieval only)")

# Instantiate once (keeps Milvus connection + Ollama client warm)
_engine = RagEngine()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ask")
def ask(
    query: str = Query(..., min_length=1, description="User query"),
    include_answer: bool = Query(False, description="Ignored (kept for backwards compatibility)"),
    max_docs: int = Query(5, ge=1, le=50, description="How many top tickets to return"),
) -> dict:
    # include_answer is intentionally ignored.
    return _engine.query(query, max_docs=max_docs)
