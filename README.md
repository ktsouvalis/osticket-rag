# osTicket RAG (Milvus + Ollama)

RAG pipeline for querying an **osTicket** knowledge base (tickets + FAQs) using:

- **Milvus** for vector search (COSINE)
- **Ollama** for embeddings + LLM answers
- **MySQL/MariaDB** (osTicket DB) as the source

This repo is intentionally split into:
- **Control scripts** (create collection, full load, incremental updates)
- A reusable **RAG engine** (`rag_core.py`) that returns a string (usable from CLI or API)
- A small **HTTP API** (`50_rag_api.py`) to integrate as a Tool in Open WebUI

---

## Files overview

- `10_create_collection.py`
  - Creates Milvus collection `osticket_knowledge`
  - Schema fields: `ticket_id`, `ticket_number`, `source_type`, `chunk_index`, `subject`, `text_payload`, `vector`
  - Creates IVF_FLAT index with `metric_type = COSINE`

- `20_load_to_milvus.py`
  - Full ingestion (first run / rebuild)
  - Loads all ticket threads + published FAQs from osTicket DB
  - Redacts common secrets **before** embedding and inserting

- `30_update_milvus.py`
  - Incremental updater
  - Detects tickets with activity after a saved watermark (`last_activity_ts`)
  - Deletes + reinserts only those tickets in Milvus
  - Keeps state in `.milvus_update_state.json`

- `rag_core.py`
  - `RagEngine.answer(query: str) -> str`
  - Uses Milvus chunk retrieval + neighbor chunk expansion
  - Returns citations and splits **used references** vs **retrieved-but-not-used**

- `40_rag_answer.py`
  - CLI runner (interactive). Intended for local testing.

- `50_rag_api.py`
  - FastAPI wrapper around `rag_core.RagEngine`
  - Endpoints:
    - `GET /health`
    - `POST /ask` → `{ "answer": "..." }`
  - Optional API key via `RAG_API_KEY` header `X-API-Key`

- Helpers (debug / diagnostics)
  - `01_HELPER_verify_milvus.py`
    - Verifies Milvus connectivity and prints server version.
  - `11_HELPER_extract_raw_ticket.py`
    - Dumps the raw osTicket thread entries (no cleanup) for a given `ticket_id` from MySQL.
    - Example usage: `python 11_HELPER_extract_raw_ticket.py 1234`
  - `41_HELPER_vector_search.py`
    - Runs a Milvus vector search for an interactive query, prints the matching chunks (`text_payload`),
      and shows a truncated full MySQL thread/FAQ for deeper debugging.
---

## Prerequisites

You need network connectivity from where you run the scripts/API to:

- **Milvus**: `SERVER_IP:19530`
- **Ollama**: `SERVER_IP:11434`
- **MySQL/MariaDB (osTicket DB)**: `MYSQL_HOST:3306`

---

## Setup

### 1) Create Conda env and install dependencies

```bash
cd /home/ktsouvalis/Desktop/Dev/osticket-rag
conda create -n osticket-rag python=3.11 -y
conda activate osticket-rag
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

- `SERVER_IP=` (host that runs Milvus + Ollama)
- `MYSQL_HOST=`
- `MYSQL_USER=`
- `MYSQL_PASSWORD=`
- `MYSQL_DATABASE=`
- `RESPONSE_MODEL_NAME=` (example: `qwen2.5:14b`)
- `BASE_TICKET_URL=` (example: `https://patra-helpdesk.uop.gr/scp/tickets.php?id=`)
- `RESET_COLLECTION=` (`0` or `1`)

Optional:
- `EMBED_MODEL_NAME=bge-m3`
- `RAG_API_KEY=...` (protects the API)

---

## Milvus workflow

### A) Create / reset collection

To recreate from scratch:

1. Set `RESET_COLLECTION=1` in `.env`
2. Run:

```bash
conda activate osticket-rag
python 10_create_collection.py
```

Then set `RESET_COLLECTION=0` again.

### B) Initial full load

```bash
conda activate osticket-rag
python 20_load_to_milvus.py
```

### C) Incremental update (new or updated tickets)

```bash
conda activate osticket-rag
python 30_update_milvus.py
```

Dry run:

```bash
conda activate osticket-rag
python 30_update_milvus.py --dry-run
```

State file: `.milvus_update_state.json` (already in `.gitignore`).

---

## Querying (CLI)

```bash
conda activate osticket-rag
python 40_rag_answer.py
```

---

## Running the API (for Open WebUI Tool)

Start the API (example port 8000):

```bash
conda activate osticket-rag
uvicorn 50_rag_api:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"install gitea on-prem"}'
```

If you set `RAG_API_KEY`, include:

```bash
-H 'X-API-Key: your-key'
```

---

## Notes

- Vector search uses `metric_type = COSINE` to match the collection index.
- If you change embedding model or embedding dimension, you must reset and rebuild:
  - `10_create_collection.py` (with `RESET_COLLECTION=1`)
  - `20_load_to_milvus.py`

---

## License

MIT — see `LICENSE`.
