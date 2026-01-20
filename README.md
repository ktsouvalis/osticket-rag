# osTicket RAG (Milvus + Ollama)

RAG pipeline for querying an **osTicket** knowledge base (tickets + FAQs) using:

- **Milvus** for vector search (COSINE)
- **Ollama** for embeddings
- **MySQL/MariaDB** (osTicket DB) as the source

This repo is intentionally split into:
- **Control scripts** (create collection, full load, incremental updates)
- A reusable **RAG engine** (`rag_core.py`) that returns related tickets (usable from CLI or API)
- A small **HTTP API** (`rag_api.py`) to integrate as a Tool in Open WebUI

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
  - Initializes `.milvus_update_state.json` for the incremental updater

- `30_update_milvus.py`
  - Incremental updater
  - Detects tickets with activity after a saved watermark (`last_activity_ts`)
  - Deletes + reinserts only those tickets in Milvus
  - Keeps state in `.milvus_update_state.json` (persisted via Docker volume)

- `rag_core.py`
  - `RagEngine.retrieve_related(query: str) -> list[dict]`
  - Uses Milvus chunk retrieval + neighbor chunk expansion
  - Returns related tickets only (no ticket content, no LLM answer)

- `rag_cli.py`
  - CLI runner (interactive).
  - Prints related ticket number, subject, and URL

- `rag_api.py`
  - FastAPI wrapper around `rag_core.RagEngine`
  - Endpoints:
    - `GET /health`
    - `POST /ask` → `{ "results": [ { "ticket_number", "subject", "url", ... } ] }`
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
- `BASE_TICKET_URL=` (example: `https://patra-helpdesk.uop.gr/scp/tickets.php?id=`)

Optional:
- `EMBED_MODEL_NAME=bge-m3`
- `RAG_API_KEY=...` (protects the API)

---

## Milvus workflow

Flow summary:

1. `10_create_collection.py` only when you need a clean rebuild (schema/model changes).
2. `20_load_to_milvus.py` once after a clean rebuild (full load + sets watermark).
3. `30_update_milvus.py` for regular operations (incremental updates only).

### A) Create / reset collection

To recreate from scratch run:

```bash
conda activate osticket-rag
python 10_create_collection.py
```

### B) Initial full load

```bash
conda activate osticket-rag
python 20_load_to_milvus.py
```

This also initializes `.milvus_update_state.json` so the incremental updater starts from the current watermark.

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
python rag_cli.py
```

---

## Running the API (for Open WebUI Tool)

Start the API (example port 8000):

```bash
conda activate osticket-rag
uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

Test:

```bash
curl -s http://localhost:8000/health

curl -G "http://localhost:8000/ask" \
  -H "X-API-Key: YOUR_API_KEY" \
  --data-urlencode "query=your search here"
```

---

## Deploy on an app server (Docker / Portainer)

This is the recommended way to run the API on an app server while **Milvus + Ollama + Open WebUI** live elsewhere (e.g. GPU server).

### 1) Clone the repo on the app server

```bash
git clone https://github.com/<YOU>/<REPO>.git
cd <REPO>
cp .env.example .env
```

Set at least:

- `SERVER_IP=` (GPU server IP where Milvus+Ollama run)
- `MYSQL_HOST=`, `MYSQL_USER=`, `MYSQL_PASSWORD=`, `MYSQL_DATABASE=`
- `RAG_API_KEY=` (recommended)

### 2) Create the Milvus collection
```bash
make create-collection
```

### 3) Initial full load
```bash
make load-initial
```

### 4) Start the API container

With Docker Compose (or a Portainer Stack using `docker-compose.yml`):

```bash
make api-up
```

Healthcheck:

```bash
curl -s http://localhost:8000/health
```

### 5) Run incremental updates

Run on-demand (one-off container):

```bash
make update
```

State persistence:

- The initial loader and the updater store their watermark in `/app/.milvus_update_state.json`, and `./.milvus_update_state.json` is bind-mounted into the container so it survives recreation.

Scheduling options:

- **Host cron** (simple/reliable): run the command above every X minutes.
- **Portainer scheduled job** (if enabled): run the same container command.

---

## Notes

- Vector search uses `metric_type = COSINE` to match the collection index.
- If you change embedding model or embedding dimension, you must reset and rebuild:
  - `10_create_collection.py`
  - `20_load_to_milvus.py`

---

## License

MIT — see `LICENSE`.
