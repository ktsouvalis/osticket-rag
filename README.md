# osTicket RAG (Milvus + Ollama + Open WebUI)

RAG pipeline for querying an **osTicket** knowledge base (tickets + FAQs) using:

- **Milvus** for vector search (HNSW index, COSINE metric)
- **Ollama** for embeddings (`bge-m3`) and chat (`qwen2.5:14b`)
- **Cross-encoder reranker** (`bge-reranker-v2-m3`) for precision
- **MySQL/MariaDB** (osTicket DB) as the source
- **Open WebUI** as the chat interface with tool integration

Users ask questions in Open WebUI, the model calls the RAG API to find relevant tickets, fetches full ticket threads from MySQL, and generates an English answer from Greek ticket content — with source URLs.

---

## How it works

```
User (Open WebUI) → qwen2.5:14b → RAG Tool → API (/ask)
                                                  ↓
                                          Embed query (bge-m3 via Ollama)
                                                  ↓
                                          Vector search (Milvus HNSW)
                                                  ↓
                                          Rerank (bge-reranker-v2-m3)
                                                  ↓
                                          Fetch full tickets (MySQL)
                                                  ↓
                                          Return context + metadata
                                                  ↓
                                   qwen2.5:14b generates English answer
                                   with source ticket URLs
```

---

## Files overview

| File | Purpose |
|------|---------|
| `rag_core.py` | `RagEngine` class — vector search, reranking, full ticket fetch from MySQL |
| `rag_api.py` | FastAPI wrapper: `GET /health`, `GET /ask?query=...` |
| `rag_cli.py` | Interactive CLI for testing queries |
| `openwebui_tool.py` | Open WebUI Tool definition — paste into Workspace > Tools |
| `10_create_collection.py` | Creates/resets Milvus collection with HNSW index |
| `20_load_to_milvus.py` | Full ingestion from MySQL → chunk → embed → Milvus |
| `30_update_milvus.py` | Incremental updates using watermark timestamp |
| `01_HELPER_verify_milvus.py` | Test Milvus connectivity |
| `11_HELPER_extract_raw_ticket.py` | Dump raw ticket thread from MySQL |
| `41_HELPER_vector_search.py` | Interactive vector search with full context display |

Scripts are prefixed with numbers indicating execution order: `10_` → `20_` → `30_`.

---

## Prerequisites

Network connectivity from the API server to:

- **Milvus**: `SERVER_IP:19530`
- **Ollama**: `SERVER_IP:11434`
- **MySQL/MariaDB** (osTicket DB): `MYSQL_HOST:3306`

---

## Setup

### 1) Create Conda env and install dependencies

```bash
conda create -n osticket-rag python=3.11 -y
conda activate osticket-rag
pip install -r requirements.txt
```

### 2) Configure environment

```bash
cp .env.example .env
```

Fill in `.env`:

- `SERVER_IP=` — host running Milvus + Ollama
- `MYSQL_HOST=`, `MYSQL_USER=`, `MYSQL_PASSWORD=`, `MYSQL_DATABASE=`
- `BASE_TICKET_URL=` (e.g. `https://patra-helpdesk.uop.gr/scp/tickets.php?id=`)
- `RAG_API_KEY=` — protects the `/ask` endpoint

Optional tuning:

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODEL_NAME` | `bge-m3` | Ollama embedding model |
| `RAG_SEARCH_LIMIT` | `120` | Initial vector search results |
| `RAG_MAX_DOCS` | `8` | Max documents returned |
| `RAG_MAX_CONTEXT_CHARS` | `24000` | Character budget for context |
| `RAG_SEARCH_EF` | `128` | HNSW search ef (auto-scales to >= limit) |
| `RAG_RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder model (empty string to disable) |
| `RAG_RERANKER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `RAG_RERANK_CANDIDATES` | `20` | Documents to rerank |
| `RAG_API_DEBUG` | `0` | Set to `1` to expose error details |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

---

## Milvus workflow

1. `10_create_collection.py` — only when you need a clean rebuild (schema/model changes).
2. `20_load_to_milvus.py` — once after a clean rebuild (full load + sets watermark).
3. `30_update_milvus.py` — for regular operations (incremental updates only).

### A) Create / reset collection

```bash
python 10_create_collection.py
```

### B) Initial full load

```bash
python 20_load_to_milvus.py
```

This also initializes `state/.milvus_update_state.json` so the incremental updater starts from the current watermark.

### C) Incremental update

```bash
python 30_update_milvus.py
python 30_update_milvus.py --dry-run    # preview without changes
python 30_update_milvus.py --include-faq # also update FAQs
```

---

## Running the API

### Local

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
make api-up        # build and start (port 8800 → 8000)
make api-down      # stop
```

### Test

```bash
curl -s http://localhost:8800/health

curl -G "http://localhost:8800/ask" \
  -H "X-API-Key: YOUR_API_KEY" \
  --data-urlencode "query=network issue"
```

The `/ask` endpoint returns:

```json
{
  "results": [
    {
      "doc_key": "ticket:000324",
      "source_type": "ticket",
      "ticket_id": 341,
      "ticket_number": "000324",
      "subject": "...",
      "top_score": 0.64,
      "url": "https://patra-helpdesk.uop.gr/scp/tickets.php?id=341",
      "context": "Subject: ...\n\n--- Post by ... ---\n..."
    }
  ]
}
```

---

## Open WebUI integration

Open WebUI provides the chat interface where users interact with the RAG system.

### 1) Add the Tool

- Go to **Workspace > Tools > "+"** in Open WebUI
- Paste the contents of `openwebui_tool.py`
- Save, then click the gear icon and fill in the Valves:
  - `rag_api_url`: your API URL (e.g. `http://195.251.13.132:8800`)
  - `rag_api_key`: your API key

### 2) Configure the model

- Go to **Workspace > Models**, create a preset using **qwen2.5:14b**
- Enable the **osTicket RAG Search** tool
- Set the system prompt:

```
You are an IT support assistant with access to an osTicket knowledge base.
Always respond in English, regardless of the language of the user's question.

When the user asks about past issues, incidents, or infrastructure topics, use the search_tickets tool to find relevant tickets.

When presenting results from the tool:
- Summarize EACH returned ticket: what the problem was, what was investigated, and how it was resolved.
- If the resolution is not clearly described in the ticket content, say "resolution not documented" instead of guessing.
- If multiple tickets were returned, describe all of them, not just the first one.
- The ticket content is in Greek. Translate and summarize it in English.
- At the end of your answer, list all source tickets in this format:

**Sources:**
- Ticket #000123 - Subject here - [Link](URL here)
- Ticket #000456 - Subject here - [Link](URL here)

Do NOT generate citation markup like [source id="1"]. Just use the format above.
```

### 3) Test

Start a chat and ask something like:

> What network issues have been reported?

The model will call the RAG tool, get ticket context, and answer in English with source URLs.

---

## Deploy on an app server (Docker / Portainer)

Recommended setup: API on an app server, Milvus + Ollama + Open WebUI on a GPU server.

### 1) Clone and configure

```bash
git clone <REPO_URL>
cd osticket-rag
cp .env.example .env
# Fill in SERVER_IP, MYSQL_*, RAG_API_KEY, BASE_TICKET_URL
```

### 2) Initialize

```bash
make install    # create-collection + load-initial + api-up
```

### 3) Incremental updates

```bash
make update           # run on-demand
make check-updates    # dry-run
```

Schedule with host cron or Portainer scheduled job.

State is persisted in `./state/` (bind-mounted as Docker volume).

---

## Notes

- If you change the embedding model or dimension, you must do a full rebuild: `make create-collection` then `make load-initial`.
- Secret redaction runs before embedding/storage — passwords, API keys, tokens, and URL credentials are replaced with `[REDACTED]`.
- The reranker model (~560MB) is downloaded from HuggingFace on first startup.
- The reranker runs on CPU by default. Set `RAG_RERANKER_DEVICE=cuda` if the API server has a compatible GPU.
- FAQs are stored with `ticket_id = faq_id + 100000` to avoid collisions with ticket IDs.

---

## License

MIT — see `LICENSE`.
