# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG (Retrieval-Augmented Generation) pipeline that enables semantic search over an osTicket knowledge base (tickets + FAQs). It uses Milvus for vector search, Ollama for embeddings, MySQL/MariaDB as the osTicket data source, and integrates with Open WebUI as a Tool so that qwen2.5:14b can answer user questions in English based on Greek ticket content.

## Commands

### Docker (production)

```bash
make install              # Full init: create-collection + load-initial + api-up
make api-up               # Build and start the API container (port 8800→8000)
make api-down             # Stop all containers
make create-collection    # Drop and recreate the Milvus collection
make load-initial         # Full initial data load into Milvus
make update               # Incremental update (delta load based on state watermark)
make check-updates        # Dry-run incremental update (no changes applied)
```

### Local development

```bash
conda activate osticket-rag
python rag_cli.py                           # Interactive CLI for testing queries
uvicorn rag_api:app --host 0.0.0.0 --port 8000  # Run API locally
python 10_create_collection.py              # Create/reset Milvus collection
python 20_load_to_milvus.py                 # Full load
python 30_update_milvus.py                  # Incremental update
python 30_update_milvus.py --dry-run        # Dry-run incremental update
```

### Helper/diagnostic scripts

```bash
python 01_HELPER_verify_milvus.py           # Test Milvus connectivity
python 11_HELPER_extract_raw_ticket.py 1234 # Dump raw ticket thread from MySQL
python 41_HELPER_vector_search.py           # Interactive vector search with full context
```

## Architecture

### End-to-end flow

1. User asks a question in **Open WebUI** (chat interface at `10.23.2.165:3000`)
2. **qwen2.5:14b** (on Ollama) decides to call the osTicket RAG Search tool
3. The tool calls the **RAG API** (`/ask` endpoint at `195.251.13.132:8800`)
4. RAG API embeds the query via **Ollama bge-m3** and searches **Milvus** (vector ANN)
5. Results are **reranked** by a cross-encoder (`bge-reranker-v2-m3`) for precision
6. Full ticket threads are fetched from **MySQL** (not just chunks)
7. API returns ticket metadata + full context text
8. qwen2.5:14b reads the Greek content and generates an English answer with source URLs

### Ingestion pipeline

1. **Data source**: osTicket MySQL DB (`ost_ticket`, `ost_thread`, `ost_thread_entry`, `ost_faq`)
2. **Text processing**: HTML cleaning (BeautifulSoup), email header stripping, junk ticket filtering, secret redaction
3. **Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=1200, overlap=200)
4. **Embedding**: Ollama `bge-m3` model (1024 dimensions)
5. **Storage**: Milvus collection `osticket_knowledge` with HNSW index (M=16, efConstruction=256), COSINE metric

### Retrieval pipeline

1. **Vector search**: Milvus ANN with HNSW, `ef = max(RAG_SEARCH_EF, search_limit)`
2. **Grouping**: Hits grouped by document (ticket/FAQ), ranked by best chunk score
3. **Reranking**: Cross-encoder (`bge-reranker-v2-m3`) re-scores top candidates using (query, chunk) pairs — fixes ordering when vector search ranks inventory/procurement tickets above actual problem reports
4. **Full ticket fetch**: For each top document, the complete thread is fetched from MySQL (not just the matching chunks), so the LLM sees the full issue-to-resolution context
5. **Character budget**: Total context capped at `RAG_MAX_CONTEXT_CHARS` (default 24000), with truncation fallback

### Key components

- **`rag_core.py`** — `RagEngine` class: core retrieval logic. Connects to Milvus, Ollama, and MySQL. Loads the cross-encoder reranker on init. Singleton used by both CLI and API.
- **`rag_api.py`** — FastAPI wrapper. `GET /health` and `GET /ask?query=...` (optional `X-API-Key` header). Returns `RelatedDoc` list with `context` field containing full ticket text.
- **`rag_cli.py`** — Interactive CLI that calls `RagEngine.retrieve_related()`.
- **`openwebui_tool.py`** — Open WebUI Tool definition. Paste into Workspace > Tools in Open WebUI. Calls the RAG API and formats results for the LLM. Configurable via Valves (API URL and key).
- **`10_create_collection.py`** — Schema definition and Milvus collection creation (HNSW index).
- **`20_load_to_milvus.py`** — Full ingestion from MySQL. Batch embeds 100 items at a time.
- **`30_update_milvus.py`** — Incremental updates using a watermark timestamp from `state/.milvus_update_state.json`. Deletes old vectors for changed tickets, then reinserts.

### Numbered script convention

Scripts are prefixed with numbers indicating execution order: `10_` (collection setup) → `20_` (full load) → `30_` (incremental updates). Helper scripts use `01_`, `11_`, `41_` prefixes.

### State management

Incremental updates rely on a watermark in `state/.milvus_update_state.json` (`last_activity_ts`, `last_faq_id`). This file is bind-mounted as a Docker volume (`./state:/app/state`) so it persists across container runs.

### FAQ ID scheme

FAQs are stored in Milvus with `ticket_id = faq_id + 100000` to avoid collisions with actual ticket IDs.

### Special retrieval modes

- **Broad query adaptation**: Detects enumeration-style queries ("list all", "which", etc.) via regex and multiplies search limits.
- **VLAN enumeration**: Special mode for extracting VLAN IDs from tickets when queries mention "vlan".

## Infrastructure

| Service | Host | Port |
|---------|------|------|
| Open WebUI | 10.23.2.165 | 3000 |
| Ollama (bge-m3, qwen2.5:14b) | 10.23.2.165 | 11434 |
| Milvus | 10.23.2.165 | 19530 |
| RAG API | 195.251.13.132 | 8800 |
| osTicket MySQL | 10.23.1.99 | 3306 |

## Environment Configuration

Copy `.env.example` to `.env`. Required variables:

- `SERVER_IP` — Host running Milvus (port 19530) and Ollama (port 11434)
- `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` — osTicket DB
- `BASE_TICKET_URL` — URL prefix for ticket links (e.g., `https://help.example.com/scp/tickets.php?id=`)

Optional tuning (see defaults in `rag_core.py:RagEngine.__init__`):

- `RAG_API_KEY` — Protect the `/ask` endpoint
- `EMBED_MODEL_NAME` (default: `bge-m3`), `RAG_SEARCH_LIMIT` (120), `RAG_MAX_DOCS` (8), `RAG_MAX_CONTEXT_CHARS` (24000), `RAG_SEARCH_EF` (128)
- `RAG_RERANKER_MODEL` (default: `BAAI/bge-reranker-v2-m3`) — Set to empty string to disable reranking
- `RAG_RERANKER_DEVICE` (default: `cpu`) — Set to `cuda` if the API server has a compatible GPU
- `RAG_RERANK_CANDIDATES` (default: `20`) — Number of top documents to rerank
- `RAG_API_DEBUG=1` — Expose error details in API responses
- `LOG_LEVEL` — Logging verbosity

## Open WebUI Setup

1. Open WebUI runs at `http://10.23.2.165:3000`
2. The Tool is defined in `openwebui_tool.py` — paste its contents into **Workspace > Tools**
3. Configure the Tool's Valves: `rag_api_url` and `rag_api_key`
4. Create a model preset using **qwen2.5:14b** with the osTicket RAG Search tool enabled
5. System prompt should instruct the model to: use the tool for ticket queries, summarize each ticket in English, not hallucinate resolutions, and list sources with URLs

## Important Notes

- Changing the embedding model or dimension requires a full rebuild: `10_create_collection.py` then `20_load_to_milvus.py`.
- Secret redaction happens before embedding/storage — passwords, API keys, tokens, and URL credentials are replaced with `[REDACTED]`.
- The reranker runs on CPU by default. First startup downloads the model (~560MB) from HuggingFace.
- The Docker API container exposes port 8800 (mapped to internal 8000).
- No unit tests exist; testing is done manually via `rag_cli.py`, the helper scripts, and Open WebUI.
