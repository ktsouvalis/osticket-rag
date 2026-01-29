# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG (Retrieval-Augmented Generation) pipeline that enables semantic search over an osTicket helpdesk knowledge base (tickets + FAQs). It uses Milvus for vector search, Ollama for embeddings, and MySQL/MariaDB as the osTicket data source. The API is designed to integrate as a Tool in Open WebUI.

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

### Pipeline flow

1. **Data source**: osTicket MySQL DB (`ost_ticket`, `ost_thread`, `ost_thread_entry`, `ost_faq`)
2. **Text processing**: HTML cleaning (BeautifulSoup), email header stripping, junk ticket filtering, secret redaction
3. **Chunking**: `RecursiveCharacterTextSplitter` (chunk_size=1200, overlap=200)
4. **Embedding**: Ollama `bge-m3` model (1024 dimensions)
5. **Storage**: Milvus collection `osticket_knowledge` with HNSW index, COSINE metric
6. **Retrieval**: Vector search → group by document → top chunks with neighbor expansion → character-budgeted output

### Key components

- **`rag_core.py`** — `RagEngine` class: core retrieval logic. Singleton used by both CLI and API. Handles vector search, chunk grouping, neighbor expansion, and result ranking.
- **`rag_api.py`** — FastAPI wrapper. `GET /health` and `GET /ask?query=...` (optional `X-API-Key` header). The `RagEngine` instance is created once at module load.
- **`rag_cli.py`** — Interactive CLI that calls `RagEngine.retrieve_related()`.
- **`10_create_collection.py`** — Schema definition and Milvus collection creation.
- **`20_load_to_milvus.py`** — Full ingestion from MySQL. Generates stable int64 PKs via SHA256. Batch embeds 100 items at a time.
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

## Environment Configuration

Copy `.env.example` to `.env`. Required variables:

- `SERVER_IP` — Host running Milvus (port 19530) and Ollama (port 11434)
- `MYSQL_HOST`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE` — osTicket DB
- `BASE_TICKET_URL` — URL prefix for ticket links (e.g., `https://help.example.com/scp/tickets.php?id=`)

Optional tuning (see defaults in `rag_core.py:RagEngine.__init__`):

- `RAG_API_KEY` — Protect the `/ask` endpoint
- `EMBED_MODEL_NAME` (default: `bge-m3`), `RAG_SEARCH_LIMIT` (120), `RAG_MAX_DOCS` (8), `RAG_TOP_CHUNKS_PER_DOC` (4), `RAG_NEIGHBOR_WINDOW` (1), `RAG_MAX_CONTEXT_CHARS` (24000), `RAG_SEARCH_EF` (64)
- `RAG_API_DEBUG=1` — Expose error details in API responses
- `LOG_LEVEL` — Logging verbosity

## Important Notes

- Changing the embedding model or dimension requires a full rebuild: run `10_create_collection.py` then `20_load_to_milvus.py`.
- Secret redaction happens before embedding and storage — credentials matching known patterns are replaced with `[REDACTED]`.
- The Docker API container exposes port 8800 (mapped to internal 8000).
- No unit tests exist; testing is done manually via `rag_cli.py` and the helper scripts.
