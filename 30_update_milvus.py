# detects tickets created/updated after the saved timestamp
# deletes + reinserts only those tickets in Milvus
# keeps a JSON state (last_activity_ts)

import argparse
import json
import os
import re
import time
from collections import defaultdict

import mysql.connector
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import Client
from pymilvus import Collection, connections


STATE_FILE_DEFAULT = ".milvus_update_state.json"
COLLECTION_NAME_DEFAULT = "osticket_knowledge"


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"last_activity_ts": 0, "last_faq_id": 0}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data.setdefault("last_activity_ts", 0)
    data.setdefault("last_faq_id", 0)
    return data


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def clean_text(html_content):
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for img in soup.find_all("img"):
        img.decompose()
    text = soup.get_text(separator="\n")
    patterns_to_cut = [
        r"From:.*",
        r"Sent:.*",
        r"---.*Forwarded Message.*---",
        r"Προωθημένο μήνυμα",
        r"On.*wrote:.*",
        r"Στις.*έγραψε:.*",
    ]
    for pattern in patterns_to_cut:
        text = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)[0]
    return re.sub(r"\n\s*\n", "\n\n", text).strip()


def redact_secrets(text: str) -> str:
    """
    Redact common credential patterns BEFORE embedding/storing.
    Keep consistent with 20_load_to_milvus.py.
    """
    if not text:
        return ""
    patterns = [
        r"(\b(?:admin|root|netadmin|user)\b)\s*/\s*([^\s\)]+)",
        r"(?i)(password\s*:\s*)(\S+)",
        r"(?i)(only\s+password\s*:\s*)(\S+)",
    ]
    redacted = text
    redacted = re.sub(patterns[0], r"\1 / [REDACTED]", redacted)
    redacted = re.sub(patterns[1], r"\1[REDACTED]", redacted)
    redacted = re.sub(patterns[2], r"\1[REDACTED]", redacted)
    return redacted


def ensure_vector_batch(embeddings):
    # Ollama may return [vector] or [[vector]] depending on input
    if embeddings and isinstance(embeddings, list) and isinstance(embeddings[0], float):
        return [embeddings]
    return embeddings


def get_embeddings_from_ollama(client: Client, model: str, texts: list[str]) -> list[list[float]]:
    resp = client.embed(model=model, input=texts)
    embs = ensure_vector_batch(resp.get("embeddings"))
    if not embs or len(embs) != len(texts):
        raise RuntimeError(f"Embedding mismatch: got {len(embs) if embs else 0} embeddings for {len(texts)} texts")
    return embs


def get_collection_vector_dim(collection_obj: Collection) -> int:
    for field in collection_obj.schema.fields:
        if field.name == "vector":
            dim = None
            if hasattr(field, "params"):
                dim = field.params.get("dim")
            if not dim and hasattr(field, "type_params"):
                dim = field.type_params.get("dim")
            if dim is None:
                raise RuntimeError("Collection vector field missing dim in schema.")
            return int(dim)
    raise RuntimeError("Collection schema missing vector field.")


def ensure_embedding_dim(client_obj: Client, model: str, collection_obj: Collection) -> None:
    resp = client_obj.embed(model=model, input="dim probe")
    embs = ensure_vector_batch(resp.get("embeddings"))
    if not embs:
        raise RuntimeError("Embedding probe failed: no embeddings returned.")
    vec = embs[0]
    dim = get_collection_vector_dim(collection_obj)
    if len(vec) != dim:
        raise RuntimeError(f"Embedding dim {len(vec)} does not match collection dim {dim} for model '{model}'.")


def _escape_milvus_str(value: str) -> str:
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def delete_ticket_rows_bulk(collection: Collection, ticket_ids: list[int], source_type: str = "ticket", chunk: int = 500):
    """
    Delete existing vectors for a set of ticket_ids (so reinsert becomes an "update").
    """
    if not ticket_ids:
        return
    st = _escape_milvus_str(source_type)
    ids = [int(x) for x in ticket_ids]
    for i in range(0, len(ids), chunk):
        part = ids[i : i + chunk]
        ids_csv = ", ".join(str(x) for x in part)
        expr = f"ticket_id in [{ids_csv}] && source_type == '{st}'"
        collection.delete(expr)
    collection.flush()


def main():
    load_dotenv()

    server_ip = os.getenv("SERVER_IP")
    if not server_ip:
        raise RuntimeError("SERVER_IP is not set in .env")

    embed_model = os.getenv("EMBED_MODEL_NAME", "bge-m3")
    collection_name = os.getenv("MILVUS_COLLECTION", COLLECTION_NAME_DEFAULT)

    mysql_host = os.getenv("MYSQL_HOST")
    mysql_user = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_database = os.getenv("MYSQL_DATABASE")
    if not all([mysql_host, mysql_user, mysql_password, mysql_database]):
        raise RuntimeError("Missing MYSQL_* env vars in .env")

    ap = argparse.ArgumentParser(
        description="Incrementally load new/updated osTicket tickets/FAQs into Milvus (delta update by activity timestamp)."
    )
    ap.add_argument("--state-file", default=STATE_FILE_DEFAULT, help="Progress state file (default: .milvus_update_state.json)")
    ap.add_argument("--since-ts", type=int, default=None, help="Override: unix epoch seconds (exclusive).")
    ap.add_argument("--include-faq", action="store_true", help="Also update FAQs incrementally (still ID-based here).")
    ap.add_argument("--since-faq-id", type=int, default=None, help="Override: start from this faq_id (exclusive)")
    ap.add_argument("--batch-size", type=int, default=100, help="Embedding batch size (default: 100)")
    ap.add_argument("--dry-run", action="store_true", help="Do not write to Milvus")
    args = ap.parse_args()

    state = load_state(args.state_file)
    last_activity_ts = int(args.since_ts) if args.since_ts is not None else int(state.get("last_activity_ts", 0))

    last_faq_id = int(args.since_faq_id) if args.since_faq_id is not None else int(state.get("last_faq_id", 0))

    # Match 20_load_to_milvus.py
    JUNK_KEYWORDS = ["backup failed", "vzdump", "unifi controller", "cron", "alert", "status", "successful backup"]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

    # Connections
    print(f"🔌 Connecting Milvus @ {server_ip}:19530 ...")
    connections.connect(host=server_ip, port="19530")
    collection = Collection(collection_name)
    collection.load()
    ollama = Client(host=f"http://{server_ip}:11434")
    ensure_embedding_dim(ollama, embed_model, collection)

    print(f"🔌 Connecting MySQL @ {mysql_host} ...")
    db = mysql.connector.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
    )
    cursor = db.cursor(dictionary=True)

    print(f"📥 Using Ollama model '{embed_model}' for embeddings...")
    print(f"🕒 Delta watermark: last_activity_ts={last_activity_ts} (unix)")

    # 1) Find tickets with ANY activity > watermark
    # Use a UNION to avoid complicated MAX/GREATEST over joins.
    cursor.execute(
        """
        SELECT ticket_id, UNIX_TIMESTAMP(MAX(ts)) AS max_activity_ts
        FROM (
            SELECT ticket_id, updated AS ts
            FROM ost_ticket
            WHERE updated > FROM_UNIXTIME(%s)

            UNION ALL
            SELECT ticket_id, lastupdate AS ts
            FROM ost_ticket
            WHERE lastupdate IS NOT NULL AND lastupdate > FROM_UNIXTIME(%s)

            UNION ALL
            SELECT t.ticket_id, e.updated AS ts
            FROM ost_ticket t
            JOIN ost_thread th ON t.ticket_id = th.object_id
            JOIN ost_thread_entry e ON th.id = e.thread_id
            WHERE th.object_type = 'T'
              AND e.updated > FROM_UNIXTIME(%s)

            UNION ALL
            SELECT t.ticket_id, e.created AS ts
            FROM ost_ticket t
            JOIN ost_thread th ON t.ticket_id = th.object_id
            JOIN ost_thread_entry e ON th.id = e.thread_id
            WHERE th.object_type = 'T'
              AND e.created > FROM_UNIXTIME(%s)
        ) x
        GROUP BY ticket_id
        ORDER BY max_activity_ts ASC
        """,
        (last_activity_ts, last_activity_ts, last_activity_ts, last_activity_ts),
    )

    changed = cursor.fetchall()
    ticket_ids_to_process = [int(r["ticket_id"]) for r in changed]
    max_seen_activity_ts = last_activity_ts
    if changed:
        max_seen_activity_ts = max(int(r["max_activity_ts"] or 0) for r in changed)

    if not ticket_ids_to_process:
        print("✅ No new/updated tickets since watermark.")
        # advance watermark (prevents re-scanning same time window)
        run_finished_ts = int(time.time())
        state["last_activity_ts"] = max(run_finished_ts, max_seen_activity_ts)
        save_state(args.state_file, state)
        print(f"🧾 State updated: last_activity_ts={state['last_activity_ts']}")
        return

    print(f"🧾 Tickets to reindex: {len(ticket_ids_to_process)}")

    # 2) Pull full threads for those ticket_ids and rebuild chunks
    placeholders = ",".join(["%s"] * len(ticket_ids_to_process))
    cursor.execute(
        f"""
        SELECT t.ticket_id, t.number, c.subject, e.body, e.poster, e.created
        FROM ost_ticket t
        JOIN ost_ticket__cdata c ON t.ticket_id = c.ticket_id
        JOIN ost_thread th ON t.ticket_id = th.object_id
        JOIN ost_thread_entry e ON th.id = e.thread_id
        WHERE th.object_type = 'T'
          AND e.body != ''
          AND t.ticket_id IN ({placeholders})
        ORDER BY t.ticket_id, e.created ASC
        """,
        tuple(ticket_ids_to_process),
    )

    tickets_data = defaultdict(lambda: {"subject": "", "ticket_number": "", "full_thread": ""})

    for row in cursor.fetchall():
        subject = row.get("subject") or ""
        if subject and any(k in subject.lower() for k in JUNK_KEYWORDS):
            continue

        t_id = int(row["ticket_id"])
        t_number = row.get("number") or str(t_id)

        body_clean = clean_text(row.get("body"))
        if len(body_clean) < 20:
            continue

        if not tickets_data[t_id]["subject"]:
            tickets_data[t_id]["subject"] = subject
            tickets_data[t_id]["ticket_number"] = t_number
            tickets_data[t_id]["full_thread"] = f"Subject: {subject}\n"

        tickets_data[t_id]["full_thread"] += f"\n--- Post by {row.get('poster')} ---\n{body_clean}\n"

    # Build insert arrays (same shape as 20_load_to_milvus.py)
    all_ticket_ids = []
    all_ticket_numbers = []
    all_source_types = []
    all_chunk_indexes = []
    all_subjects = []
    all_payloads = []

    for t_id, data in tickets_data.items():
        t_number = data["ticket_number"] or str(t_id)
        subject = data["subject"] or ""
        full_thread_redacted = redact_secrets(data["full_thread"])
        chunks = text_splitter.split_text(full_thread_redacted)

        for chunk_index, chunk in enumerate(chunks):
            all_ticket_ids.append(int(t_id))
            all_ticket_numbers.append(str(t_number))
            all_source_types.append("ticket")
            all_chunk_indexes.append(int(chunk_index))
            all_subjects.append(subject)
            all_payloads.append(chunk)

    # 3) Optional FAQ incremental (ID-based, unchanged)
    max_seen_faq_id = last_faq_id
    if args.include_faq:
        print(f"🔍 Fetching FAQs with faq_id > {last_faq_id} ...")
        cursor.execute(
            """
            SELECT faq_id, question, answer
            FROM ost_faq
            WHERE ispublished = 1
              AND faq_id > %s
            ORDER BY faq_id ASC
            """,
            (last_faq_id,),
        )
        for row in cursor.fetchall():
            faq_id = int(row["faq_id"])
            max_seen_faq_id = max(max_seen_faq_id, faq_id)

            question = row.get("question") or ""
            answer = clean_text(row.get("answer"))

            full_text = redact_secrets(f"FAQ: {question}\n\n{answer}")
            chunks = text_splitter.split_text(full_text)

            for chunk_index, chunk in enumerate(chunks):
                all_ticket_ids.append(faq_id + 100000)
                all_ticket_numbers.append(f"FAQ-{faq_id}")
                all_source_types.append("faq")
                all_chunk_indexes.append(int(chunk_index))
                all_subjects.append(question)
                all_payloads.append(chunk)

    if not all_payloads:
        print("No chunks to insert after filtering/cleaning.")
        run_finished_ts = int(time.time())
        state["last_activity_ts"] = max(run_finished_ts, max_seen_activity_ts)
        if args.include_faq:
            state["last_faq_id"] = int(max_seen_faq_id)
        save_state(args.state_file, state)
        print(f"State updated: last_activity_ts={state['last_activity_ts']}")
        return

    print(f"Encoding {len(all_payloads)} chunks using Ollama ({embed_model})...")

    all_vectors = []
    bs = int(args.batch_size)
    for i in range(0, len(all_payloads), bs):
        batch_texts = all_payloads[i : i + bs]
        vectors = get_embeddings_from_ollama(ollama, embed_model, batch_texts)
        all_vectors.extend(vectors)
        print(f"Embedded {min(i + bs, len(all_payloads))}/{len(all_payloads)}")

    if args.dry_run:
        print("Dry-run: skipping Milvus delete+insert.")
        return

    # UPDATE semantics: delete existing ticket vectors, then insert rebuilt vectors
    delete_ticket_rows_bulk(collection, ticket_ids=list(tickets_data.keys()), source_type="ticket")

    data = [
        all_ticket_ids,
        all_ticket_numbers,
        all_source_types,
        all_chunk_indexes,
        all_subjects,
        all_payloads,
        all_vectors,
    ]

    print("Inserting into Milvus...")
    collection.insert(data)
    collection.flush()

    # Update state only after successful flush
    run_finished_ts = int(time.time())
    state["last_activity_ts"] = max(run_finished_ts, max_seen_activity_ts)
    if args.include_faq:
        state["last_faq_id"] = int(max_seen_faq_id)
    save_state(args.state_file, state)

    print(f"Success! Total Entities in Milvus: {collection.num_entities}")
    print(f"State updated: last_activity_ts={state['last_activity_ts']}, last_faq_id={state.get('last_faq_id', 0)}")


if __name__ == "__main__":
    main()
