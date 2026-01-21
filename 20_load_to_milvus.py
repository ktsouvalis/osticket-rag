# populates Milvus the first time
# recreates/resets the collection (after schema changes, chunking changes, redaction changes, embedding model changes, etc.)
# recovers from a bad/partial index

import mysql.connector
from bs4 import BeautifulSoup
from pymilvus import connections, Collection
from ollama import Client
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import re
import hashlib
import json
import time
import errno
from collections import defaultdict

# 1. Setup
STATE_FILE_DEFAULT = "state/.milvus_update_state.json"
os.makedirs(os.path.dirname(STATE_FILE_DEFAULT), exist_ok=True)
load_dotenv()
SERVER_IP = os.getenv('SERVER_IP')

client = Client(host=f"http://{SERVER_IP}:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL_NAME", "bge-m3")
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
    embeddings = resp.get("embeddings")
    if not embeddings:
        raise RuntimeError("Embedding probe failed: no embeddings returned.")
    vec = embeddings if isinstance(embeddings[0], float) else embeddings[0]
    dim = get_collection_vector_dim(collection_obj)
    if len(vec) != dim:
        raise RuntimeError(f"Embedding dim {len(vec)} does not match collection dim {dim} for model '{model}'.")

print(f"Using Ollama model '{EMBED_MODEL}' for embeddings...")

connections.connect(host=SERVER_IP, port='19530')
collection = Collection("osticket_knowledge")
ensure_embedding_dim(client, EMBED_MODEL, collection)

db = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)
cursor = db.cursor(dictionary=True)

JUNK_KEYWORDS = ["backup failed", "vzdump", "unifi controller", "cron", "alert", "status", "successful backup"]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

def clean_text(html_content):
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    for img in soup.find_all('img'):
        img.decompose()
    text = soup.get_text(separator='\n')
    patterns_to_cut = [
        r"From:.*", r"Sent:.*", r"---.*Forwarded Message.*---",
        r"Προωθημένο μήνυμα", r"On.*wrote:.*", r"Στις.*έγραψε:.*"
    ]
    for pattern in patterns_to_cut:
        text = re.split(pattern, text, flags=re.IGNORECASE | re.MULTILINE)[0]
    return re.sub(r'\n\s*\n', '\n\n', text).strip()

def redact_secrets(text: str) -> str:
    """
    Redact common credential patterns BEFORE embedding/storing.
    Extend as you discover more patterns in your tickets.
    """
    if not text:
        return ""
    patterns = [
        # "admin / password" style
        r"(\b(?:admin|root|netadmin|user)\b)\s*/\s*([^\s\)]+)",
        # "password: ..."
        r"(?i)(password\s*:\s*)(\S+)",
        # "only password: ..."
        r"(?i)(only\s+password\s*:\s*)(\S+)",
    ]
    redacted = text
    redacted = re.sub(patterns[0], r"\1 / [REDACTED]", redacted)
    redacted = re.sub(patterns[1], r"\1[REDACTED]", redacted)
    redacted = re.sub(patterns[2], r"\1[REDACTED]", redacted)
    return redacted


def stable_int64(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def build_insert_data(collection_obj, field_data: dict[str, list]) -> list[list]:
    data = []
    for field in collection_obj.schema.fields:
        if getattr(field, "auto_id", False):
            continue
        if field.name not in field_data:
            raise RuntimeError(f"Missing data for field '{field.name}' required by collection schema.")
        data.append(field_data[field.name])
    return data


def save_state(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    try:
        os.replace(tmp, path)
    except OSError as exc:
        if exc.errno != errno.EBUSY:
            raise
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True)
        os.remove(tmp)

def ensure_state_file(path: str) -> None:
    """
    Ensure `path` is a JSON file.
    - If it doesn't exist -> create it.
    """
    if not os.path.exists(path):
        save_state(path, {"last_activity_ts": 0, "last_faq_id": 0})
    if not os.path.isfile(path):
        raise RuntimeError(f"State file was not created: {path}")

def get_embeddings_from_ollama(texts):
    response = client.embed(model=EMBED_MODEL, input=texts)
    return response['embeddings']

def process_and_load():
    ensure_state_file(STATE_FILE_DEFAULT)
    print("STATE PATH:", STATE_FILE_DEFAULT)
    print("STATE EXISTS:", os.path.exists(STATE_FILE_DEFAULT), "ISFILE:", os.path.isfile(STATE_FILE_DEFAULT))
    # Must match schema order (excluding pk auto_id):
    # ticket_id, ticket_number, source_type, chunk_index, subject, text_payload, vector
    all_ticket_ids = []
    all_ticket_numbers = []
    all_source_types = []
    all_chunk_indexes = []
    all_subjects = []
    all_payloads = []
    all_pks = []

    print("🔍 Fetching and Grouping Ticket Threads...")
    cursor.execute("""
        SELECT t.ticket_id, t.number, c.subject, e.body, e.poster
        FROM ost_ticket t
        JOIN ost_ticket__cdata c ON t.ticket_id = c.ticket_id
        JOIN ost_thread th ON t.ticket_id = th.object_id
        JOIN ost_thread_entry e ON th.id = e.thread_id
        WHERE th.object_type = 'T' AND e.body != ''
        ORDER BY t.ticket_id, e.created ASC
    """)

    tickets_data = defaultdict(lambda: {"subject": "", "ticket_number": "", "full_thread": ""})

    for row in cursor.fetchall():
        subject = row.get('subject') or ""
        if subject and any(k in subject.lower() for k in JUNK_KEYWORDS):
            continue

        t_id = row['ticket_id']
        t_number = row.get('number') or str(t_id)

        body_clean = clean_text(row.get('body'))
        if len(body_clean) < 20:
            continue

        if not tickets_data[t_id]["subject"]:
            tickets_data[t_id]["subject"] = subject
            tickets_data[t_id]["ticket_number"] = t_number
            tickets_data[t_id]["full_thread"] = f"Subject: {subject}\n"
        tickets_data[t_id]["full_thread"] += f"\n--- Post by {row.get('poster')} ---\n{body_clean}\n"

    # Tickets -> chunks
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
            all_pks.append(stable_int64(f"ticket:{t_id}:{chunk_index}"))

    # FAQs -> chunks
    max_seen_faq_id = 0
    cursor.execute("SELECT faq_id, question, answer FROM ost_faq WHERE ispublished = 1")
    for row in cursor.fetchall():
        faq_id = int(row["faq_id"])
        max_seen_faq_id = max(max_seen_faq_id, faq_id)
        question = row.get("question") or ""
        answer = clean_text(row.get("answer"))

        full_text = redact_secrets(f"FAQ: {question}\n\n{answer}")
        chunks = text_splitter.split_text(full_text)

        for chunk_index, chunk in enumerate(chunks):
            all_ticket_ids.append(faq_id + 100000)         # keep your existing FAQ ID scheme
            all_ticket_numbers.append(f"FAQ-{faq_id}")     # fits VARCHAR(32)
            all_source_types.append("faq")
            all_chunk_indexes.append(int(chunk_index))
            all_subjects.append(question)
            all_payloads.append(chunk)
            all_pks.append(stable_int64(f"faq:{faq_id}:{chunk_index}"))

    if not all_payloads:
        print("No payloads to embed/insert.")
        return

    print(f"🧠 Encoding {len(all_payloads)} chunks using Ollama ({EMBED_MODEL})...")

    batch_size = 100
    all_vectors = []
    for i in range(0, len(all_payloads), batch_size):
        batch_texts = all_payloads[i: i + batch_size]
        vectors = get_embeddings_from_ollama(batch_texts)
        all_vectors.extend(vectors)
        print(f"Processed {min(i + batch_size, len(all_payloads))}/{len(all_payloads)}...")

    field_data = {
        "pk": all_pks,
        "ticket_id": all_ticket_ids,
        "ticket_number": all_ticket_numbers,
        "source_type": all_source_types,
        "chunk_index": all_chunk_indexes,
        "subject": all_subjects,
        "text_payload": all_payloads,
        "vector": all_vectors,
    }
    data = build_insert_data(collection, field_data)

    collection.insert(data)
    collection.flush()
    print(f"Success! Total Entities in Milvus: {collection.num_entities}")
    cursor.execute(
        """
        SELECT UNIX_TIMESTAMP(MAX(ts)) AS max_activity_ts
        FROM (
            SELECT updated AS ts FROM ost_ticket
            UNION ALL
            SELECT lastupdate AS ts FROM ost_ticket WHERE lastupdate IS NOT NULL
            UNION ALL
            SELECT updated AS ts FROM ost_thread_entry
            UNION ALL
            SELECT created AS ts FROM ost_thread_entry
        ) x
        """
    )
    row = cursor.fetchone() or {}
    max_activity_ts = int(row.get("max_activity_ts") or 0)
    state = {
        "last_activity_ts": max(int(time.time()), max_activity_ts),
        "last_faq_id": int(max_seen_faq_id),
    }
    save_state(STATE_FILE_DEFAULT, state)
    print(f"State updated: last_activity_ts={state['last_activity_ts']}, last_faq_id={state['last_faq_id']}")

if __name__ == "__main__":
    process_and_load()
