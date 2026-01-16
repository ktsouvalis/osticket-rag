# Helper: Performs a vector search in Milvus using an Ollama embedding for the query,
# then fetches and displays the matching chunks along with their full threads from MySQL.

import os
import mysql.connector
from dotenv import load_dotenv
from pymilvus import connections, Collection
from ollama import Client

load_dotenv()


def _ensure_vector_batch(embeddings):
    """
    Ollama may return either:
      - [float, float, ...] for a single input string
      - [[float, ...], [float, ...], ...] for a list of inputs
    Milvus expects: [[float, ...]]
    """
    if embeddings and isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], float):
        return [embeddings]
    return embeddings


def get_full_ticket(cursor, ticket_id: int):
    """
    Fetch the complete thread for a ticket (or FAQ) from MySQL (debug only).
    """
    if ticket_id > 100000:
        faq_id = ticket_id - 100000
        cursor.execute(
            "SELECT question AS subject, answer AS body FROM ost_faq WHERE faq_id = %s",
            (faq_id,),
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {"subject": row.get("subject") or "", "body": row.get("body") or "", "ticket_number": f"FAQ-{faq_id}"}

    cursor.execute(
        """
        SELECT t.number, c.subject, e.body, e.poster, e.created
        FROM ost_ticket t
        JOIN ost_ticket__cdata c ON t.ticket_id = c.ticket_id
        JOIN ost_thread th ON t.ticket_id = th.object_id
        JOIN ost_thread_entry e ON th.id = e.thread_id
        WHERE t.ticket_id = %s AND e.body != ''
        ORDER BY e.created ASC
        """,
        (ticket_id,),
    )
    results = cursor.fetchall()
    if not results:
        return None

    ticket_number = results[0].get("number") or str(ticket_id)
    subject = results[0].get("subject") or ""

    full_body = ""
    for row in results:
        created = row.get("created")
        poster = row.get("poster") or "unknown"
        body = row.get("body") or ""
        full_body += f"\n--- [{created}] {poster} ---\n{body}\n"

    return {"subject": subject, "body": full_body, "ticket_number": ticket_number}


def search_and_fetch(collection: Collection, client: Client, cursor, query_text: str, *, limit: int, nprobe: int):
    embed_model = os.getenv("EMBED_MODEL_NAME", "bge-m3")

    print(f"Encoding query with Ollama model: {embed_model}")
    resp = client.embed(model=embed_model, input=query_text)
    q_vec = _ensure_vector_batch(resp.get("embeddings"))

    search_params = {"metric_type": "COSINE", "params": {}}

    results = collection.search(
        data=q_vec,
        anns_field="vector",
        param=search_params,
        limit=int(limit),
        output_fields=[
            "pk",
            "ticket_id",
            "ticket_number",
            "source_type",
            "chunk_index",
            "subject",
            "text_payload",
        ],
    )

    base_ticket_url = (os.getenv("BASE_TICKET_URL") or "").strip()

    for hits in results:
        for hit in hits:
            ent = hit.entity

            pk = ent.get("pk")
            ticket_id = ent.get("ticket_id")
            ticket_number = ent.get("ticket_number")
            source_type = ent.get("source_type")
            chunk_index = ent.get("chunk_index")
            subject = ent.get("subject") or ""
            payload = ent.get("text_payload") or ""
            score = float(hit.distance)

            print("\n" + "=" * 90)
            print(f"Match score (COSINE): {score:.4f}")
            print(f"Milvus: pk={pk} ticket_id={ticket_id} source_type={source_type} ticket_number={ticket_number} chunk_index={chunk_index}")
            if subject:
                print(f"Subject: {subject}")

            if base_ticket_url and isinstance(ticket_id, int) and ticket_id < 100000 and (source_type or "").lower() == "ticket":
                print(f"URL: {base_ticket_url}{ticket_id}")

            print("-" * 35 + " MILVUS CHUNK " + "-" * 35)
            print(payload)

            # Debug: show full MySQL thread/FAQ
            full_data = get_full_ticket(cursor, int(ticket_id)) if ticket_id is not None else None
            if full_data:
                print("-" * 33 + " FULL MYSQL THREAD " + "-" * 33)
                body = full_data.get("body") or ""
                print(body[:1500] + ("..." if len(body) > 1500 else ""))
            print("=" * 90)


def main():
    server_ip = os.getenv("SERVER_IP")
    if not server_ip:
        raise RuntimeError("SERVER_IP is not set in .env")

    nprobe = int(os.getenv("RAG_NPROBE", "10"))
    limit = int(os.getenv("SEARCH_LIMIT", "3"))

    # Milvus
    connections.connect(host=server_ip, port="19530")
    collection = Collection("osticket_knowledge")
    collection.load()
    print(f"Connected to Milvus. Total chunks: {collection.num_entities}")

    # Ollama
    client = Client(host=f"http://{server_ip}:11434")

    # MySQL
    db = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
    )
    cursor = db.cursor(dictionary=True)

    q = input("Search query: ").strip()
    if not q:
        print("Empty query.")
        return

    search_and_fetch(collection, client, cursor, q, limit=limit, nprobe=nprobe)


if __name__ == "__main__":
    main()