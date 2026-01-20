from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from dotenv import load_dotenv
import os

load_dotenv()
SERVER_IP = os.getenv("SERVER_IP")

connections.connect(host=SERVER_IP, port="19530")

collection_name = "osticket_knowledge"

# Drop only when explicitly requested
if utility.has_collection(collection_name):
    utility.drop_collection(collection_name)
    print("Dropped existing collection.")

fields = [
    FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),

    # Ticket identifiers / metadata (recommended)
    FieldSchema(name="ticket_id", dtype=DataType.INT64),
    FieldSchema(name="ticket_number", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=16),  # "ticket" | "faq"
    FieldSchema(name="chunk_index", dtype=DataType.INT64),

    # Content
    FieldSchema(name="subject", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="text_payload", dtype=DataType.VARCHAR, max_length=65535),

    # Embedding
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
]

schema = CollectionSchema(fields, description="Knowledge base from osTicket tickets and FAQs")
collection = Collection(name=collection_name, schema=schema)

index_params = {
    "metric_type": "COSINE",          # recommended unless you normalize and want IP
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024},        # adjust based on data size; 128 is OK for small corpora
}
collection.create_index(field_name="vector", index_params=index_params)

print(f"Collection '{collection_name}' created/updated.")