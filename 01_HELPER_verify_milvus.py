# Helper: Verify connection to Milvus GPU server

from pymilvus import connections, utility
from dotenv import load_dotenv
import os

load_dotenv()
SERVER_IP = os.getenv('SERVER_IP')

def test_connection():
    try:
        connections.connect(host=SERVER_IP, port='19530')
        version = utility.get_server_version()
        print(f"Connected to Milvus server at {SERVER_IP}")
        print(f"Milvus Version: {version}")

    except Exception as e:
        print(f"Connection error: {e}")
if __name__ == "__main__":
    test_connection()