# Helper: Extract raw ticket data for debugging purposes

import mysql.connector
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

db = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST"),
    user=os.getenv("MYSQL_USER"),
    password=os.getenv("MYSQL_PASSWORD"),
    database=os.getenv("MYSQL_DATABASE")
)
cursor = db.cursor(dictionary=True)

def extract_ticket_for_debug(ticket_id):
    cursor.execute("""
        SELECT t.number, c.subject, e.body, e.poster, e.created
        FROM ost_ticket t
        JOIN ost_ticket__cdata c ON t.ticket_id = c.ticket_id
        JOIN ost_thread th ON t.ticket_id = th.object_id
        JOIN ost_thread_entry e ON th.id = e.thread_id
        WHERE t.ticket_id = %s
        ORDER BY e.created ASC
    """, (ticket_id,))

    rows = cursor.fetchall()
    print(f"\n=== RAW DATA FOR TICKET ID: {ticket_id} ===")
    for row in rows:
        # Εδώ το βγάζουμε ΧΩΡΙΣ καθαρισμό re.sub για να δούμε τη δομή
        print(f"\n--- Post by {row['poster']} on {row['created']} ---")
        print(row['body']) # Raw HTML/Text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract raw ticket data for debugging.")
    parser.add_argument("ticket_id", type=int, help="ID of the ticket to extract")
    args = parser.parse_args()

    extract_ticket_for_debug(args.ticket_id)