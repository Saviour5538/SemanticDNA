"""Run schema.sql against the configured PostgreSQL database."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
import psycopg2

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Copy .env.example to .env and fill it in.")
    sys.exit(1)

schema_path = os.path.join(os.path.dirname(__file__), "..", "schema.sql")

try:
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    with open(schema_path) as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.close()
    print("Schema created successfully.")
except psycopg2.OperationalError as e:
    print(f"ERROR: Could not connect to database.\n{e}")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
