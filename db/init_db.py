#!/usr/bin/env python3
"""Idempotently create the sailing-weather DuckDB schema.

Run directly: python3 db/init_db.py
Safe to re-run — every statement in schema.sql uses IF NOT EXISTS.
"""
import sys

from common import SCHEMA_PATH, get_connection


def main():
    with open(SCHEMA_PATH) as f:
        schema_sql = f.read()

    con = get_connection()
    try:
        con.execute(schema_sql)
        tables = con.sql("SHOW TABLES").fetchall()
        print(f"Schema applied. Tables present: {[t[0] for t in tables]}")
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
