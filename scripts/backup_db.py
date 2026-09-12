#!/usr/bin/env python3
"""Task 16: Weekly backup of weather.duckdb.

DuckDB has no `.dump`/sqlite3-style dump tool built for this the way SQLite
does; the straightforward, DuckDB-appropriate approach is:
  1. CHECKPOINT the live database (flushes any WAL content into the main
     .duckdb file, so a plain file copy is a consistent snapshot).
  2. Copy the .duckdb file to backups/ with a dated filename.
  3. gzip it (DuckDB files compress well -- mostly numeric columns).

Retention: data itself is kept for 10 years per project policy (no pruning
job), but that's a *data* retention decision, not a *backup-file* retention
decision -- weekly full-file backups kept forever would grow unbounded
disk usage completely separately from the 10-year data policy. Default
here keeps the last 12 weekly backups (~3 months) so recovery is possible
from a recent point without unbounded backup growth. Adjust
BACKUP_RETENTION_COUNT if you want a different window (or a mix of
weekly+monthly retention -- flagged as a possible improvement, not built
here to keep this task simple).
"""
import gzip
import os
import shutil
import sys
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import DB_PATH, get_connection

BACKUP_DIR = "/home/firescar96/.hermes/projects/sailing-weather/backups"
BACKUP_RETENTION_COUNT = 12


def checkpoint():
    con = get_connection()
    try:
        con.execute("CHECKPOINT")
    finally:
        con.close()


def make_backup():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_path = os.path.join(BACKUP_DIR, f"weather_{ts}.duckdb.gz")
    with open(DB_PATH, "rb") as src, gzip.open(out_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return out_path


def prune_old_backups():
    backups = sorted(
        (f for f in os.listdir(BACKUP_DIR) if f.startswith("weather_") and f.endswith(".duckdb.gz")),
    )
    to_remove = backups[:-BACKUP_RETENTION_COUNT] if len(backups) > BACKUP_RETENTION_COUNT else []
    for f in to_remove:
        os.remove(os.path.join(BACKUP_DIR, f))
    return to_remove


def main():
    checkpoint()
    out_path = make_backup()
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    removed = prune_old_backups()
    print(f"Backup written: {out_path} ({size_mb:.2f} MB)")
    if removed:
        print(f"Pruned {len(removed)} old backup(s) beyond retention count {BACKUP_RETENTION_COUNT}: {removed}")


if __name__ == "__main__":
    sys.exit(main())
