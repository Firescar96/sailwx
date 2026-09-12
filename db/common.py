"""Shared helpers for the sailing-weather DuckDB project.

Keeps DB path resolution and connection setup in one place so every
ingestion script (and init_db.py) agrees on where the database lives.
"""
import os
import re
from datetime import datetime

import duckdb

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(PROJECT_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "weather.duckdb")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")

_LOCK_PID_RE = re.compile(r"Conflicting lock is held in .* \(PID (\d+)\)")


def _pid_is_alive(pid: int) -> bool:
    """True if a process with this PID currently exists. Uses signal 0
    (no-op, just checks existence/permission) rather than SIGKILL/etc."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, just owned by someone else -- still alive
    else:
        return True


def get_connection(read_only: bool = False, retries: int = 5, retry_delay_s: float = 2.0) -> duckdb.DuckDBPyConnection:
    """Open a connection to the shared weather.duckdb file.

    DuckDB is single-writer: only one read/write connection may be open at
    a time; a second writer gets an immediate IOException, not a blocking
    wait (confirmed live 2026-09-05 -- two cron jobs fired in the same
    minute and the loser got "Could not set lock on file"). This kept
    recurring throughout this project's cron history (multiple ingesters
    scheduled at the exact same "0 */4 * * *" minute as each other),
    despite scripts being short-lived (run, write, exit within seconds) --
    so a genuine STUCK job (one that acquired the lock and then hung/
    crashed without releasing it, e.g. an unhandled network timeout deep
    in a fetch) would otherwise look IDENTICAL to a normal brief overlap
    from an unrelated cron job's timing, and both were previously handled
    the same dumb way: retry blindly and hope.

    IMPROVED 2026-09-11 (per user: "find a way to detect and better
    handle the stuck jobs"): DuckDB's lock error message names the PID
    holding the conflicting lock. We parse it out and check with
    os.kill(pid, 0) whether that process is still actually alive:
      - PID alive: a genuine, currently-running competing writer (the
        normal, expected case for brief cron overlaps) -- back off and
        retry as before.
      - PID dead: the lock is logically STALE (the holder already
        exited, e.g. crashed or was killed, without the OS having fully
        reclaimed the file lock yet, or a race where it exited between
        our stat and connect) -- this is the "stuck job" case. Retry
        immediately without the normal backoff delay (nothing legitimate
        is going to release it "soon" since nothing is actually running
        anymore), and print a clear diagnostic so whoever's watching cron
        output can tell "real contention" apart from "something died
        holding the lock" at a glance, instead of every failure looking
        like generic unexplained flakiness.

    IMPORTANT: DuckDB's default session TimeZone is the OS local zone (here,
    America/New_York), NOT UTC. Binding a timezone-aware Python datetime
    into a plain TIMESTAMP column silently converts it to that local zone
    before storing (found live 2026-09-05: an init_time_utc of 12:00 UTC
    was stored/read back as 08:00 -- exactly the EDT -4h offset). Every
    downstream lead-hour computation depends on these columns actually
    being UTC, so force the session timezone to UTC on every connection.
    """
    import time

    os.makedirs(DB_DIR, exist_ok=True)
    last_err = None
    for attempt in range(retries):
        try:
            con = duckdb.connect(DB_PATH, read_only=read_only)
            con.execute("SET TimeZone='UTC'")
            return con
        except duckdb.IOException as e:
            last_err = e
            if attempt >= retries - 1:
                break
            m = _LOCK_PID_RE.search(str(e))
            if m:
                holder_pid = int(m.group(1))
                if _pid_is_alive(holder_pid):
                    print(f"[get_connection] DB lock held by live PID {holder_pid} "
                          f"(attempt {attempt + 1}/{retries}) -- backing off {retry_delay_s}s")
                    time.sleep(retry_delay_s)
                else:
                    print(f"[get_connection] DB lock references DEAD PID {holder_pid} "
                          f"(attempt {attempt + 1}/{retries}) -- STALE LOCK, retrying immediately")
                    # no sleep: nothing alive is going to release this "soon"
            else:
                # Lock error without a parseable PID (unexpected format) --
                # fall back to the original blind-backoff behavior.
                time.sleep(retry_delay_s)
    raise last_err


def _sql_literal(value):
    """Render a Python value as a safe SQL literal for bulk INSERT VALUES.

    Only used by bulk_insert, which is for internal (non-user-supplied
    structure) ingestion rows -- values here are numbers, None, or strings
    from parsed government data feeds, not free-form user input. Strings
    are escaped by doubling single quotes (standard SQL escaping).
    """
    if value is None:
        return "NULL"
    if isinstance(value, datetime):
        return f"TIMESTAMP '{value.strftime('%Y-%m-%d %H:%M:%S')}'"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    # string / anything else -> quoted, single-quote-escaped
    s = str(value).replace("'", "''")
    return f"'{s}'"


def bulk_insert(con, table, columns, rows):
    """Fast multi-row INSERT for tens of thousands of rows.

    Measured: DuckDB's Python binding is slow both with con.executemany()
    (~3-4ms/row against a PK/FK-constrained table) AND with a single
    parameterized multi-row VALUES(...) statement using a `?` placeholder
    per value (tens of thousands of placeholders bind slowly too -- both
    hung well past a minute for NDBC's ~39k-row feed).

    What IS fast (~0.9s for 38,827 rows in local testing) is a single
    INSERT with literal values inlined directly into the VALUES clause.
    Since this path is only used for parsed government data-feed rows
    (numbers/timestamps/short strings, not arbitrary user input), literals
    are rendered via `_sql_literal` with standard SQL string escaping
    (doubled single quotes) -- safe for this data shape, and dramatically
    faster than parameter binding at this row count.
    """
    if not rows:
        return
    col_list = ", ".join(columns)
    row_literals = []
    for row in rows:
        row_literals.append("(" + ", ".join(_sql_literal(v) for v in row) + ")")
    values_clause = ", ".join(row_literals)
    con.execute(f"INSERT INTO {table} ({col_list}) VALUES {values_clause}")
