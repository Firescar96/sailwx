#!/usr/bin/env python3
"""Ingest NOAA NDBC buoy 44013 (Boston Approach Buoy) realtime observations.

Source: https://www.ndbc.noaa.gov/data/realtime2/44013.txt
Plain-text, whitespace-delimited, most-recent-first, ~10 min cadence.
Columns (confirmed live 2026-09-05):
  YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS PTDY TIDE
Missing values are "MM". WDIR/WSPD/GST are already m/s in this feed
(despite the header showing "m/s" for WSPD/GST) -> convert to knots
(our canonical wind unit). ATMP/WTMP are Celsius in this feed -> converted
to Fahrenheit (our canonical temperature unit, confirmed 2026-09-09).
"""
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import bulk_insert, get_connection

URL = "https://www.ndbc.noaa.gov/data/realtime2/44013.txt"
LOCATION_ID = "ndbc_44013"
MS_TO_KT = 1.943844

ctx = ssl.create_default_context()


def fetch_text(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse(text):
    """Yields (ts_utc: datetime, dict of variable->value)."""
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    for line in lines:
        parts = line.split()
        if len(parts) < 15:
            continue
        yy, mo, dd, hh, mm = parts[0:5]
        try:
            ts = datetime(int(yy), int(mo), int(dd), int(hh), int(mm), tzinfo=timezone.utc)
        except ValueError:
            continue

        def val(i):
            v = parts[i] if i < len(parts) else "MM"
            return None if v == "MM" else float(v)

        wdir = val(5)
        wspd = val(6)
        gst = val(7)
        pres = val(12)
        atmp = val(13)
        wtmp = val(14)

        row = {}
        if wdir is not None:
            row["wind_dir_deg"] = wdir
        if wspd is not None:
            row["wind_speed_kt"] = round(wspd * MS_TO_KT, 1)
        if gst is not None:
            row["wind_gust_kt"] = round(gst * MS_TO_KT, 1)
        if pres is not None:
            row["pressure_hpa"] = pres
        if atmp is not None:
            row["air_temp_f"] = round(atmp * 9 / 5 + 32, 1)
        if wtmp is not None:
            row["water_temp_f"] = round(wtmp * 9 / 5 + 32, 1)

        if row:
            yield ts, row


def main():
    text = fetch_text(URL)
    rows = []
    for ts, variables in parse(text):
        for variable, value in variables.items():
            rows.append([LOCATION_ID, ts, variable, value])

    con = get_connection()
    try:
        # Bulk-load into an unconstrained staging table first (fast bulk
        # insert, no PK/FK checks), then do a single set-based anti-join
        # insert into the real table.
        con.execute("CREATE OR REPLACE TEMP TABLE stage_ndbc (location_id TEXT, ts_utc TIMESTAMP, variable TEXT, value DOUBLE)")
        bulk_insert(con, "stage_ndbc", ["location_id", "ts_utc", "variable", "value"], rows)
        con.execute(
            """
            INSERT INTO observations (location_id, ts_utc, variable, value, source)
            SELECT s.location_id, s.ts_utc, s.variable, ANY_VALUE(s.value), 'ndbc_realtime2'
            FROM stage_ndbc s
            ANTI JOIN observations o
                ON o.location_id = s.location_id AND o.ts_utc = s.ts_utc AND o.variable = s.variable
            GROUP BY s.location_id, s.ts_utc, s.variable
            """
        )
        con.execute("DROP TABLE stage_ndbc")
    finally:
        con.close()
    print(f"NDBC 44013: processed feed, staged {len(rows)} variable-observation rows (new ones inserted, dupes skipped)")


if __name__ == "__main__":
    sys.exit(main())
