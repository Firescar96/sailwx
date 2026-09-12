#!/usr/bin/env python3
"""Ingest KBOS (Logan International) METAR observations.

Source: https://aviationweather.gov/api/data/metar?ids=KBOS&format=json&hours=N
Confirmed live 2026-09-05. Standard hourly METAR + special obs.
wspd/wgst are knots (kept as knots -- our canonical wind unit); temp/dewp
are Celsius -> converted to Fahrenheit (our canonical temperature unit,
confirmed 2026-09-09); altim is hPa.
"""
import json
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

URL = "https://aviationweather.gov/api/data/metar?ids=KBOS&format=json&hours=3"
LOCATION_ID = "kbos"

ctx = ssl.create_default_context()


def fetch_json():
    req = urllib.request.Request(URL, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    obs = fetch_json()
    con = get_connection()
    inserted = 0
    try:
        for ob in obs:
            report_time = ob.get("reportTime")
            if not report_time:
                continue
            ts = datetime.strptime(report_time, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)

            row = {}
            if ob.get("wdir") is not None and isinstance(ob["wdir"], (int, float)):
                row["wind_dir_deg"] = ob["wdir"]
            if ob.get("wspd") is not None:
                row["wind_speed_kt"] = ob["wspd"]
            if ob.get("wgst") is not None:
                row["wind_gust_kt"] = ob["wgst"]
            if ob.get("altim") is not None:
                row["pressure_hpa"] = ob["altim"]
            if ob.get("temp") is not None:
                row["air_temp_f"] = round(ob["temp"] * 9 / 5 + 32, 1)

            for variable, value in row.items():
                con.execute(
                    """
                    INSERT INTO observations (location_id, ts_utc, variable, value, source)
                    VALUES (?, ?, ?, ?, 'metar')
                    ON CONFLICT DO NOTHING
                    """,
                    [LOCATION_ID, ts, variable, value],
                )
                inserted += 1
    finally:
        con.close()
    print(f"KBOS METAR: attempted {inserted} inserts across {len(obs)} obs")


if __name__ == "__main__":
    sys.exit(main())
