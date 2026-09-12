#!/usr/bin/env python3
"""One-off backfill for the ECMWF 12:00 UTC 2026-09-08 run, which was never
captured live (see docs/plan.md Task 11 correction: ECMWF's slower
dissemination lag meant this run was superseded before a poll caught it).

Uses Open-Meteo's Single Runs API (single-runs-api.open-meteo.com), which
retrieves the exact forecast issued at a specific historical init time --
unlike the regular forecast API, which only ever serves the current/latest
run. Confirmed live 2026-09-09.

This is a manual, one-off script for backfilling a SPECIFIC known gap, not
a general backfill tool -- per project policy (docs/plan.md Task 15) we
don't do broad historical backfills, only targeted fixes for gaps we've
actually found.
"""
import json
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import bulk_insert, get_connection

RUN_INIT_TIME = "2026-09-08T12:00"
MODEL_KEY = "ecmwf"
MODELS_SLUG = "ecmwf_ifs"
LOCATIONS = {
    "mit_pavilion": (42.3592, -71.0838),
    "ndbc_44013": (42.346, -70.651),
    "coops_8443970": (42.3539, -71.0503),
    "kbos": (42.3656, -71.0096),
}
HOURLY_VARS = "wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,temperature_2m"
VARIABLE_MAP = {
    "wind_speed_10m": "wind_speed_kt",
    "wind_gusts_10m": "wind_gust_kt",
    "wind_direction_10m": "wind_dir_deg",
    "pressure_msl": "pressure_hpa",
    "temperature_2m": "air_temp_c",
}
KMH_TO_KT = 0.539957
WIND_SPEED_VARS = {"wind_speed_kt", "wind_gust_kt"}

ctx = ssl.create_default_context()


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_single_run(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "models": MODELS_SLUG,
        "run": RUN_INIT_TIME,
        "timezone": "GMT",
    }
    url = f"https://single-runs-api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
    return fetch_json(url)


def main():
    init_time = datetime.strptime(RUN_INIT_TIME, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    retrieved_at = datetime.now(timezone.utc)

    con = get_connection()
    total = 0
    try:
        for location_id, (lat, lon) in LOCATIONS.items():
            existing = con.execute(
                "SELECT run_id FROM forecast_runs WHERE model = ? AND location_id = ? AND init_time_utc = ?",
                [MODEL_KEY, location_id, init_time],
            ).fetchone()
            if existing:
                print(f"  {location_id}: run already present (run_id={existing[0]}), skipping")
                continue

            data = get_single_run(lat, lon)
            hourly = data.get("hourly", {})
            times = hourly.get("time", [])
            if not times:
                print(f"  {location_id}: no hourly data returned from Single Runs API, skipping")
                continue

            run_id = con.execute(
                """
                INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
                VALUES (?, ?, ?, ?)
                RETURNING run_id
                """,
                [MODEL_KEY, location_id, init_time, retrieved_at],
            ).fetchone()[0]

            rows = []
            for om_var, values in hourly.items():
                if om_var == "time":
                    continue
                our_var = VARIABLE_MAP.get(om_var, om_var)
                for t_str, v in zip(times, values):
                    if v is None:
                        continue
                    valid_time = datetime.strptime(t_str, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                    value = float(v)
                    if our_var in WIND_SPEED_VARS:
                        value = value * KMH_TO_KT
                    rows.append((run_id, valid_time, our_var, round(value, 2)))

            bulk_insert(con, "forecast_values", ["run_id", "valid_time_utc", "variable", "value"], rows)
            total += len(rows)
            print(f"  {location_id}: backfilled run_id={run_id}, {len(rows)} forecast_values rows"
                  f" (valid_time range {times[0]} to {times[-1]})")
    finally:
        con.close()

    print(f"Total backfilled forecast_values rows: {total}")


if __name__ == "__main__":
    sys.exit(main())
