#!/usr/bin/env python3
"""ONE-TIME backfill of historical CAPE data, covering this project's
existing flag-reading history.

User: "how many weeks before I can add cape data to linear regression
how much data needed" -- rather than answer with a guess based on
elapsed calendar time, this backfills REAL historical CAPE (confirmed
live: Open-Meteo's Historical Forecast API serves `cape` back to 2022,
same variable/units as the live forecast ingester) so the actual
answer (how many flag readings happened during meaningfully elevated
CAPE) can be measured directly today instead of waited for.

IMPORTANT DISTINCTION from the live forecast ingester and the NDA
backfill: the Historical Forecast API returns one STITCHED continuous
timeseries per model (each run's first few hours spliced onto the
previous one), not individual per-run point forecasts at varying lead
times. There is no "init_time" to recover here -- this is fundamentally
a different shape of data than forecast_runs/forecast_values was built
for (which assumes a real run + lead-time structure, used for e.g.
Forecast Stability's run-to-run comparison and lead-time-bucketed
accuracy). Writing this into forecast_runs under a fake init_time would
misrepresent it as something it isn't.

Instead, stored in forecast_values with a single dedicated per-model
forecast_runs row tagged model="<model>_historical_cape", init_time_utc
set to the window END (so date_diff('hour', init, valid) lead_hours
comes out negative/zero, visibly NOT a real forward-looking forecast
if anyone queries lead-time buckets against it) -- clearly distinct
from the live model keys ("gfs", "ecmwf", etc), won't pollute Forecast
Stability/Model Accuracy, won't be selectable from the model dropdowns,
and can be dropped/re-backfilled trivially since it's fully derived
data, isolated behind its own model-name prefix.
"""
import json
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import bulk_insert, get_connection

LOCATIONS = {
    "mit_pavilion": (42.3592, -71.0838),
    "ndbc_44013": (42.346, -70.651),
    "coops_8443970": (42.3539, -71.0503),
    "kbos": (42.3656, -71.0096),
    "cbi_dockhouse": (42.3598, -71.0731),
    "courageous_sailing": (42.3715, -71.0509),
}

MODEL_SLUGS = {
    "gfs": "gfs_seamless",
    "ecmwf": "ecmwf_ifs",
    "hrrr": "gfs_hrrr",
    "icon": "icon_seamless",
    "nam": "ncep_nam_conus",
    "hrdps": "gem_hrdps_continental",
}

ctx = ssl.create_default_context()


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return json.loads(resp.read())

def get_historical_cape(lat, lon, model_slug, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cape",
        "models": model_slug,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "GMT",
    }
    url = f"https://historical-forecast-api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
    return fetch_json(url)


def main():
    con = get_connection()
    start_date = "2026-09-08"
    end_date = "2026-10-19"  # inclusive-ish, matches real flag history window + 1 day buffer
    window_end = datetime(2026, 10, 19, tzinfo=timezone.utc)
    total_rows = 0
    try:
        for location_id, (lat, lon) in LOCATIONS.items():
            for model_key, slug in MODEL_SLUGS.items():
                hist_model_key = f"{model_key}_historical_cape"
                existing = con.execute(
                    "SELECT run_id FROM forecast_runs WHERE model = ? AND location_id = ? AND init_time_utc = ?",
                    [hist_model_key, location_id, window_end],
                ).fetchone()
                if existing:
                    print(f"  {location_id}/{model_key}: already backfilled, skipping")
                    continue

                data = get_historical_cape(lat, lon, slug, start_date, end_date)
                hourly = data.get("hourly", {})
                times = hourly.get("time", [])
                values = hourly.get("cape", [])
                if not times:
                    print(f"  {location_id}/{model_key}: no data returned, skipping")
                    continue

                run_id = con.execute(
                    """
                    INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
                    VALUES (?, ?, ?, ?)
                    RETURNING run_id
                    """,
                    [hist_model_key, location_id, window_end, datetime.now(timezone.utc)],
                ).fetchone()[0]

                rows = []
                for t_str, v in zip(times, values):
                    if v is None:
                        continue
                    valid_time = datetime.strptime(t_str, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
                    rows.append((run_id, valid_time, "cape_jkg", round(float(v), 2)))

                bulk_insert(con, "forecast_values", ["run_id", "valid_time_utc", "variable", "value"], rows)
                con.commit()
                total_rows += len(rows)
                print(f"  {location_id}/{model_key}: backfilled {len(rows)} historical cape_jkg rows (run_id={run_id})")

        print(f"\nTotal historical CAPE rows written: {total_rows}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
