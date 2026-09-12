#!/usr/bin/env python3
"""Ingest GFS / ECMWF / HRRR forecasts (via Open-Meteo) for every location
we track observations for (mit_pavilion, ndbc_44013, coops_8443970, kbos) --
one forecast pull per location per model, so accuracy verification compares
each location's forecast against *that same location's* observations
(like-to-like), not one shared forecast point against 4 different physical
locations miles apart.

Each poll makes one call per (location, model) to the forecast API (max
horizon each), plus one call per model to Open-Meteo's *metadata* API to
get the exact model cycle/init time -- no guessing/approximating required
(see docs/plan.md Task 9 for how this was researched:
api.open-meteo.com/data/<meta_id>/static/meta.json returns
`last_run_initialisation_time` as a Unix timestamp, which is the true init
time of the run currently being served by the forecast endpoint for that
model). The metadata call is per-model only (not per-location) since a
model's cycle/init time is the same regardless of which grid point you're
reading -- no need to repeat it 4x.

IMPORTANT: the metadata API's internal model id is NOT always the same
string as the forecast API's `models=` slug. Confirmed live 2026-09-05:
  forecast slug      -> metadata id
  gfs_seamless        -> ncep_gfs013
  ecmwf_ifs            -> ecmwf_ifs
  gfs_hrrr              -> ncep_hrrr_conus
"""
import json
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import bulk_insert, get_connection

# Every location we pull forecasts for -- keep in sync with db/seed_locations.py.
# (lat, lon) match the same coordinates used to seed the locations table.
LOCATIONS = {
    "mit_pavilion": (42.3592, -71.0838),
    "ndbc_44013": (42.346, -70.651),
    "coops_8443970": (42.3539, -71.0503),
    "kbos": (42.3656, -71.0096),
    "cbi_dockhouse": (42.3598, -71.0731),
}

HOURLY_VARS = "wind_speed_10m,wind_gusts_10m,wind_direction_10m,pressure_msl,temperature_2m"

# model key -> (forecast API models= slug, metadata API id, extra forecast params)
# past_days=2 added 2026-09-10: Open-Meteo's default is to return data only
# from "today 00:00 UTC" forward, even if the model's own init_time (and
# our poll) happened hours earlier -- e.g. an 18:00 UTC run polled the same
# evening would return nothing for 18:00-23:59, silently truncating the
# start of every run. past_days backfills a couple of trailing days so the
# "Also show past" dashboard feature actually has data to display, and so
# the very start of each run (right after init_time) isn't clipped off.
#
# icon/hrdps/nam added 2026-09-10 (user request, after investigating a
# Windy API integration that turned out to only offer TEST/randomized data
# on the available key -- abandoned, see docs/plan.md). These 3 give
# genuinely distinct wind values from GFS/HRRR at CBI's coordinate
# (confirmed live), unlike Open-Meteo's own HRRR ("gfs_hrrr" slug), which
# blends into GFS at that exact spot and shows no distinct HRRR signal
# there. Real forecast horizons are much shorter than their forecast_days
# request would suggest -- Open-Meteo pads the rest with nulls, which our
# ingester already skips (`if v is None: continue`), so no explicit
# forecast_days/forecast_hours tuning is needed beyond a generous request:
#   icon_seamless          -- real data ~168h (7 days)
#   ncep_nam_conus         -- real data ~60h (2.5 days)
#   gem_hrdps_continental  -- real data ~42h (~1.75 days)
#
# HRRR PAST-DATA BUG (found 2026-09-10): `forecast_hours` and `past_days`
# are MUTUALLY EXCLUSIVE on Open-Meteo's API -- when both are set, past_days
# is silently ignored (confirmed live: request returned zero hours before
# "now" despite past_days=2 being set). Every other model here uses
# forecast_days (not forecast_hours), which DOES combine fine with
# past_days -- HRRR is the only one using forecast_hours (to avoid pulling
# HRRR's blended-with-GFS tail beyond its real ~48h horizon), which is what
# created the conflict. Fixed by using `past_hours` instead of `past_days`
# for HRRR specifically -- same unit family as forecast_hours, and doesn't
# trigger the exclusivity conflict. This bug meant HRRR's "Also show past"
# dashboard overlay showed literally zero historical data, only future --
# user-reported 2026-09-10 ("i am only seeing future data").
MODELS = {
    "gfs": ("gfs_seamless", "ncep_gfs013", {"forecast_days": "16", "past_days": "2"}),
    "ecmwf": ("ecmwf_ifs", "ecmwf_ifs", {"forecast_days": "15", "past_days": "2"}),
    "hrrr": ("gfs_hrrr", "ncep_hrrr_conus", {"forecast_hours": "48", "past_hours": "48"}),
    "icon": ("icon_seamless", "dwd_icon", {"forecast_days": "16", "past_days": "2"}),
    "nam": ("ncep_nam_conus", "ncep_nam_conus", {"forecast_days": "16", "past_days": "2"}),
    "hrdps": ("gem_hrdps_continental", "cmc_gem_hrdps", {"forecast_days": "16", "past_days": "2"}),
}

# Open-Meteo variable name -> our observations/forecast_values variable name.
# IMPORTANT: must match the units observations actually use (knots, hPa,
# Fahrenheit, degrees) so Phase 4's verification queries can compare
# forecast vs. observed directly without a unit-conversion step at query
# time. Open-Meteo defaults to km/h for wind and Celsius for temp --
# converted to knots and Fahrenheit (canonical units for this project,
# confirmed 2026-09-09) below.
VARIABLE_MAP = {
    "wind_speed_10m": "wind_speed_kt",
    "wind_gusts_10m": "wind_gust_kt",
    "wind_direction_10m": "wind_dir_deg",
    "pressure_msl": "pressure_hpa",
    "temperature_2m": "air_temp_f",
}
KMH_TO_KT = 0.539957
# Which of our variable names need the km/h -> knots conversion applied.
WIND_SPEED_VARS = {"wind_speed_kt", "wind_gust_kt"}
# Which of our variable names need the Celsius -> Fahrenheit conversion applied.
TEMP_VARS = {"air_temp_f", "water_temp_f"}

ctx = ssl.create_default_context()


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_init_time(meta_id):
    """Returns the current run's init time as a UTC datetime, via the
    metadata API (exact, not approximated). Same for every location -- a
    model's cycle time doesn't vary by grid point."""
    url = f"https://api.open-meteo.com/data/{meta_id}/static/meta.json"
    meta = fetch_json(url)
    ts = meta["last_run_initialisation_time"]
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def get_forecast(lat, lon, models_slug, extra_params):
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "models": models_slug,
        "timezone": "GMT",
    }
    params.update(extra_params)
    url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
    return fetch_json(url)


def ingest_model_location(con, model_key, models_slug, init_time, extra_params,
                           location_id, lat, lon, retrieved_at):
    data = get_forecast(lat, lon, models_slug, extra_params)

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        print(f"  {model_key}/{location_id}: no hourly data returned, skipping")
        return 0

    # Upsert forecast_runs row (UNIQUE constraint on model/location/init_time_utc
    # means a duplicate poll within the same cycle just no-ops here).
    existing = con.execute(
        "SELECT run_id FROM forecast_runs WHERE model = ? AND location_id = ? AND init_time_utc = ?",
        [model_key, location_id, init_time],
    ).fetchone()
    if existing:
        run_id = existing[0]
        print(f"  {model_key}/{location_id}: run for init_time {init_time} already ingested (run_id={run_id}), skipping values")
        return 0

    run_id = con.execute(
        """
        INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
        VALUES (?, ?, ?, ?)
        RETURNING run_id
        """,
        [model_key, location_id, init_time, retrieved_at],
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
            elif our_var in TEMP_VARS:
                value = value * 9 / 5 + 32
            rows.append((run_id, valid_time, our_var, round(value, 2)))

    bulk_insert(con, "forecast_values", ["run_id", "valid_time_utc", "variable", "value"], rows)
    print(f"  {model_key}/{location_id}: new run_id={run_id}, init_time={init_time}, inserted {len(rows)} forecast_values rows")
    return len(rows)


def main():
    retrieved_at = datetime.now(timezone.utc)
    con = get_connection()
    total = 0
    errors = []
    try:
        for model_key, (models_slug, meta_id, extra_params) in MODELS.items():
            try:
                init_time = get_init_time(meta_id)
            except Exception as e:
                errors.append(f"{model_key} (metadata): {e}")
                continue
            for location_id, (lat, lon) in LOCATIONS.items():
                try:
                    total += ingest_model_location(
                        con, model_key, models_slug, init_time, extra_params,
                        location_id, lat, lon, retrieved_at,
                    )
                except Exception as e:
                    errors.append(f"{model_key}/{location_id}: {e}")
    finally:
        con.close()

    print(f"Total new forecast_values rows: {total}")
    if errors:
        print("Errors:")
        for e in errors:
            print(" ", e)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())

