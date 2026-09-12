#!/usr/bin/env python3
"""Ingest NOAA CO-OPS station 8443970 (Boston, Long Wharf) observations.

CO-OPS Data API is one-product-per-call. Station 8443970 is primarily a
tide gauge: confirmed live 2026-09-05 that `wind` and `water_temperature`
return "No data was found" at this station, but `air_temperature` and
`water_level` both work. If a wind sensor is ever added here, add
'wind' to PRODUCTS below and map wind speed/dir/gust similarly to the
NDBC ingester.

UNITS: this station's `air_temperature` was originally requested with
`units=english` (air_temp_f), matching most everything else, but was
briefly switched to `units=metric`/air_temp_c on 2026-09-09 during an
investigation into "no model accuracy at Long Wharf" -- root cause was
actually that Celsius was the ODD ONE OUT (every other observation
source, and forecasts, now use Fahrenheit, per user decision 2026-09-09
to standardize on Fahrenheit project-wide). Reverted back to
`units=english`/`air_temp_f` the same day once the canonical unit was
clarified -- this was in fact already correct before the brief
metric detour. Water level stays in feet (MLLW is a US-specific vertical
datum with no forecast counterpart to compare against anyway).
"""
import json
import ssl
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

BASE = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
STATION = "8443970"
LOCATION_ID = "coops_8443970"

# product -> our variable name
PRODUCTS = {
    "air_temperature": "air_temp_f",
    "water_level": "water_level_ft_mllw",
}
# Units requested per-product: CO-OPS' units= param applies to the whole
# call, and each product is its own call, so different products can use
# different unit systems safely.
PRODUCT_UNITS = {
    "air_temperature": "english",
    "water_level": "english",
}

ctx = ssl.create_default_context()


def fetch_json(product):
    params = {
        "station": STATION,
        "product": product,
        "date": "latest",
        "units": PRODUCT_UNITS.get(product, "metric"),
        "time_zone": "gmt",
        "format": "json",
    }
    if product == "water_level":
        params["datum"] = "MLLW"
    url = f"{BASE}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "sailing-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    con = get_connection()
    inserted = 0
    errors = []
    try:
        for product, variable in PRODUCTS.items():
            try:
                data = fetch_json(product)
            except Exception as e:
                errors.append(f"{product}: {e}")
                continue
            if "error" in data:
                errors.append(f"{product}: {data['error'].get('message')}")
                continue
            for point in data.get("data", []):
                # "t": "2026-09-05 23:24" (station-local requested as gmt)
                ts = datetime.strptime(point["t"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
                try:
                    value = float(point["v"])
                except (KeyError, ValueError):
                    continue
                con.execute(
                    """
                    INSERT INTO observations (location_id, ts_utc, variable, value, source)
                    VALUES (?, ?, ?, ?, 'coops_api')
                    ON CONFLICT DO NOTHING
                    """,
                    [LOCATION_ID, ts, variable, value],
                )
                inserted += 1
    finally:
        con.close()
    print(f"CO-OPS 8443970: attempted {inserted} inserts")
    if errors:
        print("Errors:")
        for e in errors:
            print(" ", e)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
