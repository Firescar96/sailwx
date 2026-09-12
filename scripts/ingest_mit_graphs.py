#!/usr/bin/env python3
"""MIT Sailing Pavilion pressure/air-temp/water-temp/wind-direction
extractor -- reads the station's weewx day-graph PNGs (same source as
extract_wind.py's wind speed/gust scraper), covering the fields that
were previously only captured as a single point-in-time snapshot via
ingest_mit_conditions.py.

WHY THIS EXISTS (superseding the current-conditions-page approach,
2026-09-10): the live "Current Conditions" page only gives ONE reading
per poll with no way to recover history after a missed cron run, and
polling that human-facing page every 15 minutes was needlessly
aggressive for a single point value. This script instead reads each
day-graph's full 24h rolling window every poll (same pattern as
extract_wind.py), so a missed run just gets backfilled by the next one's
overlapping window -- consistent with the "fill history buckets on the
next run" approach used elsewhere in this project. Runs every 4 hours
(see cron job), matching the other 4-hourly ingesters, instead of every
15 minutes.

GRAPHS (all 700x196px, same weewx template geometry as daywind.png):
  - daybarometer.png      -> pressure_hpa   (axis in mbar == hPa, no conversion)
  - dayouttemphilo.png    -> air_temp_f     (axis already in °F)
  - daywatertemphilo.png  -> water_temp_f   (axis already in °F)
  - daywinddir.png        -> wind_dir_deg   (fixed 0-360 axis, scatter dots not a line)

DYNAMIC AXIS SCALES: unlike wind speed/gust's originally-assumed-fixed
0-30mph ceiling (see extract_wind.py), the barometer/temp/water-temp
graphs' Y-axis auto-scales to that day's actual min/max range -- CANNOT
be hardcoded. Read via OCR (mit_graph_ocr.read_axis_scale) on every
fetch instead. Per user instruction 2026-09-10, treat wind's axis as
potentially non-fixed too, even though empirically it has shown a
consistent 30mph ceiling so far -- see extract_wind.py, which is a
SEPARATE script with its own hardcoded PLOT_Y_MIN/Y_MAX_MPH constants
still in place; if a low-wind day is ever observed with a visibly
different axis there, that script needs the same OCR-based fix applied.

WIND DIRECTION is a special case: it's rendered as small square dot
markers scattered vertically (each reading plotted independently, not
connected into a line), not a continuous colored line -- confirmed live
2026-09-10. Its axis is a fixed 0-360 range (an angular quantity, not
something weewx would sensibly auto-scale), so no OCR is needed there;
same PLOT_Y_TOP/PLOT_Y_BOTTOM geometry, hardcoded 0/360 value range.
"""
import os
import ssl
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from io import BytesIO

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mit_graph_ocr import PLOT_X_MAX, PLOT_X_MIN, PLOT_Y_BOTTOM, PLOT_Y_TOP, read_axis_scale, y_to_value

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import bulk_insert, get_connection

BASE = "https://sailing.mit.edu/weather/"
LOCATION_ID = "mit_pavilion"

DARK_GREEN = (48, 160, 48)   # single-line color used by all 3 line graphs here
COLOR_TOLERANCE = 18

SPAN_HOURS = 24
NATIVE_RESOLUTION_MINUTES = (SPAN_HOURS * 60) / (PLOT_X_MAX - PLOT_X_MIN)  # ~2.27 min, same as wind

# graph filename -> (our variable name, is_line_graph)
# is_line_graph=False means "scattered dot markers on a fixed 0-360 axis"
# (wind direction), not a continuous colored line needing OCR calibration.
GRAPHS = {
    "daybarometer.png": ("pressure_hpa", True),
    "dayouttemphilo.png": ("air_temp_f", True),
    "daywatertemphilo.png": ("water_temp_f", True),
    "daywinddir.png": ("wind_dir_deg", False),
}

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mit-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return resp.read()


def color_match(px_color, target, tol=COLOR_TOLERANCE):
    return all(abs(px_color[i] - target[i]) <= tol for i in range(3))


def columns_to_timestamps(width, end_time, span_hours):
    start_time = end_time - timedelta(hours=span_hours)
    return [start_time + (end_time - start_time) * (col / (width - 1) if width > 1 else 0)
            for col in range(width)]


def extract_line_series(img, top_value, bottom_value):
    """For a single-line graph (pressure/temp/water-temp): for each pixel
    column, find the topmost dark-green pixel and convert to a value."""
    px = img.load()
    width = PLOT_X_MAX - PLOT_X_MIN
    series = [None] * width
    for col in range(width):
        x = PLOT_X_MIN + col
        for y in range(PLOT_Y_TOP, PLOT_Y_BOTTOM + 1):
            if color_match(px[x, y][:3], DARK_GREEN):
                series[col] = round(y_to_value(y, top_value, bottom_value), 1)
                break
    return series


def extract_scatter_series(img):
    """For wind direction: dots are scattered vertically per column, not a
    single continuous line. Average all matching-color pixel rows found
    in each column (there may be 0, 1, or several dots stacked/near each
    other in a given column) and map to 0-360 degrees."""
    px = img.load()
    width = PLOT_X_MAX - PLOT_X_MIN
    series = [None] * width
    for col in range(width):
        x = PLOT_X_MIN + col
        matches = [y for y in range(PLOT_Y_TOP, PLOT_Y_BOTTOM + 1) if color_match(px[x, y][:3], DARK_GREEN)]
        if matches:
            avg_y = sum(matches) / len(matches)
            series[col] = round(y_to_value(avg_y, 360.0, 0.0), 1)
    return series


def resample_to_native(timestamps, series):
    """Buckets raw per-column readings to the graph's native per-pixel
    time resolution (mean of any values landing in the same bucket),
    same approach as extract_wind.py's resample_to_native."""
    bucket_minutes = NATIVE_RESOLUTION_MINUTES
    buckets = {}
    for t, v in zip(timestamps, series):
        if v is None:
            continue
        minutes_since_hour = t.minute + t.second / 60
        bucket_idx = int(minutes_since_hour // bucket_minutes)
        bucket_key = t.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=bucket_idx * bucket_minutes)
        buckets.setdefault(bucket_key, []).append(v)
    return {k: round(sum(vs) / len(vs), 2) for k, vs in buckets.items()}


def process_graph(filename, variable, is_line_graph):
    data = fetch_bytes(f"{BASE}{filename}")
    img = Image.open(BytesIO(data))
    width = PLOT_X_MAX - PLOT_X_MIN

    if is_line_graph:
        top_value, bottom_value = read_axis_scale(img)
        series = extract_line_series(img, top_value, bottom_value)
    else:
        series = extract_scatter_series(img)

    now_utc = datetime.now(timezone.utc)
    timestamps = columns_to_timestamps(width, now_utc, SPAN_HOURS)
    buckets = resample_to_native(timestamps, series)

    rows = [(LOCATION_ID, ts, variable, val) for ts, val in buckets.items()]
    return rows


def main():
    con = get_connection()
    total = 0
    errors = []
    try:
        for filename, (variable, is_line_graph) in GRAPHS.items():
            try:
                rows = process_graph(filename, variable, is_line_graph)
            except Exception as e:
                errors.append(f"{filename}: {e}")
                continue

            con.execute(
                "CREATE OR REPLACE TEMP TABLE stage_mit (location_id TEXT, ts_utc TIMESTAMP, variable TEXT, value DOUBLE)"
            )
            bulk_insert(con, "stage_mit", ["location_id", "ts_utc", "variable", "value"], rows)
            before = con.execute(
                "SELECT count(*) FROM observations WHERE location_id=? AND variable=?", [LOCATION_ID, variable]
            ).fetchone()[0]
            con.execute(
                """
                INSERT INTO observations (location_id, ts_utc, variable, value, source)
                SELECT s.location_id, s.ts_utc, s.variable, ANY_VALUE(s.value), 'mit_graph_ocr'
                FROM stage_mit s
                ANTI JOIN observations o
                    ON o.location_id = s.location_id AND o.ts_utc = s.ts_utc AND o.variable = s.variable
                GROUP BY s.location_id, s.ts_utc, s.variable
                """
            )
            con.execute("DROP TABLE stage_mit")
            after = con.execute(
                "SELECT count(*) FROM observations WHERE location_id=? AND variable=?", [LOCATION_ID, variable]
            ).fetchone()[0]
            new_count = after - before
            total += new_count
            print(f"{filename} -> {variable}: {len(rows)} readings in 24h window, {new_count} new rows inserted")
    finally:
        con.close()

    print(f"Total new rows: {total}")
    if errors:
        print("Errors:")
        for e in errors:
            print(" ", e)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
