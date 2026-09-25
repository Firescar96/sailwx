#!/usr/bin/env python3
"""MIT Sailing Pavilion combined weewx day-graph extractor.

MERGED 2026-09-11 (per user instruction: "combine the functions cleanly
from both mit scrapers into one") from what were previously two separate
scripts/cron jobs -- extract_wind.py (wind speed/gust) and
ingest_mit_graphs.py (pressure/air-temp/water-temp/wind-direction).
Combining them into a single script/cron job means:
  - Only ONE DuckDB write-lock acquisition per run instead of two,
    directly reducing the lock-contention "stuck job" failures that kept
    recurring this session (both scripts were scheduled at the exact
    same "0 */4 * * *" minute as each other AND as the forecast
    ingester, guaranteeing collisions every single run -- see
    db/common.py's stale-lock detection for the other half of this fix).
  - Only ONE HTTP fetch round-trip pattern / cron job to monitor instead
    of two nearly-identical ones.
  - All 5 graphs (wind, barometer, air-temp, water-temp, wind-dir) share
    the exact same weewx template geometry (mit_graph_ocr.py) and OCR
    axis-reading approach; the wind graph's extra edge-trimming/3-vote
    pixel-resolution logic (extract_wind.py's more careful
    _column_value_candidates/resolve_reading, added after real accuracy
    bugs were found) is preserved as-is for wind, while the simpler
    topmost-match approach (still correct, just less defensive) is kept
    for the other 4 graphs, matching each script's original logic
    exactly -- this merge does not change either script's extraction
    behavior, only where they run.

Runs every 4 hours (see cron job "MIT Pavilion Weather Graph Ingester").
Each graph shows a rolling 24h window, so a missed run is self-healing:
the next run's window overlaps and backfills anything missed.

GRAPHS (all 700x196px):
  - daywind.png           -> wind_speed_kt, wind_gust_kt (line pair, dynamic
                              axis, 3-candidate-vote + edge-trim + clip
                              detection -- see extract_wind_graph())
  - daybarometer.png      -> pressure_hpa   (line, dynamic axis)
  - dayouttemphilo.png    -> air_temp_f     (line, dynamic axis)
  - daywatertemphilo.png  -> water_temp_f   (line, dynamic axis)
  - daywinddir.png        -> wind_dir_deg   (scatter dots, fixed 0-360 axis)
"""
import csv
import os
import ssl
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from io import BytesIO

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mit_graph_ocr import PLOT_X_MAX, PLOT_X_MIN, PLOT_Y_BOTTOM, PLOT_Y_TOP, read_axis_scale, y_to_value

PROJECT_DIR = os.path.expanduser("~/.hermes/projects/sailing-weather")
sys.path.insert(0, os.path.join(PROJECT_DIR, "db"))
from common import bulk_insert, get_connection  # noqa: E402

BASE = "https://sailing.mit.edu/weather/"
LOCATION_ID = "mit_pavilion"
MPH_TO_KT = 0.868976

DATA_DIR = os.path.join(PROJECT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "wind_native_res.csv")  # unchanged path/format from extract_wind.py

DARK_GREEN = (48, 160, 48)     # Wind Speed (sustained); also the single line
                                # color used by barometer/temp/water-temp/wind-dir
LIGHT_GREEN = (128, 208, 144)  # Gust Speed only
COLOR_TOLERANCE = 18

SPAN_HOURS = 24
NATIVE_RESOLUTION_MINUTES = (SPAN_HOURS * 60) / (PLOT_X_MAX - PLOT_X_MIN)  # ~2.27 min

CLIP_ROW_TOLERANCE = 1
EDGE_TRIM_COLUMNS = 4  # ~9 minutes at ~2.27 min/column, each side (wind graph only)

# graph filename -> (variable name, is_line_graph, target_color).
# is_line_graph=False means "scattered dot markers on a fixed 0-360 axis"
# (wind direction). target_color picks out ONE line from graphs that
# plot multiple lines together (e.g. dayinouttempdew.png has 4 lines --
# outside temp/dew point/inside temp/water temp -- we only want dew
# point from it, since outside/water temp are already captured from
# their own dedicated -hilo graphs).
#
# ADDED 2026-09-25 (user: "add to the mit scraper, you need to scrape
# all of the charts in the daily scraper including radiation... this
# page lists all the charts available", referring to
# https://sailing.mit.edu/weather/history.html). Checked every chart on
# that page: dayradiation.png (solar radiation, genuinely new data),
# dayinouthum.png (outside/inside humidity, genuinely new), and the dew
# point line within dayinouttempdew.png (genuinely new -- outside/water
# temp from that same image were already covered by the dedicated -hilo
# graphs, so only dew point needed pulling from it) are the 3 real new
# data sources added. Explicitly SKIPPED per this same investigation:
# daywindvec.png (wind vectors) and daytempchill.png (wind chill/heat
# index) are just visual RECOMBINATIONS of data already captured
# elsewhere (wind speed+direction; temp+humidity/wind), not new
# information; dayrain.png and daylightning.png use a bar-chart style
# (discrete per-hour bars, not a continuous traced line) that this
# module's line/scatter extraction can't handle as-is -- left for a
# future dedicated bar-chart extractor if ever needed.
OTHER_GRAPHS = {
    "daybarometer.png": ("pressure_hpa", True, DARK_GREEN),
    "dayouttemphilo.png": ("air_temp_f", True, DARK_GREEN),
    "daywatertemphilo.png": ("water_temp_f", True, DARK_GREEN),
    "daywinddir.png": ("wind_dir_deg", False, DARK_GREEN),
    "dayradiation.png": ("solar_radiation_wm2", True, DARK_GREEN),
    "dayinouthum.png": ("humidity_pct", True, DARK_GREEN),  # outside humidity specifically (dark green); inside humidity (light green) not captured
    "dayinouttempdew.png": ("dew_point_f", True, LIGHT_GREEN),  # dew point is the LIGHT_GREEN line on this 4-line graph
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


# ---------------------------------------------------------------------------
# Wind speed/gust extraction (from extract_wind.py, unchanged logic)
# ---------------------------------------------------------------------------

def _column_value_candidates(px, x, target_color):
    """3 independent value-candidates (pixel rows) per column/color:
    topmost, bottommost-of-contiguous-run-near-top, median of all
    matches -- see resolve_reading for how disagreement is handled."""
    matches = [y for y in range(PLOT_Y_TOP, PLOT_Y_BOTTOM + 1) if color_match(px[x, y][:3], target_color)]
    if not matches:
        return None
    topmost = matches[0]
    bottommost_of_run = topmost
    for y in matches[1:]:
        if y - bottommost_of_run <= 3:
            bottommost_of_run = y
        else:
            break
    median_row = matches[len(matches) // 2]
    return topmost, bottommost_of_run, median_row


def resolve_reading(candidates):
    """Majority vote among the 3 candidates; median fallback on no majority."""
    if candidates is None:
        return None
    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    best_count = max(counts.values())
    winners = [v for v, cnt in counts.items() if cnt == best_count]
    if len(winners) == 1:
        return winners[0]
    return sorted(candidates)[1]


def extract_wind_series(img, top_value, bottom_value):
    """Returns (sustained[], gust[]) lists of (mph_value, is_clipped) or
    None, per pixel column, edge-trimmed and 3-vote resolved."""
    px = img.load()
    width = PLOT_X_MAX - PLOT_X_MIN
    sustained = [None] * width
    gust = [None] * width

    for col in range(width):
        if col < EDGE_TRIM_COLUMNS or col >= width - EDGE_TRIM_COLUMNS:
            continue
        x = PLOT_X_MIN + col

        dark_row = resolve_reading(_column_value_candidates(px, x, DARK_GREEN))
        if dark_row is not None:
            clipped = (dark_row - PLOT_Y_TOP) <= CLIP_ROW_TOLERANCE
            sustained[col] = (round(y_to_value(dark_row, top_value, bottom_value), 1), clipped)

        light_row = resolve_reading(_column_value_candidates(px, x, LIGHT_GREEN))
        if light_row is not None:
            clipped = (light_row - PLOT_Y_TOP) <= CLIP_ROW_TOLERANCE
            gust[col] = (round(y_to_value(light_row, top_value, bottom_value), 1), clipped)

    return sustained, gust


SMOOTH_WINDOW_COLUMNS = 5  # ~11 minutes at ~2.27 min/column, centered on each column
SMOOTH_OUTLIER_MAD_MULTIPLIER = 3.5  # how many MADs from the local median counts as "genuinely an outlier" (see smooth_column_series)


def smooth_column_series(series, window=SMOOTH_WINDOW_COLUMNS):
    """Outlier-only correction across pixel columns (user 2026-09-19:
    "MIT sailing data is very noisy I think due to the scraping, maybe
    you could take more samples, but average them or take a clear mode
    to make that better"; then 2026-09-23, after this filter shipped:
    "mit sailing data is too smooth, there are some touches of 20 knots
    in the past day that are getting smoothed out, i want to keep the
    highs").

    REDESIGNED 2026-09-23 -- the original implementation was a blanket
    rolling MEDIAN that unconditionally replaced EVERY column's value
    with its neighborhood's median, no matter what. That's exactly why
    it worked for the noise case (a real spike like 1.7kt -> 6.9kt ->
    1.3kt got corrected) but ALSO quietly clipped real brief peaks --
    confirmed empirically: 2 genuine gust readings (20.6kt, 20.1kt) got
    pulled down to 19.3kt by the blanket median, even though they
    weren't noise at all, just a real gust that only lasted one pixel
    column. A single-column noise artifact and a single-column-wide
    real gust peak look statistically identical to a filter that always
    overwrites with the local median -- the filter can't tell "this
    point is wrong" from "this point is the true, brief maximum of a
    real gust" using magnitude alone.

    Fix: switched to a Hampel-filter-style OUTLIER-ONLY correction.
    Still computes the local median (and now also the median absolute
    deviation, MAD, a robust measure of "how much do points around here
    normally vary") over the same window, but only overrides a column
    when it deviates from that local median by more than
    SMOOTH_OUTLIER_MAD_MULTIPLIER times the MAD -- i.e. only when a
    point is a genuine, dramatic outlier relative to its own local
    variability, not merely "higher than its immediate neighbors" (which
    describes every real gust peak by definition). Real noise spikes
    (isolated, both neighbors far away in value) still get caught and
    corrected; real gust peaks (even ones lasting just one column) now
    pass through completely untouched, preserving the true high.
    """
    n = len(series)
    half = window // 2
    smoothed = [None] * n
    for i in range(n):
        if series[i] is None:
            continue
        raw_val, clipped = series[i]
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighborhood = [series[j][0] for j in range(lo, hi) if series[j] is not None]
        if len(neighborhood) < 3:
            # Not enough real neighbors to judge "normal variability" at
            # all -- keep the raw reading rather than guess.
            smoothed[i] = (raw_val, clipped)
            continue
        neighborhood.sort()
        m = len(neighborhood)
        median_val = (
            neighborhood[m // 2]
            if m % 2 == 1
            else (neighborhood[m // 2 - 1] + neighborhood[m // 2]) / 2
        )
        deviations = sorted(abs(v - median_val) for v in neighborhood)
        mad = (
            deviations[m // 2]
            if m % 2 == 1
            else (deviations[m // 2 - 1] + deviations[m // 2]) / 2
        )
        # A near-zero MAD (a genuinely flat/calm stretch) would make even
        # a tiny real fluctuation look like a huge multiple of it -- a
        # small floor keeps the outlier test meaningful instead of
        # over-triggering during calm periods.
        mad_floor = 0.3
        threshold = SMOOTH_OUTLIER_MAD_MULTIPLIER * max(mad, mad_floor)
        if abs(raw_val - median_val) > threshold:
            smoothed[i] = (round(median_val, 1), clipped)
        else:
            smoothed[i] = (raw_val, clipped)
    return smoothed


def resample_wind_to_native(timestamps, sustained, gust):
    bucket_minutes = NATIVE_RESOLUTION_MINUTES
    buckets = {}
    for t, s, g in zip(timestamps, sustained, gust):
        minutes_since_hour = t.minute + t.second / 60
        bucket_idx = int(minutes_since_hour // bucket_minutes)
        bucket_key = t.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=bucket_idx * bucket_minutes)
        bucket_key = bucket_key - timedelta(seconds=bucket_key.second, microseconds=bucket_key.microsecond)

        buckets.setdefault(bucket_key, {"s": [], "s_clip": False, "g": [], "g_clip": False})
        if s is not None:
            val, clipped = s
            buckets[bucket_key]["s"].append(val)
            buckets[bucket_key]["s_clip"] = buckets[bucket_key]["s_clip"] or clipped
        if g is not None:
            val, clipped = g
            buckets[bucket_key]["g"].append(val)
            buckets[bucket_key]["g_clip"] = buckets[bucket_key]["g_clip"] or clipped

    result = {}
    for key, vals in buckets.items():
        s_avg = round(sum(vals["s"]) / len(vals["s"]), 1) if vals["s"] else None
        g_avg = round(sum(vals["g"]) / len(vals["g"]), 1) if vals["g"] else None
        result[key] = (s_avg, vals["s_clip"], g_avg, vals["g_clip"])
    return result


def load_existing_wind_timestamps():
    if not os.path.exists(CSV_PATH):
        return set()
    existing = set()
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                existing.add(row[0])
    return existing


def append_wind_csv_rows(rows):
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_utc", "wind_sustained_mph", "sustained_clipped",
                "wind_gust_mph", "gust_clipped",
            ])
        for ts, s, s_clip, g, g_clip in rows:
            writer.writerow([ts, s if s is not None else "", s_clip, g if g is not None else "", g_clip])


def write_wind_rows_to_db(con, rows):
    """Upsert (not insert-if-new) -- a fresh re-read is always at least as
    reliable as an older one, never less (see extract_wind.py history)."""
    for ts, s, s_clip, g, g_clip in rows:
        if s is not None:
            quality = "clipped_high" if s_clip else None
            con.execute(
                """
                INSERT INTO observations (location_id, ts_utc, variable, value, source, quality)
                VALUES (?, ?, 'wind_speed_kt', ?, 'mit_pixel_scrape', ?)
                ON CONFLICT (location_id, ts_utc, variable) DO UPDATE SET
                    value = excluded.value, quality = excluded.quality, source = excluded.source
                """,
                [LOCATION_ID, ts, round(s * MPH_TO_KT, 1), quality],
            )
        if g is not None:
            quality = "clipped_high" if g_clip else None
            con.execute(
                """
                INSERT INTO observations (location_id, ts_utc, variable, value, source, quality)
                VALUES (?, ?, 'wind_gust_kt', ?, 'mit_pixel_scrape', ?)
                ON CONFLICT (location_id, ts_utc, variable) DO UPDATE SET
                    value = excluded.value, quality = excluded.quality, source = excluded.source
                """,
                [LOCATION_ID, ts, round(g * MPH_TO_KT, 1), quality],
            )


def process_wind_graph(con, existing_wind_timestamps):
    """Returns (new_row_count, clipped_count)."""
    data = fetch_bytes(f"{BASE}daywind.png")
    img = Image.open(BytesIO(data))
    width = PLOT_X_MAX - PLOT_X_MIN
    top_value, bottom_value = read_axis_scale(img)
    sustained, gust = extract_wind_series(img, top_value, bottom_value)
    # Rolling-median smoothing across raw pixel columns, BEFORE bucketing
    # into native-resolution output rows -- this is where an isolated
    # bad column (surrounded by consistent neighbors) actually gets
    # corrected; smoothing after bucketing would be too coarse (each
    # bucket already averages multiple columns together at
    # NATIVE_RESOLUTION_MINUTES, but that doesn't help when the bad
    # column IS the whole bucket at higher time resolutions).
    sustained = smooth_column_series(sustained)
    gust = smooth_column_series(gust)

    now_utc = datetime.now(timezone.utc)
    timestamps = columns_to_timestamps(width, now_utc, SPAN_HOURS)
    buckets = resample_wind_to_native(timestamps, sustained, gust)

    csv_new_rows, db_rows, clipped_count = [], [], 0
    for ts, (s, s_clip, g, g_clip) in sorted(buckets.items()):
        if s is None and g is None:
            continue
        iso = ts.isoformat()
        db_rows.append((iso, s, s_clip, g, g_clip))
        if iso not in existing_wind_timestamps:
            csv_new_rows.append((iso, s, s_clip, g, g_clip))
            existing_wind_timestamps.add(iso)
        if s_clip or g_clip:
            clipped_count += 1

    if csv_new_rows:
        append_wind_csv_rows(csv_new_rows)
    if db_rows:
        write_wind_rows_to_db(con, db_rows)
    return len(db_rows), clipped_count


# ---------------------------------------------------------------------------
# Pressure/air-temp/water-temp/wind-direction extraction (from
# ingest_mit_graphs.py, unchanged logic)
# ---------------------------------------------------------------------------

def extract_line_series(img, top_value, bottom_value, target_color=DARK_GREEN):
    """Extracts a single line series matching `target_color`. Some MIT
    graphs plot MULTIPLE lines together (e.g. dayinouttempdew.png shows
    outside temp / dew point / inside temp / water temp as 4 different
    colors on one chart) -- target_color lets a caller pick out just the
    one line it actually wants from a multi-line graph, ignoring the
    others entirely, the same way daywind.png already distinguishes
    DARK_GREEN (sustained) from LIGHT_GREEN (gust)."""
    px = img.load()
    width = PLOT_X_MAX - PLOT_X_MIN
    series = [None] * width
    for col in range(width):
        x = PLOT_X_MIN + col
        for y in range(PLOT_Y_TOP, PLOT_Y_BOTTOM + 1):
            if color_match(px[x, y][:3], target_color):
                series[col] = round(y_to_value(y, top_value, bottom_value), 1)
                break
    return series


def extract_scatter_series(img):
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


def resample_other_to_native(timestamps, series):
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


def process_other_graph(filename, variable, is_line_graph, target_color=DARK_GREEN):
    data = fetch_bytes(f"{BASE}{filename}")
    img = Image.open(BytesIO(data))
    width = PLOT_X_MAX - PLOT_X_MIN

    if is_line_graph:
        top_value, bottom_value = read_axis_scale(img)
        series = extract_line_series(img, top_value, bottom_value, target_color)
    else:
        series = extract_scatter_series(img)

    now_utc = datetime.now(timezone.utc)
    timestamps = columns_to_timestamps(width, now_utc, SPAN_HOURS)
    buckets = resample_other_to_native(timestamps, series)
    return [(LOCATION_ID, ts, variable, val) for ts, val in buckets.items()]


def write_other_rows_to_db(con, filename, variable, rows):
    """Insert-if-new (not upsert) -- matches ingest_mit_graphs.py's
    original behavior exactly (these 4 variables don't have the same
    known stale-value-correction need that wind's upsert was fixing)."""
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
    return after - before


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_wind_timestamps = load_existing_wind_timestamps()

    errors = []
    total_new = 0

    # ONE connection for the whole run (this is the actual point of the
    # merge -- was 2 separate get_connection() calls/lock acquisitions
    # across 2 separate cron jobs before).
    con = get_connection()
    try:
        try:
            n, clipped = process_wind_graph(con, existing_wind_timestamps)
            print(f"daywind.png -> wind_speed_kt/wind_gust_kt: {n} new rows at "
                  f"~{NATIVE_RESOLUTION_MINUTES:.2f}-min native resolution "
                  f"({clipped} with a clipped/ceiling-hit reading)")
            total_new += n
        except Exception as e:
            errors.append(f"daywind.png: {e}")

        for filename, (variable, is_line_graph, target_color) in OTHER_GRAPHS.items():
            try:
                rows = process_other_graph(filename, variable, is_line_graph, target_color)
                new_count = write_other_rows_to_db(con, filename, variable, rows)
                total_new += new_count
                print(f"{filename} -> {variable}: {len(rows)} readings in 24h window, {new_count} new rows inserted")
            except Exception as e:
                errors.append(f"{filename}: {e}")
    finally:
        con.close()

    print(f"Total new rows: {total_new}")
    if errors:
        print("Errors:")
        for e in errors:
            print(" ", e)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
