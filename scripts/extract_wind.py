#!/usr/bin/env python3
"""
MIT Sailing Pavilion wind data extractor.

The station's public site does not expose sub-daily wind readings as text
(only daily min/max/avg in the NOAA monthly summaries). The only place
higher-resolution data exists is in the auto-generated 'day' PNG graph,
which is overwritten continuously and never archived by the site itself.
This script downloads the current day wind graph, reads the pixel data to
reconstruct approximate Wind Speed (sustained) and Gust Speed values, and
writes them into the shared DB (and CSV) -- then discards the image (we
only keep the extracted numbers).

ONLY THE DAY GRAPH IS USED (per user instruction 2026-09-09). The week
graph was previously used as a backfill net, but its 7-day span across the
same 634px plot width gives only ~16 min/pixel resolution (vs. the day
graph's ~2.3 min/pixel) -- far too coarse, and it was also the source of a
misleading comparison during a 2026-09-08 forecast-accuracy investigation
where a week-graph column position, computed against a *later* fetch time
than the original ingestion, no longer corresponded to the same real-world
timestamp (the week graph is a rolling window that shifts every time it's
fetched, so re-deriving "what column was X at Y o'clock" after the fact is
unreliable). Only use the week graph if explicitly asked to check
something else.

RESOLUTION: previously bucketed to 5-minute averages, discarding real
resolution the graph already has -- the day graph's plot area is 634px
wide covering 24h, i.e. ~2.27 minutes/pixel, finer than 5-minute buckets.
Now resamples to native pixel resolution (~2.27 min) instead, so no
detail is thrown away that the source image actually contained.

CLIPPING: the graph's Y-axis has a hard ceiling determined by whatever
range weewx auto-scaled to that day (see AXIS SCALE note below). When
true wind speed/gust exceeds that ceiling, the plotted line visually
flattens at the top pixel row and there is no way to read the true value
from the image -- the extracted number silently reads back as exactly
the ceiling value even though the real wind could have been anything
higher. Root-caused 2026-09-09 after a user reported a recorded 26.1kt
gust that didn't match their on-the-water experience; traced to
sustained top-row pixel matches (y=24, the plot ceiling) across multiple
consecutive columns around 2026-09-08 16:20-16:25 UTC -- the signature
of clipping, not noise (a real peak-and-decay curve doesn't flatline
across 3+ consecutive columns). Every extracted reading that lands
within 1 pixel row of the ceiling is now flagged with
quality='clipped_high' in the `observations.quality` column, so
downstream analysis (forecast accuracy comparisons, flag correlation,
etc.) can choose to exclude or specially handle these known-
underestimated points instead of silently treating them as exact truth.

AXIS SCALE (CORRECTED 2026-09-10 -- this was a real, active bug, not just
a theoretical concern): originally assumed the day graph's Y axis was
FIXED at 0-30 mph, based on what it happened to show during initial
development. That assumption was WRONG -- confirmed live 2026-09-10 that
on a calmer day the same graph auto-scaled to 0-16 mph instead, meaning
every wind_speed_kt/wind_gust_kt reading recorded under the old fixed-
scale assumption on any day where the true range differed from 0-30 mph
was silently WRONG (not just clipped -- systematically mis-scaled,
roughly 2x too low on the 0-16mph day found). weewx auto-scales this
axis to each day's actual min/max, same as the pressure/temp/water-temp
graphs (see mit_graph_ocr.py, built for those). FIXED: now reads the
actual tick labels via OCR (mit_graph_ocr.read_axis_scale) on every
fetch, exactly like the other MIT graphs, instead of trusting a
hardcoded constant. See ingest_mit_graphs.py for the sibling script
that covers pressure/temp/water-temp/wind-direction using the same OCR
module.

Because pixel extraction is inherently approximate (~1 mph resolution,
antialiasing, line overlap) even away from the ceiling, treat these as
good estimates, not lab-grade measurements. Cross-validated against the
site's live "ACTUAL/HIGH" readout on 2026-09-05 and matched to within 0.1
mph away from any ceiling clipping (at the time, the axis genuinely was
0-30 mph that day -- the bug is the FIXED assumption for all days, not
that 0-30 was ever wrong as a one-time observation).

Graph geometry (700x196 px, confirmed empirically):
  - Plot area x: 44 to 678 (634 px wide)
  - Plot area y: 24 (=axis max) to 160 (=axis min), gridlines every 4-5 units
    (axis max/min now read fresh via OCR each fetch -- see AXIS SCALE above)
  - Day graph spans the last 24 hours ending "now"
  - Wind Speed (sustained) line color: (48, 160, 48) dark green
  - Gust Speed line color: (128, 208, 144) light green
"""
import csv
import os
import ssl
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mit_graph_ocr import PLOT_X_MAX, PLOT_X_MIN, PLOT_Y_BOTTOM, PLOT_Y_TOP, read_axis_scale, y_to_value

BASE = "https://sailing.mit.edu/weather/"
PROJECT_DIR = os.path.expanduser("~/.hermes/projects/sailing-weather")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "wind_native_res.csv")

# Also write into the shared DuckDB (db/ is a sibling of scripts/).
sys.path.insert(0, os.path.join(PROJECT_DIR, "db"))
from common import get_connection  # noqa: E402

LOCATION_ID = "mit_pavilion"
MPH_TO_KT = 0.868976

CLIP_ROW_TOLERANCE = 1  # rows within the axis-max row counted as "clipped"

DARK_GREEN = (48, 160, 48)     # Wind Speed (sustained)
LIGHT_GREEN = (128, 208, 144)  # Gust Speed
COLOR_TOLERANCE = 18  # per-channel distance allowed for antialiasing

# Native per-pixel time resolution: 24h across 634 columns.
SPAN_HOURS = 24
NATIVE_RESOLUTION_MINUTES = (SPAN_HOURS * 60) / (PLOT_X_MAX - PLOT_X_MIN)  # ~2.27 min

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch_bytes(url):
    req = urllib.request.Request(url, headers={"User-Agent": "mit-weather-archiver/1.0"})
    with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
        return resp.read()


def color_match(px_color, target, tol=COLOR_TOLERANCE):
    return all(abs(px_color[i] - target[i]) <= tol for i in range(3))


# Edge trimming: readings from the outermost columns of the plot are
# excluded entirely (per user instruction 2026-09-10) -- the leftmost
# edge is the oldest/about-to-roll-off data and the rightmost edge is
# "right now," which may still be actively updating/incompletely
# rendered at fetch time. Trimming a small margin from both sides avoids
# trusting whatever is least stable about the live rendering.
EDGE_TRIM_COLUMNS = 4  # ~9 minutes at ~2.27 min/column, each side


def _column_value_candidates(px, x, target_color):
    """Returns up to 3 independent value-candidates (as pixel rows) for
    a single column/color, using different analysis strategies on the
    SAME column data -- topmost match, bottommost match (within the
    contiguous matching run nearest the top), and the median of ALL
    matching rows in the column. These usually agree; when they don't,
    it's a signal that something about that column's read is unstable
    (e.g. a stray isolated pixel match unrelated to the real line, or
    the line crossing/overlapping the other series) -- see
    `resolve_reading` below for how disagreement is handled.
    Returns None if no pixel in the column matches at all.
    """
    matches = [y for y in range(PLOT_Y_TOP, PLOT_Y_BOTTOM + 1) if color_match(px[x, y][:3], target_color)]
    if not matches:
        return None

    topmost = matches[0]
    # "Bottommost within the contiguous run nearest the top" -- walk
    # matches until a gap larger than a couple pixels appears, which
    # separates the real line's thickness/antialiasing halo from an
    # unrelated stray match further down the column (e.g. the other
    # series' line, or a coincidental background pixel).
    bottommost_of_run = topmost
    for y in matches[1:]:
        if y - bottommost_of_run <= 3:
            bottommost_of_run = y
        else:
            break
    median_row = matches[len(matches) // 2]

    return topmost, bottommost_of_run, median_row


def resolve_reading(candidates):
    """Given the 3 candidate pixel rows from `_column_value_candidates`,
    pick the one that appears the most (majority vote among the 3), per
    user instruction 2026-09-10 ('take 3 different readings...choose the
    number that appears the most'). Rows are compared after rounding to
    the nearest pixel (they're already ints) -- if all 3 disagree with no
    majority, falls back to the median of the 3 as the least-bad single
    estimate (still better than blindly trusting the first/topmost read
    alone, which was the previous behavior).
    """
    if candidates is None:
        return None
    counts = {}
    for c in candidates:
        counts[c] = counts.get(c, 0) + 1
    best_count = max(counts.values())
    winners = [v for v, cnt in counts.items() if cnt == best_count]
    if len(winners) == 1:
        return winners[0]
    # No unique majority (all 3 distinct, or a 3-way tie) -- use the
    # median of the 3 as a robust fallback.
    return sorted(candidates)[1]


def extract_series(img, top_value, bottom_value):
    """
    For each pixel column in the plot area (excluding EDGE_TRIM_COLUMNS
    at both ends -- see note above), reads 3 independent candidate pixel
    rows per color (topmost / bottommost-of-run / median -- see
    `_column_value_candidates`) and takes the majority-vote result (see
    `resolve_reading`) instead of trusting a single topmost-match read.
    This hardens against isolated stray-pixel misreads that a
    single-method scan would silently accept (e.g. a background pixel
    coincidentally matching the line color at a very different height
    than where the real line actually is that column).

    Returns two lists of (mph_value, is_clipped) tuples indexed by
    column position: sustained[], gust[] (None where no reliable
    reading, e.g. gaps in the recorded data or a trimmed edge column).
    is_clipped is True when the resolved row lands within
    CLIP_ROW_TOLERANCE rows of the plot ceiling (PLOT_Y_TOP) -- meaning
    the true value may exceed what's recorded. top_value/bottom_value
    come from an OCR read of THIS fetch's actual axis labels (see AXIS
    SCALE note in the module docstring) -- no longer a hardcoded
    constant.
    """
    px = img.load()
    width = PLOT_X_MAX - PLOT_X_MIN
    sustained = [None] * width
    gust = [None] * width

    for col in range(width):
        if col < EDGE_TRIM_COLUMNS or col >= width - EDGE_TRIM_COLUMNS:
            continue
        x = PLOT_X_MIN + col

        dark_candidates = _column_value_candidates(px, x, DARK_GREEN)
        dark_row = resolve_reading(dark_candidates)
        if dark_row is not None:
            clipped = (dark_row - PLOT_Y_TOP) <= CLIP_ROW_TOLERANCE
            sustained[col] = (round(y_to_value(dark_row, top_value, bottom_value), 1), clipped)

        light_candidates = _column_value_candidates(px, x, LIGHT_GREEN)
        light_row = resolve_reading(light_candidates)
        if light_row is not None:
            clipped = (light_row - PLOT_Y_TOP) <= CLIP_ROW_TOLERANCE
            gust[col] = (round(y_to_value(light_row, top_value, bottom_value), 1), clipped)

    return sustained, gust


def columns_to_timestamps(width, end_time, span_hours):
    """
    Map each pixel column to an approximate UTC timestamp, assuming the
    graph's right edge is 'now' (end_time) and the left edge is
    span_hours before that, linearly spaced.
    """
    start_time = end_time - timedelta(hours=span_hours)
    timestamps = []
    for col in range(width):
        frac = col / (width - 1) if width > 1 else 0
        t = start_time + (end_time - start_time) * frac
        timestamps.append(t)
    return timestamps


def resample_to_native(timestamps, sustained, gust):
    """
    Bucket the raw per-pixel-column readings into buckets matching the
    graph's native per-pixel time resolution (~2.27 min for the day graph)
    instead of throwing away resolution with a coarser 5-min bucket. Takes
    the mean of value(s) in each bucket; a bucket is marked clipped if ANY
    contributing pixel was clipped (conservative -- if we can't tell if it
    clipped, prefer flagging it over silently trusting it).

    Returns a dict keyed by rounded UTC datetime ->
    (sustained_mph, sustained_clipped, gust_mph, gust_clipped).
    """
    bucket_minutes = NATIVE_RESOLUTION_MINUTES
    buckets = {}
    for t, s, g in zip(timestamps, sustained, gust):
        # round down to the nearest native-resolution bucket, anchored to
        # the top of the hour so bucket keys are stable/comparable run to run
        minutes_since_hour = t.minute + t.second / 60
        bucket_idx = int(minutes_since_hour // bucket_minutes)
        bucket_minute = bucket_idx * bucket_minutes
        bucket_key = t.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=bucket_minute)
        bucket_key = bucket_key.replace(microsecond=0)
        # round to nearest second for a clean DB timestamp
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


def load_existing_timestamps():
    if not os.path.exists(CSV_PATH):
        return set()
    existing = set()
    with open(CSV_PATH, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                existing.add(row[0])
    return existing


def append_rows(rows):
    """rows: list of (iso_ts, s_mph, s_clip, g_mph, g_clip)"""
    file_exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_utc", "wind_sustained_mph", "sustained_clipped",
                "wind_gust_mph", "gust_clipped",
            ])
        for ts, s, s_clip, g, g_clip in rows:
            writer.writerow([
                ts, s if s is not None else "", s_clip,
                g if g is not None else "", g_clip,
            ])


def write_rows_to_db(rows):
    """rows: list of (iso_ts, sustained_mph, sustained_clipped, gust_mph, gust_clipped).

    The pixel-scraped graph's Y axis is in mph (confirmed against the
    site's own live "ACTUAL" readout, also mph). Converts to knots (our
    canonical wind unit, matching NDBC/METAR/forecast ingesters) before
    writing into the shared observations table alongside the CSV
    (transition period -- keep both until the DB path is trusted).

    UPSERT, not "insert if new" (CORRECTED 2026-09-10): previously used
    ON CONFLICT DO NOTHING, which meant whatever value was recorded the
    FIRST time a given timestamp bucket was seen got locked in forever --
    even a genuinely bad read (e.g. a stray pixel misread near the
    graph's compressed left/oldest edge) could never be corrected by a
    later, better-positioned re-read of that same real-world moment as it
    moved rightward across subsequent polls. Root-caused 2026-09-10 after
    a user reported the last-24h data looked "totally off" -- confirmed a
    real ~19mph/16kt peak was visible in the live graph but the DB held a
    stale ~6.5mph/5.6kt value from an earlier, less-reliable read of that
    exact timestamp. Now overwrites with the latest read every time,
    which is the right default since a fresh re-read after edge-trimming
    and 3-method voting (see extract_series) is always at least as
    reliable as an older one, never less.
    """
    con = get_connection()
    try:
        for ts, s, s_clip, g, g_clip in rows:
            if s is not None:
                quality = "clipped_high" if s_clip else None
                con.execute(
                    """
                    INSERT INTO observations (location_id, ts_utc, variable, value, source, quality)
                    VALUES (?, ?, 'wind_speed_kt', ?, 'mit_pixel_scrape', ?)
                    ON CONFLICT (location_id, ts_utc, variable) DO UPDATE SET
                        value = excluded.value,
                        quality = excluded.quality,
                        source = excluded.source
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
                        value = excluded.value,
                        quality = excluded.quality,
                        source = excluded.source
                    """,
                    [LOCATION_ID, ts, round(g * MPH_TO_KT, 1), quality],
                )
    finally:
        con.close()


def process_day_graph(existing_timestamps):
    """Fetches and processes the day graph only. Returns (new_row_count, clipped_count).
    Axis scale is read fresh via OCR every fetch (see AXIS SCALE note in
    the module docstring) -- no longer a hardcoded 0-30mph assumption.

    Re-processes EVERY bucket in the current fetch's 24h window every
    time, not just ones never seen before (see write_rows_to_db's
    UPSERT note for why) -- `existing_timestamps` is now only used to
    track what's already in the CSV (append-only, so it still needs a
    dedup check there), not to gate DB writes."""
    url = f"{BASE}daywind.png"
    data = fetch_bytes(url)
    from io import BytesIO
    img = Image.open(BytesIO(data))
    width = PLOT_X_MAX - PLOT_X_MIN
    top_value, bottom_value = read_axis_scale(img)
    sustained, gust = extract_series(img, top_value, bottom_value)

    now_utc = datetime.now(timezone.utc)
    timestamps = columns_to_timestamps(width, now_utc, SPAN_HOURS)
    buckets = resample_to_native(timestamps, sustained, gust)

    csv_new_rows = []
    db_rows = []
    clipped_count = 0
    for ts, (s, s_clip, g, g_clip) in sorted(buckets.items()):
        if s is None and g is None:
            continue
        iso = ts.isoformat()
        db_rows.append((iso, s, s_clip, g, g_clip))
        if iso not in existing_timestamps:
            csv_new_rows.append((iso, s, s_clip, g, g_clip))
            existing_timestamps.add(iso)
        if s_clip or g_clip:
            clipped_count += 1

    if csv_new_rows:
        append_rows(csv_new_rows)
    if db_rows:
        write_rows_to_db(db_rows)  # upsert -- re-writes already-seen timestamps too, correcting stale reads
    return len(db_rows), clipped_count


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing = load_existing_timestamps()

    try:
        n, clipped = process_day_graph(existing)
        print(f"day graph: {n} new rows at ~{NATIVE_RESOLUTION_MINUTES:.2f}-min native resolution"
              f" ({clipped} with a clipped/ceiling-hit reading)")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
