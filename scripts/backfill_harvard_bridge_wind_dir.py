#!/usr/bin/env python3
"""One-off (or periodic) Harvard Bridge wind-DIRECTION history backfill.

Uses scripts/harvard_bridge_wind_dir.py's pixel-extraction algorithm
against SailFlow's chart-IMAGE endpoint (which allows a real 7-day
time_start/time_end range, unlike the 48h-capped numeric JSON endpoint
used by the regular ingest_harvard_bridge.py poller). See that module's
docstring for the full algorithm writeup.

Run manually for backfill (2 chunks covers 14 real days):
    python3 scripts/backfill_harvard_bridge_wind_dir.py --days 14
"""
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

from playwright.sync_api import sync_playwright

PROJECT_DIR = os.path.expanduser("~/.hermes/projects/sailing-weather")
sys.path.insert(0, os.path.join(PROJECT_DIR, "db"))
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))
from common import bulk_insert, get_connection  # noqa: E402
from harvard_bridge_wind_dir import (  # noqa: E402
    CHUNK_DAYS,
    IMG_HEIGHT,
    IMG_WIDTH,
    extract_arrow_bearings,
    map_x_to_timestamp,
)

SPOT_URL = "https://www.sailflow.com/spot/1834"
SPOT_ID = "1834"
LOCATION_ID = "harvard_bridge"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


def fetch_chunk_image(page, token, tz_offset_hours, chunk_start_local, chunk_end_local):
    """Requests one <=7-day chart image and returns (png_bytes,
    chunk_start_utc, chunk_end_utc)."""
    start_str = chunk_start_local.strftime("%Y-%m-%d %H:%M:%S")
    end_str = chunk_end_local.strftime("%Y-%m-%d %H:%M:%S")
    result = page.evaluate(
        """
        async ([token, spotId, startStr, endStr, w, h]) => {
            const params = `fields=wind&time_start=${encodeURIComponent(startStr)}`
                + `&time_end=${encodeURIComponent(endStr)}&format=json&type=line4`
                + `&graph_width=${w}&graph_height=${h}`;
            const url = `https://api.weatherflow.com/wxengine/rest/graph/getGraph`
                + `?units_wind=mph&units_temp=f&units_distance=mi&units_precip=in`
                + `&spot_id=${spotId}&model_ids=-101&wf_token=${token}&${params}`;
            const resp = await fetch(url);
            return await resp.text();
        }
        """,
        [token, SPOT_ID, start_str, end_str, IMG_WIDTH, IMG_HEIGHT],
    )
    import json
    data = json.loads(result)
    status = data.get("status", {"status_code": 0})
    if status.get("status_code") != 0:
        raise RuntimeError(f"getGraph image request failed: {status}")
    image_url = data["image_url"]
    resp = page.request.get(image_url)
    if resp.status != 200:
        raise RuntimeError(f"image fetch failed: HTTP {resp.status}")

    chunk_start_utc = chunk_start_local - timedelta(hours=tz_offset_hours)
    chunk_end_utc = chunk_end_local - timedelta(hours=tz_offset_hours)
    return resp.body(), chunk_start_utc, chunk_end_utc


def backfill(days):
    rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=USER_AGENT)
        page.goto(SPOT_URL, timeout=45000, wait_until="load")
        page.wait_for_timeout(6000)

        token = page.evaluate(
            """
            () => {
                const m = document.cookie.match(/(^| )wfToken=([^;]+)/);
                return m ? decodeURIComponent(m[2]) : null;
            }
            """
        )
        if not token:
            browser.close()
            raise RuntimeError("could not find wfToken cookie")

        # Get tz_offset + current local time from a cheap 48h dataonly call
        # (same one the regular poller uses) to anchor local<->UTC math.
        meta_result = page.evaluate(
            """
            async ([token, spotId]) => {
                const url = `https://api.weatherflow.com/wxengine/rest/graph/getGraph`
                    + `?units_wind=mph&fields=wind&format=json&spot_id=${spotId}`
                    + `&time_start_offset_hours=-1&time_end_offset_hours=0`
                    + `&type=dataonly&model_ids=-101&wf_token=${token}`;
                const resp = await fetch(url);
                return await resp.text();
            }
            """,
            [token, SPOT_ID],
        )
        import json
        meta = json.loads(meta_result)
        tz_offset_hours = meta["tz_offset"]
        now_local = datetime.strptime(meta["current_time_local"], "%Y-%m-%d %H:%M:%S")
        print(f"Station local_timezone={meta['local_timezone']} tz_offset={tz_offset_hours}h "
              f"current_time_local={now_local}")

        # Walk backward in CHUNK_DAYS-sized windows, ending at "now local".
        chunk_end_local = now_local
        remaining_days = days
        chunk_num = 0
        while remaining_days > 0:
            chunk_num += 1
            this_chunk_days = min(CHUNK_DAYS, remaining_days)
            chunk_start_local = chunk_end_local - timedelta(days=this_chunk_days)
            print(f"Chunk {chunk_num}: {chunk_start_local} to {chunk_end_local} local "
                  f"({this_chunk_days} days)")

            png_bytes, chunk_start_utc, chunk_end_utc = fetch_chunk_image(
                page, token, tz_offset_hours, chunk_start_local, chunk_end_local
            )
            arrows = extract_arrow_bearings(png_bytes)
            print(f"  extracted {len(arrows)} arrows from this chunk's image")

            x_centers = [x for x, _ in arrows]
            bearings = [b for _, b in arrows]
            timestamps = map_x_to_timestamp(x_centers, chunk_start_utc, chunk_end_utc, IMG_WIDTH)
            for ts, bearing in zip(timestamps, bearings):
                rows.append((LOCATION_ID, ts, "wind_dir_deg", round(bearing, 1)))

            chunk_end_local = chunk_start_local
            remaining_days -= this_chunk_days

        browser.close()

    print(f"Total rows extracted across all chunks: {len(rows)}")

    con = get_connection()
    try:
        if not rows:
            return
        con.execute(
            "CREATE OR REPLACE TEMP TABLE stage_hb_winddir "
            "(location_id TEXT, ts_utc TIMESTAMP, variable TEXT, value DOUBLE)"
        )
        bulk_insert(con, "stage_hb_winddir", ["location_id", "ts_utc", "variable", "value"], rows)
        before = con.execute(
            "SELECT count(*) FROM observations WHERE location_id=? AND variable='wind_dir_deg'",
            [LOCATION_ID],
        ).fetchone()[0]
        con.execute(
            """
            INSERT INTO observations (location_id, ts_utc, variable, value, source)
            SELECT s.location_id, s.ts_utc, s.variable, ANY_VALUE(s.value), 'sailflow_harvard_bridge_pixel_scrape'
            FROM stage_hb_winddir s
            ANTI JOIN observations o
                ON o.location_id = s.location_id AND o.ts_utc = s.ts_utc AND o.variable = s.variable
            GROUP BY s.location_id, s.ts_utc, s.variable
            """
        )
        con.execute("DROP TABLE stage_hb_winddir")
        after = con.execute(
            "SELECT count(*) FROM observations WHERE location_id=? AND variable='wind_dir_deg'",
            [LOCATION_ID],
        ).fetchone()[0]
        print(f"Inserted {after - before} new wind_dir_deg rows for {LOCATION_ID} "
              f"(had {before}, now {after})")
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14)
    args = parser.parse_args()
    sys.exit(backfill(args.days))
