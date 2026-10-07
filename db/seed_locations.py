#!/usr/bin/env python3
"""Seed the locations table with the 4 fixed reference points.

Idempotent: uses INSERT OR REPLACE so re-running just refreshes values.
Run after init_db.py: python3 db/seed_locations.py
"""
import sys

from common import get_connection

LOCATIONS = [
    # location_id,     name,                                lat,       lon,        kind
    ("mit_pavilion",  "MIT Sailing Pavilion",              42.3592,  -71.0838,  "coastal_station"),
    ("ndbc_44013",    "NDBC 44013 - Boston Approach Buoy",  42.346,  -70.651,   "buoy"),
    ("coops_8443970", "NOAA CO-OPS 8443970 - Boston (Long Wharf)", 42.3539, -71.0503, "buoy"),
    ("kbos",          "Logan International Airport (KBOS)", 42.3656, -71.0096, "airport"),
    ("cbi_dockhouse", "Community Boating Inc. Dockhouse (Charles River Basin)", 42.3598, -71.0731, "coastal_station"),
    # ADDED 2026-09-27 (user: "start tracking the harvard bridge
    # location ... that's available for free"). SailFlow spot #1834,
    # a real WeatherFlow-network station on the Charles River between
    # MIT and CBI -- see scripts/ingest_harvard_bridge.py.
    ("harvard_bridge", "Harvard Bridge (SailFlow, Charles River)", 42.35475, -71.09131, "coastal_station"),
    # ADDED 2026-10-06 (user: "I think courageous sailing is on
    # sailflow, can they be added as data source" -> "yes, add the new
    # location"). SailFlow spot #1842, "Courageous Sailing Center" --
    # found via Courageous Sailing's own public "Whiteboard" conditions
    # page (https://courageoussailing.org/whiteboard/), which embeds
    # this exact spot_id in a live SailFlow widget iframe. Unlike every
    # other location in this table, this is on BOSTON HARBOR (Pier 4,
    # Charlestown Navy Yard), not the Charles River basin -- a
    # genuinely different body of water, not a ground-truth substitute
    # for any existing Charles River location. Coordinates confirmed
    # via Nominatim lookup for "Courageous Sailing, Pier 4,
    # Charlestown". Same ingestion mechanism/48h-window limitation as
    # Harvard Bridge -- see scripts/ingest_courageous_sailing.py.
    ("courageous_sailing", "Courageous Sailing Center (SailFlow, Boston Harbor)", 42.3715, -71.0509, "coastal_station"),
]


def main():
    con = get_connection()
    try:
        for location_id, name, lat, lon, kind in LOCATIONS:
            con.execute(
                """
                INSERT INTO locations (location_id, name, lat, lon, kind)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (location_id) DO UPDATE SET
                    name = EXCLUDED.name, lat = EXCLUDED.lat,
                    lon = EXCLUDED.lon, kind = EXCLUDED.kind
                """,
                [location_id, name, lat, lon, kind],
            )
        rows = con.sql("SELECT location_id, name, lat, lon, kind FROM locations ORDER BY location_id").fetchall()
        print(f"locations table has {len(rows)} rows:")
        for r in rows:
            print(" ", r)
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main())
