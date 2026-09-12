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
