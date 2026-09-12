#!/usr/bin/env python3
"""Task (added 2026-09-08): Flag-color vs. wind-speed/gust probability report.

Correlates CBI's dockhouse flag color with observed wind conditions at the
SAME time, to answer "given a certain base wind and gust, what flag color
is it usually?" (and the inverse: "given this flag color, what wind range
does it typically correspond to?").

Nearest-observation matching: each flag reading (~every 30 min) is matched
to the nearest mit_pavilion wind_speed_kt/wind_gust_kt observation within a
tolerance window (default 20 min, wider than Phase 4's 7.5 min forecast
tolerance since flag polls are 30 min apart and we want to catch the
closest observation on either side).

Location choice: matched against mit_pavilion, not cbi_dockhouse itself
(CBI has no independent wind sensor in this DB) -- MIT Pavilion is the
closest wind observation source on the same stretch of the Charles River
Basin. NDBC/CO-OPS/KBOS are all further away / different water body and
would be less representative of conditions actually driving the
dockmaster's flag call.

Usage:
    python3 report_flag_wind_correlation.py          # prints live query
    (bucket edges below are wind SPEED buckets in knots; gust distribution
    is shown as a summary stat within each speed bucket, not bucketed
    separately, to keep the table small -- see docs/plan.md if you want a
    full 2D speed x gust breakdown instead)
"""
import sys

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

WIND_LOCATION_ID = "mit_pavilion"
TOLERANCE_MINUTES = 20

SPEED_BUCKETS = [
    (0, 5, "0-5kt"),
    (5, 10, "5-10kt"),
    (10, 15, "10-15kt"),
    (15, 20, "15-20kt"),
    (20, 25, "20-25kt"),
    (25, 1000, "25kt+"),
]


def matched_flag_wind_sql(tolerance_minutes=TOLERANCE_MINUTES, location_id=WIND_LOCATION_ID):
    tol = f"{tolerance_minutes} minutes"
    return f"""
    WITH speed_obs AS (
        SELECT ts_utc, value AS wind_speed_kt
        FROM observations
        WHERE location_id = '{location_id}' AND variable = 'wind_speed_kt'
    ),
    gust_obs AS (
        SELECT ts_utc, value AS wind_gust_kt
        FROM observations
        WHERE location_id = '{location_id}' AND variable = 'wind_gust_kt'
    ),
    flag_matched AS (
        SELECT
            f.ts_utc AS flag_ts_utc,
            f.flag_color,
            s.wind_speed_kt,
            g.wind_gust_kt,
            row_number() OVER (
                PARTITION BY f.location_id, f.ts_utc
                ORDER BY abs(epoch(f.ts_utc) - epoch(s.ts_utc))
            ) AS rn
        FROM flags f
        JOIN speed_obs s
            ON s.ts_utc BETWEEN f.ts_utc - INTERVAL '{tol}' AND f.ts_utc + INTERVAL '{tol}'
        LEFT JOIN gust_obs g
            ON g.ts_utc = s.ts_utc
    )
    SELECT flag_ts_utc, flag_color, wind_speed_kt, wind_gust_kt
    FROM flag_matched
    WHERE rn = 1
    """


def build_probability_report_sql():
    matched = matched_flag_wind_sql()
    bucket_case = "\n            ".join(
        f"WHEN wind_speed_kt >= {lo} AND wind_speed_kt < {hi} THEN '{label}'"
        for lo, hi, label in SPEED_BUCKETS
    )
    return f"""
    WITH matched AS (
        {matched}
    ),
    bucketed AS (
        SELECT
            CASE
            {bucket_case}
            ELSE 'other'
            END AS speed_bucket,
            flag_color,
            wind_gust_kt
        FROM matched
    ),
    bucket_totals AS (
        SELECT speed_bucket, count(*) AS bucket_n
        FROM bucketed
        GROUP BY speed_bucket
    )
    SELECT
        b.speed_bucket,
        b.flag_color,
        count(*) AS n,
        bt.bucket_n,
        round(100.0 * count(*) / bt.bucket_n, 1) AS pct_of_bucket,
        round(avg(b.wind_gust_kt), 1) AS avg_gust_kt
    FROM bucketed b
    JOIN bucket_totals bt ON bt.speed_bucket = b.speed_bucket
    GROUP BY b.speed_bucket, b.flag_color, bt.bucket_n
    ORDER BY
        CASE b.speed_bucket
            WHEN '0-5kt' THEN 0 WHEN '5-10kt' THEN 1 WHEN '10-15kt' THEN 2
            WHEN '15-20kt' THEN 3 WHEN '20-25kt' THEN 4 WHEN '25kt+' THEN 5
            ELSE 6
        END,
        n DESC
    """


def print_report(con):
    rows = con.execute(build_probability_report_sql()).fetchall()
    if not rows:
        print("No matched flag/wind pairs yet -- need more ingestion history (flag polls every 30 min).")
        return
    header = f"{'speed_bucket':<12} {'flag_color':<10} {'n':>5} {'bucket_n':>9} {'pct_of_bucket':>14} {'avg_gust_kt':>12}"
    print(header)
    print("-" * len(header))
    for speed_bucket, flag_color, n, bucket_n, pct, avg_gust in rows:
        print(f"{speed_bucket:<12} {flag_color:<10} {n:>5} {bucket_n:>9} {pct:>14} {avg_gust if avg_gust is not None else '':>12}")


if __name__ == "__main__":
    con = get_connection(read_only=True)
    try:
        print_report(con)
    finally:
        con.close()
    sys.exit(0)
