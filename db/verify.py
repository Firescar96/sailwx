#!/usr/bin/env python3
"""Verification helpers: join forecast_values to nearest observations.

Task 12: forecast valid_time_utc (hourly) rarely lands exactly on an
observation ts_utc (5/10-min buckets), so this does a nearest-within-
tolerance join instead of requiring exact timestamp equality.
"""
import sys
from datetime import timedelta

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

DEFAULT_TOLERANCE_MINUTES = 7.5


def nearest_obs_join_sql(tolerance_minutes=DEFAULT_TOLERANCE_MINUTES):
    """Returns a SQL fragment (as a full SELECT) that joins every
    forecast_values row to its nearest observations row (same location +
    variable) within `tolerance_minutes`, picking the single closest match
    per forecast row via a window function.

    Nearest-match logic: for each forecast_values row, find all
    observations within the tolerance window, rank by absolute time
    distance, keep the closest (row_number() = 1). If no observation
    falls inside the window, the forecast row is excluded (INNER JOIN
    semantics) -- callers wanting "forecast rows with no matching
    observation" should query forecast_values directly instead.
    """
    tol = f"{tolerance_minutes} minutes"
    return f"""
    WITH candidates AS (
        SELECT
            fv.run_id,
            fv.valid_time_utc,
            fv.variable,
            fv.value AS forecast_value,
            o.ts_utc AS obs_ts_utc,
            o.value AS observed_value,
            abs(epoch(fv.valid_time_utc) - epoch(o.ts_utc)) AS time_diff_seconds,
            row_number() OVER (
                PARTITION BY fv.run_id, fv.valid_time_utc, fv.variable
                ORDER BY abs(epoch(fv.valid_time_utc) - epoch(o.ts_utc))
            ) AS rn
        FROM forecast_values fv
        JOIN forecast_runs fr ON fr.run_id = fv.run_id
        JOIN observations o
            ON o.location_id = fr.location_id
           AND o.variable = fv.variable
           AND o.ts_utc BETWEEN fv.valid_time_utc - INTERVAL '{tol}'
                             AND fv.valid_time_utc + INTERVAL '{tol}'
    )
    SELECT run_id, valid_time_utc, variable, forecast_value, obs_ts_utc,
           observed_value, time_diff_seconds
    FROM candidates
    WHERE rn = 1
    """


def matched_forecast_observation_view(con, tolerance_minutes=DEFAULT_TOLERANCE_MINUTES):
    """Returns rows: (run_id, valid_time_utc, variable, forecast_value,
    obs_ts_utc, observed_value, time_diff_seconds)."""
    return con.execute(nearest_obs_join_sql(tolerance_minutes)).fetchall()


def create_matched_view(con, tolerance_minutes=DEFAULT_TOLERANCE_MINUTES, view_name="v_forecast_vs_observed"):
    """Materializes the nearest-match join as a DuckDB VIEW so other
    scripts/reports can just `SELECT * FROM v_forecast_vs_observed`
    instead of re-embedding this SQL."""
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {nearest_obs_join_sql(tolerance_minutes)}")


if __name__ == "__main__":
    # Smoke test: build the view and print a small sample + row count.
    con = get_connection(read_only=True)
    try:
        rows = matched_forecast_observation_view(con)
        print(f"Matched {len(rows)} (forecast, observation) pairs within {DEFAULT_TOLERANCE_MINUTES} min tolerance")
        for r in rows[:5]:
            print(" ", r)
    finally:
        con.close()
    sys.exit(0)
