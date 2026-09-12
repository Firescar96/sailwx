#!/usr/bin/env python3
"""Task 14: Forecast-convergence report.

Fixes a target valid_time (e.g. an upcoming race day/hour) and shows how
each model's prediction for that exact moment changed across successive
runs as the valid time got closer -- "how did tonight's forecast for
Saturday change since yesterday's run".

Usage:
    python3 report_forecast_convergence.py "2026-09-10 15:00:00" [variable] [location_id]

If no valid_time is given, defaults to 24 hours from now (a stand-in
"upcoming" moment) so the script is runnable standalone for a smoke test.
location_id defaults to mit_pavilion -- REQUIRED to be explicit about which
location's forecast you're looking at, since (as of 2026-09-07) forecasts
are pulled per-location and the same valid_time+variable now has rows for
all 4 locations.
"""
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection


def convergence_sql():
    return """
    SELECT
        fr.model,
        fr.init_time_utc,
        date_diff('hour', fr.init_time_utc, ?) AS lead_hours_at_init,
        fv.value AS forecast_value
    FROM forecast_values fv
    JOIN forecast_runs fr ON fr.run_id = fv.run_id
    WHERE fv.valid_time_utc = ?
      AND fv.variable = ?
      AND fr.location_id = ?
    ORDER BY fr.model, fr.init_time_utc
    """


def print_report(con, valid_time, variable, location_id):
    rows = con.execute(convergence_sql(), [valid_time, valid_time, variable, location_id]).fetchall()
    print(f"Forecast convergence for {variable} at {location_id}, valid_time {valid_time} UTC")
    if not rows:
        print("  No forecast_values rows cover this exact valid_time+variable+location yet.")
        print("  (Forecasts are hourly UTC timestamps -- pick one on the hour, and one that a")
        print("   currently-ingested run's horizon actually reaches.)")
        return
    header = f"{'model':<8} {'init_time_utc':<20} {'lead_h_at_init':>14} {'forecast_value':>14}"
    print(header)
    print("-" * len(header))
    for model, init_time, lead_hours, value in rows:
        print(f"{model:<8} {str(init_time):<20} {lead_hours:>14} {value:>14}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        valid_time = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    else:
        # Default: next top-of-hour ~24h out, so a smoke-test run has a
        # chance of hitting a real ingested forecast point.
        valid_time = (datetime.now(timezone.utc) + timedelta(hours=24)).replace(minute=0, second=0, microsecond=0)
    variable = sys.argv[2] if len(sys.argv) > 2 else "wind_speed_kt"
    location_id = sys.argv[3] if len(sys.argv) > 3 else "mit_pavilion"

    con = get_connection(read_only=True)
    try:
        print_report(con, valid_time, variable, location_id)
    finally:
        con.close()
    sys.exit(0)
