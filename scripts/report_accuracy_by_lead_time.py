#!/usr/bin/env python3
"""Task 13: Accuracy-by-lead-time report -- thin print wrapper around the
live DuckDB view `v_accuracy_by_lead_time` (defined in db/schema.sql).

The view does all the computation (nearest-observation match, lead-hour
bucketing, MAE/RMSE) directly in SQL against the permanent raw data
(forecast_values/observations) -- nothing is cached or snapshotted. Query
it directly with any SQL client if you don't need this pretty-printed
version:  SELECT * FROM v_accuracy_by_lead_time;
"""
import sys

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection


def print_report(con):
    rows = con.execute(
        """
        SELECT location_id, model, variable, lead_bucket, n, mae, rmse
        FROM v_accuracy_by_lead_time
        ORDER BY location_id, model, variable,
            CASE lead_bucket
                WHEN '0-6h' THEN 0 WHEN '6-12h' THEN 1 WHEN '12-24h' THEN 2
                WHEN '24-48h' THEN 3 WHEN '48-72h' THEN 4 WHEN '72h+' THEN 5
                ELSE 6
            END
        """
    ).fetchall()
    if not rows:
        print("No matched forecast/observation pairs yet -- need more ingestion history.")
        return
    header = f"{'location':<14} {'model':<8} {'variable':<16} {'lead_bucket':<10} {'n':>6} {'mae':>8} {'rmse':>8}"
    print(header)
    print("-" * len(header))
    for location_id, model, variable, lead_bucket, n, mae, rmse in rows:
        print(f"{location_id:<14} {model:<8} {variable:<16} {lead_bucket:<10} {n:>6} {mae:>8} {rmse:>8}")


if __name__ == "__main__":
    con = get_connection(read_only=True)
    try:
        print_report(con)
    finally:
        con.close()
    sys.exit(0)
