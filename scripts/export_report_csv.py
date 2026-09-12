#!/usr/bin/env python3
"""On-demand CSV export of the accuracy-by-lead-time view.

Run manually whenever you actually want a CSV file -- never triggered
automatically/on a schedule. Reads straight from the live DuckDB view
`v_accuracy_by_lead_time` (db/schema.sql), which computes MAE/RMSE
on-the-fly from the permanent forecast_values/observations data -- no
separate snapshot table to keep in sync.

Usage:
    python3 export_report_csv.py [out_path]

If out_path is omitted, defaults to ./accuracy_by_lead_time.csv in the
current directory (not an auto-managed folder -- you asked for this file,
you decide where it goes).
"""
import sys

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "accuracy_by_lead_time.csv"

    con = get_connection(read_only=True)
    try:
        rows = con.execute(
            "SELECT location_id, model, variable, lead_bucket, n, mae, rmse FROM v_accuracy_by_lead_time"
        ).fetchall()
        with open(out_path, "w") as f:
            f.write("location_id,model,variable,lead_bucket,n,mae,rmse\n")
            for location_id, model, variable, lead_bucket, n, mae, rmse in rows:
                f.write(f"{location_id},{model},{variable},{lead_bucket},{n},{mae},{rmse}\n")
        print(f"Wrote {len(rows)} rows to {out_path}")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
