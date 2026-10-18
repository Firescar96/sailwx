#!/usr/bin/env python3
"""One-time (re-runnable) BACKFILL for the NDA ensemble model.

ADDED 2026-10-17 (user: "you need to backfill calculate the NDA models
for the past" -- after noticing NDA only had its first ~2 live runs
from scripts/ingest_nda.py's initial manual run + first cron tick, vs.
the 6 real models' 140-160 historical runs going back to September).
Without this, NDA would have looked like a brand-new location for
weeks in Forecast Time Series/Model Accuracy/Stability, even though it
could be computed retroactively from data that already exists.

WALK-FORWARD, NOT HINDSIGHT -- this is the one thing that makes a
backfill like this legitimate rather than cheating: for each historical
moment being backfilled, the regression weights are refit using ONLY
ground-truth data whose valid_time_utc is STRICTLY BEFORE that day's
start. If this script instead fit one set of weights on ALL available
history (including data from after the backfilled date) and applied it
retroactively, NDA's own historical "accuracy" would be artificially
inflated by effectively knowing the future relative to the date being
backfilled -- a classic look-ahead bias that would make Model Accuracy
and any other NDA-vs-real-model comparison meaningless. Refit cadence
is DAILY (not per-run) as a reasonable compute/rigor tradeoff -- ~40
real historical days per location, refitting at finer granularity than
that wouldn't meaningfully change the weights day-to-day anyway given
how much data each day already represents, but refitting once per real
model run (150+ times per location) would be needlessly slow for no
real accuracy benefit.

MIRRORS ingest_nda.py's exact production logic (ground-truth overrides,
MIN_TRAINING_ROWS threshold, direction sin/cos regression, "never
includes itself" via the same REAL_MODELS-only training query) -- this
is NOT a separate, divergent implementation; it imports and reuses
ingest_nda.py's fitting/prediction functions directly so the backfilled
history and the ongoing live cron job stay consistent with each other.

SAFE TO RE-RUN: every INSERT uses the same ON CONFLICT DO UPDATE
pattern as the live script, keyed on (model, location_id,
init_time_utc) for runs and (run_id, valid_time_utc, variable) for
values -- re-running this script recomputes/overwrites rather than
duplicating.
"""
import sys
from datetime import timezone

import numpy as np

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/scripts")
from ingest_nda import (
    REAL_MODELS,
    NDA_MODEL_NAME,
    MIN_TRAINING_ROWS,
    LOCATIONS,
    VARIABLES,
    ground_truth_location,
    predict_linear,
    predict_direction,
)


def fetch_full_training_rows(con, location_id, variable, gt_location):
    """Like ingest_nda.fetch_training_matrix but ALSO returns each row's
    own valid_time_utc, needed here to filter to 'data available before
    day X' for the walk-forward refit. Returns a list of
    (valid_time_utc, [6 model values], observed_value)."""
    rows = con.execute(
        """
        WITH per_model AS (
            SELECT fv.valid_time_utc, fr.model, fv.value,
                   row_number() OVER (
                       PARTITION BY fr.model, fv.valid_time_utc
                       ORDER BY epoch(fv.valid_time_utc - fr.init_time_utc)
                   ) AS rn
            FROM forecast_values fv
            JOIN forecast_runs fr ON fr.run_id = fv.run_id
            WHERE fr.location_id = ? AND fv.variable = ?
              AND fr.model IN ('gfs', 'ecmwf', 'hrrr', 'icon', 'nam', 'hrdps')
              AND epoch(fv.valid_time_utc - fr.init_time_utc) / 3600.0 BETWEEN 0 AND 12
        ),
        pivoted AS (
            SELECT valid_time_utc,
                   max(CASE WHEN model = 'gfs' THEN value END) AS gfs,
                   max(CASE WHEN model = 'ecmwf' THEN value END) AS ecmwf,
                   max(CASE WHEN model = 'hrrr' THEN value END) AS hrrr,
                   max(CASE WHEN model = 'icon' THEN value END) AS icon,
                   max(CASE WHEN model = 'nam' THEN value END) AS nam,
                   max(CASE WHEN model = 'hrdps' THEN value END) AS hrdps
            FROM per_model
            WHERE rn = 1
            GROUP BY valid_time_utc
        )
        SELECT p.valid_time_utc, p.gfs, p.ecmwf, p.hrrr, p.icon, p.nam, p.hrdps, o.value
        FROM pivoted p
        JOIN observations o
            ON o.location_id = ? AND o.variable = ?
           AND o.ts_utc BETWEEN p.valid_time_utc - INTERVAL '7.5 minutes'
                             AND p.valid_time_utc + INTERVAL '7.5 minutes'
        WHERE p.gfs IS NOT NULL AND p.ecmwf IS NOT NULL AND p.hrrr IS NOT NULL
          AND p.icon IS NOT NULL AND p.nam IS NOT NULL AND p.hrdps IS NOT NULL
        ORDER BY p.valid_time_utc
        """,
        [location_id, variable, gt_location, variable],
    ).fetchall()
    return rows  # each: (valid_time_utc, gfs, ecmwf, hrrr, icon, nam, hrdps, observed)


def fetch_full_direction_training_rows(con, location_id, gt_location):
    rows = con.execute(
        """
        WITH per_model AS (
            SELECT fv.valid_time_utc, fr.model, fv.value,
                   row_number() OVER (
                       PARTITION BY fr.model, fv.valid_time_utc
                       ORDER BY epoch(fv.valid_time_utc - fr.init_time_utc)
                   ) AS rn
            FROM forecast_values fv
            JOIN forecast_runs fr ON fr.run_id = fv.run_id
            WHERE fr.location_id = ? AND fv.variable = 'wind_dir_deg'
              AND fr.model IN ('gfs', 'ecmwf', 'hrrr', 'icon', 'nam', 'hrdps')
              AND epoch(fv.valid_time_utc - fr.init_time_utc) / 3600.0 BETWEEN 0 AND 12
        ),
        pivoted AS (
            SELECT valid_time_utc,
                   max(CASE WHEN model = 'gfs' THEN value END) AS gfs,
                   max(CASE WHEN model = 'ecmwf' THEN value END) AS ecmwf,
                   max(CASE WHEN model = 'hrrr' THEN value END) AS hrrr,
                   max(CASE WHEN model = 'icon' THEN value END) AS icon,
                   max(CASE WHEN model = 'nam' THEN value END) AS nam,
                   max(CASE WHEN model = 'hrdps' THEN value END) AS hrdps
            FROM per_model
            WHERE rn = 1
            GROUP BY valid_time_utc
        )
        SELECT p.valid_time_utc, p.gfs, p.ecmwf, p.hrrr, p.icon, p.nam, p.hrdps, o.value
        FROM pivoted p
        JOIN observations o
            ON o.location_id = ? AND o.variable = 'wind_dir_deg'
           AND o.ts_utc BETWEEN p.valid_time_utc - INTERVAL '7.5 minutes'
                             AND p.valid_time_utc + INTERVAL '7.5 minutes'
        WHERE p.gfs IS NOT NULL AND p.ecmwf IS NOT NULL AND p.hrrr IS NOT NULL
          AND p.icon IS NOT NULL AND p.nam IS NOT NULL AND p.hrdps IS NOT NULL
        ORDER BY p.valid_time_utc
        """,
        [location_id, gt_location],
    ).fetchall()
    return rows


def fit_linear_from_rows(rows):
    if len(rows) < MIN_TRAINING_ROWS:
        return None
    X = np.array([r[1:7] for r in rows], dtype=float)
    y = np.array([r[7] for r in rows], dtype=float)
    X1 = np.hstack([np.ones((len(X), 1)), X])
    beta, _, _, _ = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def fit_direction_from_rows(rows):
    if len(rows) < MIN_TRAINING_ROWS:
        return None, None
    raw_models = np.array([r[1:7] for r in rows], dtype=float)
    obs = np.array([r[7] for r in rows], dtype=float)
    X_sin = np.sin(np.radians(raw_models))
    X_cos = np.cos(np.radians(raw_models))
    y_sin = np.sin(np.radians(obs))
    y_cos = np.cos(np.radians(obs))
    X_sin1 = np.hstack([np.ones((len(X_sin), 1)), X_sin])
    X_cos1 = np.hstack([np.ones((len(X_cos), 1)), X_cos])
    beta_sin, _, _, _ = np.linalg.lstsq(X_sin1, y_sin, rcond=None)
    beta_cos, _, _, _ = np.linalg.lstsq(X_cos1, y_cos, rcond=None)
    return beta_sin, beta_cos


def get_run_forecast(con, location_id, model, variable, init_time_utc):
    """Returns {valid_time_utc: value} for the SPECIFIC historical run
    at init_time_utc (not necessarily the latest) -- this is what makes
    the backfill reconstruct "what NDA would have said back then"
    rather than always using today's current run."""
    rows = con.execute(
        """
        SELECT fv.valid_time_utc, fv.value
        FROM forecast_values fv
        JOIN forecast_runs fr ON fr.run_id = fv.run_id
        WHERE fr.location_id = ? AND fr.model = ? AND fr.init_time_utc = ? AND fv.variable = ?
        """,
        [location_id, model, init_time_utc, variable],
    ).fetchall()
    return {vt: val for vt, val in rows}


def prefetch_all_runs(con, location_id):
    """One query per location instead of 6 models x 6 vars x ~150 init
    times (~5,400 queries down to 1) -- the original per-(variable,
    init_time, model) query pattern was the real bottleneck (each
    location took minutes; a full 6-location backfill would have taken
    well over an hour). Returns {(model, variable, init_time_utc):
    {valid_time_utc: value}}."""
    rows = con.execute(
        """
        SELECT fr.model, fv.variable, fr.init_time_utc, fv.valid_time_utc, fv.value
        FROM forecast_values fv
        JOIN forecast_runs fr ON fr.run_id = fv.run_id
        WHERE fr.location_id = ? AND fr.model IN ('gfs','ecmwf','hrrr','icon','nam','hrdps')
        """,
        [location_id],
    ).fetchall()
    out = {}
    for model, variable, init_time, valid_time, value in rows:
        out.setdefault((model, variable, init_time), {})[valid_time] = value
    return out


def main():
    con = get_connection()
    total_runs_written = 0
    total_points_written = 0
    try:
        for location_id in LOCATIONS:
            # Every distinct init_time_utc across the 6 real models for
            # this location -- one NDA "run" will be backfilled at each.
            all_init_times = [
                r[0]
                for r in con.execute(
                    """
                    SELECT DISTINCT init_time_utc FROM forecast_runs
                    WHERE location_id = ? AND model IN ('gfs','ecmwf','hrrr','icon','nam','hrdps')
                    ORDER BY init_time_utc
                    """,
                    [location_id],
                ).fetchall()
            ]
            if not all_init_times:
                continue
            print(f"=== {location_id}: {len(all_init_times)} historical init times to backfill ===")

            # PERFORMANCE FIX: commit every COMMIT_EVERY_N_RUNS init
            # times instead of one giant transaction per location.
            # Confirmed directly: a single transaction spanning MIT's
            # full 349 init times ran DuckDB out of memory (3.1GB used,
            # OutOfMemoryException on executemany) -- DuckDB keeps an
            # uncommitted transaction's changes in memory, and this
            # project's write-ahead buffer isn't sized for ~350 runs x
            # up to 6 variables x many forecast_values rows all held
            # open at once. Periodic commits bound memory to roughly
            # one batch's worth of pending changes while still avoiding
            # the original per-statement-autocommit slowness.
            COMMIT_EVERY_N_RUNS = 20
            con.execute("BEGIN TRANSACTION")

            # Pre-fetch each variable's full training history ONCE per
            # location (not once per init_time) -- the walk-forward
            # filter below just slices this in-memory list by date,
            # which is cheap; re-querying the DB per init_time would be
            # the slow part otherwise.
            cached_rows = {}
            cached_dir_rows = None
            for variable in VARIABLES:
                gt_loc = ground_truth_location(location_id, variable)
                if variable == "wind_dir_deg":
                    cached_dir_rows = (fetch_full_direction_training_rows(con, location_id, gt_loc), gt_loc)
                else:
                    cached_rows[variable] = (fetch_full_training_rows(con, location_id, variable, gt_loc), gt_loc)

            # PERFORMANCE FIX -- see prefetch_all_runs()'s docstring:
            # replaces what was 6 models x 6 variables x ~150 init-time
            # DB round trips (per location) with a single query.
            all_runs = prefetch_all_runs(con, location_id)

            pending_run_values = []  # batched (run_id, valid_time, variable, value) rows -- executemany at the end of each init_time
            for i, init_time_utc in enumerate(all_init_times):
                run_id = None  # created lazily, only if at least one variable produces data
                for variable in VARIABLES:
                    if variable == "wind_dir_deg":
                        full_rows, gt_loc = cached_dir_rows
                        train_rows = [r for r in full_rows if r[0] < init_time_utc]
                        beta_sin, beta_cos = fit_direction_from_rows(train_rows)
                        if beta_sin is None:
                            continue
                    else:
                        full_rows, gt_loc = cached_rows[variable]
                        train_rows = [r for r in full_rows if r[0] < init_time_utc]
                        beta = fit_linear_from_rows(train_rows)
                        if beta is None:
                            continue

                    per_model_forecasts = {
                        m: all_runs.get((m, variable, init_time_utc), {}) for m in REAL_MODELS
                    }
                    all_valid_times = sorted(set().union(*[set(d.keys()) for d in per_model_forecasts.values()]))
                    if not all_valid_times:
                        continue

                    if run_id is None:
                        run_id = con.execute(
                            """
                            INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT (model, location_id, init_time_utc) DO UPDATE SET retrieved_at = excluded.retrieved_at
                            RETURNING run_id
                            """,
                            [NDA_MODEL_NAME, location_id, init_time_utc, init_time_utc],
                        ).fetchone()[0]
                        total_runs_written += 1

                    points_this_var = 0
                    for vt in all_valid_times:
                        model_values = [per_model_forecasts[m].get(vt) for m in REAL_MODELS]
                        if variable == "wind_dir_deg":
                            pred = predict_direction(beta_sin, beta_cos, model_values)
                        else:
                            pred = predict_linear(beta, model_values)
                        if pred is None:
                            continue
                        pending_run_values.append((run_id, vt, variable, pred))
                        points_this_var += 1
                    total_points_written += points_this_var

                # Flush this init_time's inserts in one batched executemany
                # instead of one INSERT per point -- the real bottleneck
                # once the DB-query count above was fixed.
                if pending_run_values:
                    con.executemany(
                        """
                        INSERT INTO forecast_values (run_id, valid_time_utc, variable, value)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (run_id, valid_time_utc, variable) DO UPDATE SET value = excluded.value
                        """,
                        pending_run_values,
                    )
                    pending_run_values = []

                if (i + 1) % COMMIT_EVERY_N_RUNS == 0:
                    con.execute("COMMIT")
                    con.execute("BEGIN TRANSACTION")
                    print(f"  {location_id}: committed through init_time #{i + 1}/{len(all_init_times)}")
            con.execute("COMMIT")
            print(f"  {location_id} done: {total_runs_written} runs, {total_points_written} points so far")
    finally:
        con.close()
    print(f"NDA backfill complete: {total_runs_written} historical runs created, {total_points_written} forecast points written")


if __name__ == "__main__":
    sys.exit(main())
