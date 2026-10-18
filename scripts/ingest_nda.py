#!/usr/bin/env python3
"""NDA -- the "weighted voting" ensemble forecast model.

ADDED 2026-10-17 (user: "for every metric for every location for every
graph, i need a new model, the Nchinda model that does linear
regression across predictions for all models to give its own answer,
weighted voting" -> named "NDA" per follow-up). For every (location,
variable) combo that has enough real history, fits a genuine linear
regression -- NDA = beta0 + beta1*gfs + beta2*ecmwf + beta3*hrrr +
beta4*icon + beta5*nam + beta6*hrdps -- against real ground-truth
observations, then writes its own OWN forecast_runs/forecast_values
rows (model='nda') for the CURRENT run of each of the 6 real models,
exactly like any other model would. This is a deliberate architectural
choice: by writing real rows into the same tables every other model
uses, NDA automatically shows up in EVERY existing chart (Forecast
Time Series, Forecast Stability, Model Accuracy, Wind Rose, Gustiness,
Wind/Gust Prediction, Flag Prediction) with zero special-casing needed
in any of those 6+ separate chart-rendering code paths -- the
alternative (hand-wiring a synthetic "6th model" into every endpoint)
would have been far more invasive and error-prone for a user request
explicitly scoped to "every metric, every location, every graph".

GROUND TRUTH, per explicit user direction (both the general
ground-truth-substitution pattern already used everywhere else in this
project, AND an explicit instruction during this feature's design:
"when given the option to use harvard bridge data vs mit sailing
pavilion data to calculate regressions, use the harvard bridge ground
truth data"):
    - wind_speed_kt, wind_gust_kt, wind_dir_deg, pressure_hpa:
      Harvard Bridge observations (mirrors the Wind/Gust Prediction +
      Flag Prediction + Accuracy views' existing substitution for
      mit_pavilion -- EXTENDED here to cbi_dockhouse too, since CBI
      also has no sensor of its own and Harvard Bridge is the only
      real ground truth available for those variables at CBI).
    - air_temp_f, water_temp_f (Harvard Bridge has no temperature
      sensor at all): falls back to that location's OWN observations.
      This is a deliberate, narrower exception confirmed with the user
      during design -- temps were never part of the "use Harvard
      Bridge" instruction (which was about wind/gust/direction/
      pressure specifically), and Harvard Bridge genuinely has no
      temperature data to substitute even if the instruction were
      read that broadly.
    - kbos, ndbc_44013, courageous_sailing, coops_8443970: use their
      own real sensors directly (no substitution available/needed --
      Harvard Bridge ground-truth substitution has ALWAYS been
      specific to mit_pavilion/cbi_dockhouse in this project, since
      those are the two locations on the Charles River basin close
      enough to Harvard Bridge to be a meaningful stand-in; KBOS/NDBC/
      Courageous/CO-OPS are real independent sensors already).

NEVER INCLUDES ITSELF: per explicit user instruction ("as long as NDA
isn't computed to include itself"), the regression is fit ONLY from
the 6 real NWP models' (gfs, ecmwf, hrrr, icon, nam, hrdps) historical
forecast values -- this script never reads back its own prior 'nda'
rows as a regression input, and the SQL below explicitly filters
fr.model IN (...) to the 6 real models, excluding 'nda' by construction
even if a prior run's rows are still in the table.

DIRECTION (wind_dir_deg) IS CIRCULAR -- per explicit user direction
("regress the sin/cos components separately ... then reconstruct the
angle -- keeps it genuinely 'linear regression', handles wraparound
correctly"): two SEPARATE linear regressions are fit, one predicting
sin(observed_direction) from each model's sin(forecast_direction), one
predicting cos(...) the same way, then NDA's own direction is
reconstructed via atan2(predicted_sin, predicted_cos). This correctly
handles wraparound (e.g. models disagreeing between 350 degrees and 10
degrees, which are actually only 20 degrees apart) the same way
circularMeanDeg() does on the frontend, but as a genuine fitted
regression rather than an unweighted average.

MINIMUM DATA BAR: reuses the same MIN_TRAINING_ROWS=20 threshold
already established in wind_regression_model.py (per user: "same
minimum-row threshold already used elsewhere in this project") --
skips writing NDA rows for any (location, variable) that doesn't have
at least 20 real matched (all 6 models + ground truth) historical rows
to fit from. This naturally means a brand-new location like Courageous
Sailing may not get NDA coverage for a while after being added, and a
variable with sparse ground truth (e.g. KBOS's wind_gust_kt, which has
real METAR reporting gaps -- see the "Logan Airport gust-reporting
convention" investigation elsewhere in this project) may also fall
below the bar even at an otherwise well-covered location.

CADENCE: run on the SAME cron cadence as the main Forecast Ingester
(see sw_ingest_nda.sh), right after it -- NDA needs each of the 6 real
models' CURRENT run to already be in the database before it can
produce its own forecast for "right now", so it must run after (not
concurrently with) the main forecast ingester.
"""
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

REAL_MODELS = ["gfs", "ecmwf", "hrrr", "icon", "nam", "hrdps"]
NDA_MODEL_NAME = "nda"
MIN_TRAINING_ROWS = 20  # mirrors website/wind_regression_model.py's threshold

# (location_id, variable) -> ground_truth_location_id. Variables/locations
# not listed here use their own location_id as ground truth (the "use own
# sensor" default -- see module docstring).
GROUND_TRUTH_OVERRIDE = {}
for _loc in ("mit_pavilion", "cbi_dockhouse"):
    for _var in ("wind_speed_kt", "wind_gust_kt", "wind_dir_deg", "pressure_hpa"):
        GROUND_TRUTH_OVERRIDE[(_loc, _var)] = "harvard_bridge"

LOCATIONS = ["mit_pavilion", "kbos", "ndbc_44013", "cbi_dockhouse", "courageous_sailing", "coops_8443970"]
VARIABLES = ["wind_speed_kt", "wind_gust_kt", "wind_dir_deg", "pressure_hpa", "air_temp_f", "water_temp_f"]


def ground_truth_location(location_id, variable):
    return GROUND_TRUTH_OVERRIDE.get((location_id, variable), location_id)


def fetch_training_matrix(con, location_id, variable, gt_location):
    """Returns (X, y) where X is an (n, 6) array of each real model's
    forecast value (columns in REAL_MODELS order) and y is the matched
    ground-truth observed value, for every historical valid_time_utc
    where ALL 6 models AND a ground-truth observation are available.
    Uses the nearest-observation-within-7.5min matching convention
    already established by v_accuracy_by_lead_time/the dedup-by-real-
    event pattern elsewhere in this project."""
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
        SELECT p.gfs, p.ecmwf, p.hrrr, p.icon, p.nam, p.hrdps, o.value
        FROM pivoted p
        JOIN observations o
            ON o.location_id = ? AND o.variable = ?
           AND o.ts_utc BETWEEN p.valid_time_utc - INTERVAL '7.5 minutes'
                             AND p.valid_time_utc + INTERVAL '7.5 minutes'
        WHERE p.gfs IS NOT NULL AND p.ecmwf IS NOT NULL AND p.hrrr IS NOT NULL
          AND p.icon IS NOT NULL AND p.nam IS NOT NULL AND p.hrdps IS NOT NULL
        """,
        [location_id, variable, gt_location, variable],
    ).fetchall()
    if not rows:
        return None, None
    X = np.array([r[:6] for r in rows], dtype=float)
    y = np.array([r[6] for r in rows], dtype=float)
    return X, y


def fit_linear_ensemble(X, y):
    """Ordinary least squares with intercept: y ~= beta0 + beta . X.
    Same "Bayesian under a flat prior" framing as
    wind_regression_model.py -- OLS is the MAP solution under a
    Gaussian likelihood with an uninformative prior, not a different
    thing. Returns None if there's not enough data."""
    if X is None or len(X) < MIN_TRAINING_ROWS:
        return None
    X1 = np.hstack([np.ones((len(X), 1)), X])
    beta, _residuals, _rank, _sv = np.linalg.lstsq(X1, y, rcond=None)
    return beta


def predict_linear(beta, model_values):
    """model_values: list of 6 floats in REAL_MODELS order (or None for
    a missing model -- substituted with the mean of the other present
    models so one missing model doesn't zero out the whole prediction)."""
    present = [v for v in model_values if v is not None]
    if not present:
        return None
    fallback = sum(present) / len(present)
    row = np.array([1.0] + [v if v is not None else fallback for v in model_values])
    return float(row @ beta)


def fetch_direction_training_matrix(con, location_id, gt_location):
    """Same shape as fetch_training_matrix but returns sin/cos pairs
    for circular regression -- see module docstring's "DIRECTION IS
    CIRCULAR" section."""
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
        SELECT p.gfs, p.ecmwf, p.hrrr, p.icon, p.nam, p.hrdps, o.value
        FROM pivoted p
        JOIN observations o
            ON o.location_id = ? AND o.variable = 'wind_dir_deg'
           AND o.ts_utc BETWEEN p.valid_time_utc - INTERVAL '7.5 minutes'
                             AND p.valid_time_utc + INTERVAL '7.5 minutes'
        WHERE p.gfs IS NOT NULL AND p.ecmwf IS NOT NULL AND p.hrrr IS NOT NULL
          AND p.icon IS NOT NULL AND p.nam IS NOT NULL AND p.hrdps IS NOT NULL
        """,
        [location_id, gt_location],
    ).fetchall()
    if not rows:
        return None, None, None
    raw_models = np.array([r[:6] for r in rows], dtype=float)
    obs = np.array([r[6] for r in rows], dtype=float)
    X_sin = np.sin(np.radians(raw_models))
    X_cos = np.cos(np.radians(raw_models))
    y_sin = np.sin(np.radians(obs))
    y_cos = np.cos(np.radians(obs))
    return X_sin, X_cos, (y_sin, y_cos)


def predict_direction(beta_sin, beta_cos, model_degrees):
    """model_degrees: list of 6 raw direction forecasts (degrees, or
    None). Predicts sin/cos separately via the 2 fitted regressions,
    then reconstructs the angle via atan2 -- correctly handles
    wraparound (e.g. models split between 350 and 10 degrees)."""
    present = [d for d in model_degrees if d is not None]
    if not present:
        return None
    sins = [np.sin(np.radians(d)) if d is not None else None for d in model_degrees]
    coss = [np.cos(np.radians(d)) if d is not None else None for d in model_degrees]
    pred_sin = predict_linear(beta_sin, sins)
    pred_cos = predict_linear(beta_cos, coss)
    if pred_sin is None or pred_cos is None:
        return None
    return float(np.degrees(np.arctan2(pred_sin, pred_cos)) % 360)


def get_latest_run_forecast(con, location_id, model, variable):
    """Returns {valid_time_utc: value} for the latest run of (location,
    model, variable)."""
    rows = con.execute(
        """
        WITH latest_run AS (
            SELECT run_id FROM forecast_runs
            WHERE location_id = ? AND model = ?
            ORDER BY init_time_utc DESC LIMIT 1
        )
        SELECT fv.valid_time_utc, fv.value
        FROM forecast_values fv JOIN latest_run lr ON lr.run_id = fv.run_id
        WHERE fv.variable = ?
        """,
        [location_id, model, variable],
    ).fetchall()
    return {vt: val for vt, val in rows}


def main():
    con = get_connection()
    now = datetime.now(timezone.utc)
    total_rows_written = 0
    try:
        for location_id in LOCATIONS:
            # ONE run per location per script execution, shared across
            # every variable for that location -- mirrors how a real
            # model's run works (one init_time_utc covers all its
            # variables). FIXED 2026-10-17: an earlier version created a
            # separate run per (location, variable), which collided on
            # forecast_runs' (model, location_id, init_time_utc) UNIQUE
            # constraint since every variable in the same script
            # execution shares the same `now` timestamp -- the second
            # variable's INSERT ON CONFLICT DO UPDATE tried to reuse/
            # recreate the first variable's run_id, which DuckDB
            # rejected with a foreign-key violation (the first run_id
            # was already referenced by that first variable's
            # forecast_values rows).
            run_id = con.execute(
                """
                INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (model, location_id, init_time_utc) DO UPDATE SET retrieved_at = excluded.retrieved_at
                RETURNING run_id
                """,
                [NDA_MODEL_NAME, location_id, now, now],
            ).fetchone()[0]

            for variable in VARIABLES:
                gt_loc = ground_truth_location(location_id, variable)

                if variable == "wind_dir_deg":
                    X_sin, X_cos, y_pair = fetch_direction_training_matrix(con, location_id, gt_loc)
                    if X_sin is None or len(X_sin) < MIN_TRAINING_ROWS:
                        continue
                    y_sin, y_cos = y_pair
                    X_sin1 = np.hstack([np.ones((len(X_sin), 1)), X_sin])
                    X_cos1 = np.hstack([np.ones((len(X_cos), 1)), X_cos])
                    beta_sin, _, _, _ = np.linalg.lstsq(X_sin1, y_sin, rcond=None)
                    beta_cos, _, _, _ = np.linalg.lstsq(X_cos1, y_cos, rcond=None)
                else:
                    X, y = fetch_training_matrix(con, location_id, variable, gt_loc)
                    beta = fit_linear_ensemble(X, y)
                    if beta is None:
                        continue

                # Gather each real model's CURRENT latest-run forecast,
                # keyed by valid_time_utc, so NDA produces one point per
                # valid_time (matching whichever models actually cover it).
                per_model_forecasts = {
                    m: get_latest_run_forecast(con, location_id, m, variable) for m in REAL_MODELS
                }
                all_valid_times = sorted(set().union(*[set(d.keys()) for d in per_model_forecasts.values()]))
                if not all_valid_times:
                    continue

                rows_written = 0
                for vt in all_valid_times:
                    model_values = [per_model_forecasts[m].get(vt) for m in REAL_MODELS]
                    if variable == "wind_dir_deg":
                        pred = predict_direction(beta_sin, beta_cos, model_values)
                    else:
                        pred = predict_linear(beta, model_values)
                    if pred is None:
                        continue
                    con.execute(
                        """
                        INSERT INTO forecast_values (run_id, valid_time_utc, variable, value)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (run_id, valid_time_utc, variable) DO UPDATE SET value = excluded.value
                        """,
                        [run_id, vt, variable, pred],
                    )
                    rows_written += 1
                total_rows_written += rows_written
                print(f"NDA {location_id}/{variable}: fit on ground truth {gt_loc}, wrote {rows_written} points")
    finally:
        con.close()
    print(f"NDA ensemble: {total_rows_written} total forecast points written across all location/variable combos")


if __name__ == "__main__":
    sys.exit(main())
