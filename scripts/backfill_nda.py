#!/usr/bin/env python3
"""One-time (re-runnable) BACKFILL for the NDA ensemble model.

ADDED 2026-10-17 (user: "you need to backfill calculate the NDA models
for the past"). REWRITTEN 2026-10-22 alongside ingest_nda.py's own
rewrite (see that module's docstring for the full "1-5" rationale) --
this script must mirror ingest_nda.py's CURRENT design (linear, time-
as-a-feature, per-active-model-set fitting, no missing-model averaging)
or the backfilled history and the live cron job would represent two
different, inconsistent models.

WALK-FORWARD, NOT HINDSIGHT -- the one thing that makes a backfill like
this legitimate rather than cheating: for each historical init_time
being backfilled, every regression is refit using ONLY ground-truth
data whose valid_time_utc is STRICTLY BEFORE that init_time. Refitting
once on ALL available history and applying it retroactively would give
NDA artificially inflated historical "accuracy" via look-ahead bias.

PER-ACTIVE-MODEL-SET, PER-LEAD-BUCKET, PER-INIT-TIME -- mirrors
ingest_nda.py's design exactly: at a given historical init_time,
whichever of the 6 real models actually had data for a given
valid_time_utc determines that hour's active set; which LEAD_BUCKETS
range its own lead_hours falls into determines its bucket; a SEPARATE
regression (with lead_hours as a feature) is fit per (active set,
bucket) pair, trained only on rows STRICTLY BEFORE this init_time where
that same active set AND lead-time bucket both held. No missing model
is ever backfilled with a substitute/average value. ADDED 2026-10-27
(user: "NDA model should extend out as far as other models go, but
over time eg. 10 days out if there is only one model to do the
regression with only use that model... there should be a lead time
parameter of sorts in NDA so that it can use different weights when
predicting today from 10 days from now") -- see ingest_nda.py's module
docstring for the full root-cause investigation (the old per-active-
-model-set-only design's "freshest run wins" dedup starved long-lead
training data, capping NDA's real reach to ~35h even when GFS/ECMWF
reach ~380h).

PERFORMANCE: fitting is now per (init_time, active-model-set) rather
than per (init_time, variable) -- more total fits than the old design,
but each fit's training-row lookup is a fast in-memory list
comprehension over a pre-fetched array (same pattern as before), not a
new DB query, so this stays tractable. See prefetch/caching comments
below for the specifics.

SAFE TO RE-RUN: same ON CONFLICT DO UPDATE pattern as the live script.
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
    LEAD_BUCKETS,
    lead_bucket_for,
    ground_truth_location,
)


def fetch_full_rows_any_lead(con, location_id, variable, gt_location):
    """Returns EVERY historical (valid_time_utc, lead_hours, model,
    value, observed_value) row for this location/variable, regardless
    of which models are present for a given valid_time -- unlike
    ingest_nda.py's live fetch_bucket_training_rows (which is called
    once per active-model-set with a fixed SQL shape for that set),
    this backfill needs one flexible in-memory structure it can filter
    BOTH by "active set" AND by "strictly before this init_time"
    without re-querying the DB for every (init_time, active-set)
    combination -- there could be dozens of init_times x several
    active-sets per location, and a DB round trip for each would be
    the real bottleneck (same lesson as the old backfill's
    prefetch_all_runs fix).

    Returns a dict: valid_time_utc -> {"obs": observed_value,
    "models": {model: (value, lead_hours)}}."""
    rows = con.execute(
        """
        SELECT fv.valid_time_utc, fr.model, fv.value,
               epoch(fv.valid_time_utc - fr.init_time_utc) / 3600.0 AS lead_hours,
               o.value AS observed_value,
               row_number() OVER (
                   PARTITION BY fr.model, fv.valid_time_utc
                   ORDER BY epoch(fv.valid_time_utc - fr.init_time_utc)
               ) AS rn
        FROM forecast_values fv
        JOIN forecast_runs fr ON fr.run_id = fv.run_id
        JOIN observations o
            ON o.location_id = ? AND o.variable = ?
           AND o.ts_utc BETWEEN fv.valid_time_utc - INTERVAL '7.5 minutes'
                             AND fv.valid_time_utc + INTERVAL '7.5 minutes'
        WHERE fr.location_id = ? AND fv.variable = ?
          AND fr.model IN ('gfs', 'ecmwf', 'hrrr', 'icon', 'nam', 'hrdps')
          AND fv.valid_time_utc >= fr.init_time_utc
        """,
        [gt_location, variable, location_id, variable],
    ).fetchall()
    by_valid_time = {}
    for vt, model, value, lead_hours, obs, rn in rows:
        if rn != 1:
            continue  # keep only the freshest (shortest-lead) run's value for this (model, valid_time), mirrors ingest_nda.py's live "latest run" semantics
        entry = by_valid_time.setdefault(vt, {"obs": obs, "models": {}})
        entry["models"][model] = (value, lead_hours)
    return by_valid_time


def fetch_full_direction_rows(con, location_id, gt_location):
    return fetch_full_rows_any_lead(con, location_id, "wind_dir_deg", gt_location)


WIND_SPEED_VARS = {"wind_speed_kt", "wind_gust_kt"}  # mirrors ingest_nda.py's WIND_SPEED_VARS -- which variables get the WIND_SANITY_RANGE_KT output clamp
WIND_SANITY_RANGE_KT = (0.0, None)  # upper bound removed 2026-10-24 (user: "remove your wind clamps at the top end"); mirrors ingest_nda.py's WIND_SANITY_RANGE_KT -- see that module's predict_linear() docstring for the full rationale


def fit_linear(rows, n_active_models):
    """rows: list of (lead_hours, model_val_1, ..., model_val_k,
    observed_value). Mirrors ingest_nda.py's fit_linear() exactly --
    see that function's docstring, including the 2026-10-24 fixes:
    (1) scaling the minimum-rows floor by parameter count (found via
    direct testing on this exact script: courageous_sailing's backfill
    produced ~7% physically-impossible wind_speed_kt predictions,
    traced to 1-model fits with too little walk-forward training
    data), and (2) returning (beta, max_train_lead) so predict_linear()
    can refuse to extrapolate far beyond what a fit actually saw
    (user: "shouldn't be 80 knot predictions...unless which models said
    the same?" -- traced to 1-model fits trained on 0-5h data being
    asked to predict at 300+h)."""
    min_rows = max(MIN_TRAINING_ROWS, (2 + n_active_models) * 10)
    if len(rows) < min_rows:
        return None
    arr = np.array(rows, dtype=float)
    X = arr[:, :-1]
    y = arr[:, -1]
    X1 = np.hstack([np.ones((len(X), 1)), X])
    beta, _residuals, _rank, _sv = np.linalg.lstsq(X1, y, rcond=None)
    max_train_lead = float(X[:, 0].max())
    return beta, max_train_lead


MAX_EXTRAPOLATION_FACTOR = 2.0  # mirrors ingest_nda.py's MAX_EXTRAPOLATION_FACTOR -- see that module's predict_linear() docstring


def predict_linear(beta_and_range, lead_hours, model_values, clamp_range=None):
    """Mirrors ingest_nda.py's predict_linear() -- clamp_range, if
    given, clips the output to (lo, hi), used for wind_speed_kt/
    wind_gust_kt only. beta_and_range is fit_linear()'s (beta,
    max_train_lead) -- refuses to predict (returns None) when
    extrapolating too far beyond the fit's own training range."""
    if beta_and_range is None:
        return None
    beta, max_train_lead = beta_and_range
    if lead_hours > max_train_lead * MAX_EXTRAPOLATION_FACTOR and lead_hours > max_train_lead + 24:
        return None
    row = np.array([1.0, lead_hours] + list(model_values))
    pred = float(row @ beta)
    if clamp_range is not None:
        lo, hi = clamp_range
        pred = max(lo, pred)
        if hi is not None:
            pred = min(hi, pred)
    return pred


def fit_direction(rows):
    """rows: list of (lead_hours, model_deg_1, ..., model_deg_k,
    observed_deg). Mirrors ingest_nda.py's fit_direction() exactly."""
    if len(rows) < MIN_TRAINING_ROWS:
        return None, None
    arr = np.array(rows, dtype=float)
    lead_hours = arr[:, 0:1]
    model_degrees = arr[:, 1:-1]
    obs_degrees = arr[:, -1]
    X_sin = np.hstack([lead_hours, np.sin(np.radians(model_degrees))])
    X_cos = np.hstack([lead_hours, np.cos(np.radians(model_degrees))])
    y_sin = np.sin(np.radians(obs_degrees))
    y_cos = np.cos(np.radians(obs_degrees))
    X_sin1 = np.hstack([np.ones((len(X_sin), 1)), X_sin])
    X_cos1 = np.hstack([np.ones((len(X_cos), 1)), X_cos])
    beta_sin, *_ = np.linalg.lstsq(X_sin1, y_sin, rcond=None)
    beta_cos, *_ = np.linalg.lstsq(X_cos1, y_cos, rcond=None)
    return beta_sin, beta_cos


def predict_direction(beta_sin, beta_cos, lead_hours, model_degrees):
    if beta_sin is None or beta_cos is None:
        return None
    sins = [np.sin(np.radians(d)) for d in model_degrees]
    coss = [np.cos(np.radians(d)) for d in model_degrees]
    pred_sin = float(np.array([1.0, lead_hours] + sins) @ beta_sin)
    pred_cos = float(np.array([1.0, lead_hours] + coss) @ beta_cos)
    return float(np.degrees(np.arctan2(pred_sin, pred_cos)) % 360)


def prefetch_all_runs(con, location_id):
    """Same as the old backfill's version -- {(model, variable,
    init_time_utc): {valid_time_utc: value}}, used to reconstruct what
    each model's forecast looked like AT a specific historical
    init_time (not necessarily its latest run).

    FIXED 2026-10-24 (same bug/investigation as
    fetch_full_rows_any_lead's identical fix -- user: "shouldn't be 80
    knot predictions...unless which models said the same?"): without
    the valid_time_utc >= init_time_utc filter, a model's bundled
    PRE-init analysis hours (negative lead_hours -- data for times
    before the run even started) were being treated as real forecast
    points for that init_time, which is nonsensical for "what did this
    model predict going forward from here" and corrupted the
    active-model-set grouping/lead_hours used when actually making a
    prediction (as opposed to fetch_full_rows_any_lead's training-data
    bug, this is the SAME flaw on the prediction side)."""
    rows = con.execute(
        """
        SELECT fr.model, fv.variable, fr.init_time_utc, fv.valid_time_utc, fv.value
        FROM forecast_values fv
        JOIN forecast_runs fr ON fr.run_id = fv.run_id
        WHERE fr.location_id = ? AND fr.model IN ('gfs','ecmwf','hrrr','icon','nam','hrdps')
          AND fv.valid_time_utc >= fr.init_time_utc
        """,
        [location_id],
    ).fetchall()
    out = {}
    for model, variable, init_time, valid_time, value in rows:
        out.setdefault((model, variable, init_time), {})[valid_time] = value
    return out


def main(locations=None):
    con = get_connection()
    total_runs_written = 0
    total_points_written = 0
    try:
        for location_id in (locations or LOCATIONS):
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

            COMMIT_EVERY_N_RUNS = 20
            con.execute("BEGIN TRANSACTION")

            # Pre-fetch ONCE per location/variable -- walk-forward
            # filtering and active-model-set grouping below are all
            # cheap in-memory operations over this, not new DB queries.
            #
            # PERFORMANCE FIX (found via direct profiling -- a 149-
            # init-time test on courageous_sailing didn't finish in
            # 120s): the original version re-scanned the ENTIRE
            # full_by_vt dict (every historical valid_time this
            # location/variable has ever had, easily thousands of
            # entries) for EVERY (init_time, active-model-set)
            # combination -- with ~149 init_times x 6 variables x
            # several active-sets each, that's a full-dataset scan
            # repeated many hundreds of times per location. Fixed by
            # sorting each variable's rows once into a (valid_time,
            # entry) list, so the walk-forward "strictly before
            # init_time_utc" cutoff can be found with a single
            # bisect_left call (O(log n)) instead of an O(n) scan, and
            # only that prefix is ever iterated.
            import bisect
            cached_rows = {}
            cached_dir_rows = None
            for variable in VARIABLES:
                gt_loc = ground_truth_location(location_id, variable)
                if variable == "wind_dir_deg":
                    full_by_vt = fetch_full_direction_rows(con, location_id, gt_loc)
                else:
                    full_by_vt = fetch_full_rows_any_lead(con, location_id, variable, gt_loc)
                sorted_vts = sorted(full_by_vt.keys())
                sorted_entries = [full_by_vt[vt] for vt in sorted_vts]
                if variable == "wind_dir_deg":
                    cached_dir_rows = (sorted_vts, sorted_entries, gt_loc)
                else:
                    cached_rows[variable] = (sorted_vts, sorted_entries, gt_loc)

            all_runs = prefetch_all_runs(con, location_id)

            pending_run_values = []
            for i, init_time_utc in enumerate(all_init_times):
                run_id = None
                for variable in VARIABLES:
                    sorted_vts, sorted_entries, gt_loc = cached_dir_rows if variable == "wind_dir_deg" else cached_rows[variable]
                    # Everything at index < cutoff has valid_time_utc
                    # STRICTLY BEFORE init_time_utc (walk-forward) --
                    # replaces the old full-dict scan with one bisect.
                    cutoff = bisect.bisect_left(sorted_vts, init_time_utc)
                    training_entries = sorted_entries[:cutoff]

                    per_model_forecasts = {
                        m: all_runs.get((m, variable, init_time_utc), {}) for m in REAL_MODELS
                    }
                    all_valid_times = sorted(set().union(*[set(d.keys()) for d in per_model_forecasts.values()]))
                    if not all_valid_times:
                        continue

                    # Group this init_time's valid_times by their exact
                    # active-model-set (point 5) AND lead-time bucket
                    # (see ingest_nda.py's LEAD_BUCKETS module comment
                    # -- "different weights when predicting today from
                    # 10 days from now") -- same logic as ingest_nda.py's
                    # live main(), just using init_time_utc as the
                    # reference point instead of "now" (this IS "now"
                    # for a historical walk-forward backfill run).
                    groups = {}
                    for vt in all_valid_times:
                        active = tuple(m for m in REAL_MODELS if per_model_forecasts[m].get(vt) is not None)
                        if not active:
                            continue
                        lead_hours_for_vt = (vt - init_time_utc).total_seconds() / 3600.0
                        bucket = lead_bucket_for(lead_hours_for_vt)
                        groups.setdefault((active, bucket), []).append(vt)

                    points_this_var = 0
                    for (active_models, bucket), vts in groups.items():
                        # WALK-FORWARD training rows: every historical
                        # valid_time STRICTLY BEFORE this init_time
                        # (training_entries, already sliced above) where
                        # every model in active_models had real data,
                        # AND whose own lead_hours falls in this SAME
                        # bucket (mirrors ingest_nda.py's live
                        # fetch_bucket_training_rows's lead_filter --
                        # without this, a bucket's fit would be trained
                        # on whatever lead_hours its "strictly before"
                        # training rows happened to have, not genuinely
                        # lead-matched examples).
                        lo, hi = bucket
                        if variable == "wind_dir_deg":
                            train_rows = []
                            for entry in training_entries:
                                if not all(m in entry["models"] for m in active_models):
                                    continue
                                leads = [entry["models"][m][1] for m in active_models]
                                avg_lead = sum(leads) / len(leads)
                                if avg_lead < lo or (hi is not None and avg_lead >= hi):
                                    continue
                                degs = [entry["models"][m][0] for m in active_models]
                                train_rows.append([avg_lead] + degs + [entry["obs"]])
                            beta_sin, beta_cos = fit_direction(train_rows)
                            fitted = (beta_sin, beta_cos)
                            is_missing = beta_sin is None
                        else:
                            train_rows = []
                            for entry in training_entries:
                                if not all(m in entry["models"] for m in active_models):
                                    continue
                                leads = [entry["models"][m][1] for m in active_models]
                                avg_lead = sum(leads) / len(leads)
                                if avg_lead < lo or (hi is not None and avg_lead >= hi):
                                    continue
                                vals = [entry["models"][m][0] for m in active_models]
                                train_rows.append([avg_lead] + vals + [entry["obs"]])
                            fitted = fit_linear(train_rows, len(active_models))
                            is_missing = fitted is None

                        if is_missing:
                            continue

                        if run_id is None:
                            # FIXED 2026-10-24 (found via direct testing
                            # on courageous_sailing): the previous
                            # ON CONFLICT ... DO UPDATE ... RETURNING
                            # run_id pattern raised a ConstraintException
                            # ("still referenced by a foreign key in a
                            # different table") whenever the conflicting
                            # row ALREADY had forecast_values children --
                            # a real DuckDB 1.5.5 limitation around
                            # UPSERT-into-a-row-with-FK-referencing-
                            # children, not a logic bug in this script.
                            # Query-then-branch instead: look up an
                            # existing run first, and only fall back to
                            # a plain INSERT if none exists -- avoids
                            # the UPSERT path (and its RETURNING clause)
                            # entirely for the conflict case.
                            existing_run = con.execute(
                                "SELECT run_id FROM forecast_runs WHERE model = ? AND location_id = ? AND init_time_utc = ?",
                                [NDA_MODEL_NAME, location_id, init_time_utc],
                            ).fetchone()
                            if existing_run:
                                run_id = existing_run[0]
                                con.execute(
                                    "UPDATE forecast_runs SET retrieved_at = ? WHERE run_id = ?",
                                    [init_time_utc, run_id],
                                )
                            else:
                                run_id = con.execute(
                                    """
                                    INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
                                    VALUES (?, ?, ?, ?)
                                    RETURNING run_id
                                    """,
                                    [NDA_MODEL_NAME, location_id, init_time_utc, init_time_utc],
                                ).fetchone()[0]
                                total_runs_written += 1

                        for vt in vts:
                            lead_hours = (vt - init_time_utc).total_seconds() / 3600.0
                            model_values = [per_model_forecasts[m][vt] for m in active_models]
                            if variable == "wind_dir_deg":
                                pred = predict_direction(fitted[0], fitted[1], lead_hours, model_values)
                            else:
                                clamp = WIND_SANITY_RANGE_KT if variable in WIND_SPEED_VARS else None
                                pred = predict_linear(fitted, lead_hours, model_values, clamp_range=clamp)
                            if pred is None:
                                continue
                            pending_run_values.append((run_id, vt, variable, pred))
                            points_this_var += 1
                    total_points_written += points_this_var

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
    # Optional CLI override for a quick single/few-location smoke test
    # before committing to a full run: `python3 backfill_nda.py
    # courageous_sailing`.
    override = sys.argv[1:] or None
    sys.exit(main(locations=override))
