#!/usr/bin/env python3
"""Sailing Weather Dashboard backend.

Tiny stdlib-only HTTP server (no Flask/FastAPI installed in the target
interpreter) that serves:
  - static frontend files from ./static
  - a small JSON API backed by the read-only DuckDB weather database

Run with:
    /home/firescar96/.pyenv/versions/3.11.1/bin/python3 app.py

Listens on http://localhost:8420 by default (override with PORT env var).

IMPORTANT: every DB connection is opened with read_only=True. This process
never writes to the database, so it never conflicts with the ingestion
cron jobs that also touch weather.duckdb on their own schedule.
"""
import json
import os
import sys
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "db", "weather.duckdb")
STATIC_DIR = os.path.join(HERE, "static")
PORT = int(os.environ.get("PORT", "8420"))

sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
from cbi_hours import is_cbi_open


def db():
    """Open a fresh read-only connection. Cheap for this data volume and
    avoids any question of duckdb connection thread-safety across
    concurrent requests.

    MUST set session TimeZone to UTC (found 2026-09-11): DuckDB's default
    session TimeZone is the OS local zone (America/New_York here), and
    ALL of `now()`'s time-window queries (api_forecast, api_observations,
    api_flags_history) compare it directly against
    valid_time_utc/ts_utc columns that are naive UTC timestamps. Without
    forcing UTC, now() returned local wall-clock time (e.g. 08:15 EDT)
    while being compared against UTC-stored columns -- a raw ~4h
    (EDT) / ~5h (EST) offset baked into every "now +/- N hours" window.
    This is exactly why model forecast lines extended further back than
    requested (user-reported 2026-09-11: "trim the models look back time
    so they don't extend before the graph start") -- the past_hours
    window was silently ~4h wider than asked. The ingestion side
    (db/common.py get_connection()) already does this for the same
    reason; this was the one connection path that didn't."""
    con = duckdb.connect(DB_PATH, read_only=True)
    con.execute("SET TimeZone='UTC'")
    return con


def rows_as_dicts(cursor):
    cols = [d[0] for d in cursor.description]
    out = []
    for row in cursor.fetchall():
        d = {}
        for c, v in zip(cols, row):
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            d[c] = v
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# API handlers -- each takes a dict of query params, returns a JSON-able obj
# ---------------------------------------------------------------------------

def api_locations(params):
    con = db()
    try:
        cur = con.execute(
            "SELECT location_id, name, lat, lon, kind FROM locations ORDER BY location_id"
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_current_conditions(params):
    """Most recent observation value per (location, variable)."""
    con = db()
    try:
        cur = con.execute(
            """
            SELECT o.location_id, l.name AS location_name, o.variable, o.value, o.ts_utc, o.source
            FROM observations o
            JOIN locations l ON l.location_id = o.location_id
            QUALIFY row_number() OVER (
                PARTITION BY o.location_id, o.variable
                ORDER BY o.ts_utc DESC
            ) = 1
            ORDER BY o.location_id, o.variable
            """
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_flags_latest(params):
    con = db()
    try:
        cur = con.execute(
            """
            SELECT location_id, ts_utc, flag_color, source
            FROM flags
            QUALIFY row_number() OVER (PARTITION BY location_id ORDER BY ts_utc DESC) = 1
            """
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_flags_history(params):
    """Flag color readings for a location over a time window, for chart
    background shading (see static/app.js loadForecastChart). Only
    location_id='cbi_dockhouse' has any rows currently -- CBI is the only
    tracked flag source -- but this endpoint works generically for any
    location that has flags.

    Anchored to real wall-clock now() (FIXED 2026-09-11), not
    max(ts_utc) FROM flags -- the same stale-anchor bug already found
    and fixed in api_forecast(). Flags only get polled hourly and can
    lag "now" by a few minutes normally, but once the forecast chart's
    own x-axis switched to anchoring on real now(), a flags-history
    window anchored on ITS OWN latest reading meant the two could drift
    out of sync -- e.g. once the forecast window moved to a forward-
    looking "next 72h" range while the last flag reading was still
    hours in the past, nearly the entire flag timeline fell outside the
    chart's visible x-domain (user-reported 2026-09-11: "gaps inbetween
    the colors and the date range is wrong")."""
    location = params.get("location")
    hours = int(params.get("hours", "72"))
    if not location:
        return {"error": "location query param is required"}
    con = db()
    try:
        cur = con.execute(
            """
            SELECT ts_utc, flag_color
            FROM flags
            WHERE location_id = ?
              AND ts_utc >= now() - (? * INTERVAL '1 hour')
            ORDER BY ts_utc
            """,
            [location, hours],
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_accuracy(params):
    """Accuracy (MAE/RMSE) per model x lead_bucket for a location+variable,
    optionally restricted to a recent time window.

    ADDED time-window filtering 2026-09-21 (user: "add a dropdown that
    let's me pick a time window, with a default of 7 days" -- following
    on from confirming v_accuracy_by_lead_time has NO time filter at all,
    it's genuinely all-time/all-history accuracy, accumulating forever).
    The view itself can't be parameterized (it's a plain CREATE VIEW,
    and doesn't even expose the underlying observation timestamp to
    filter on after the fact -- only the already-aggregated MAE/RMSE per
    bucket), so this reimplements the view's matching/bucketing logic
    inline with an added `hours` bound on the OBSERVATION's own
    timestamp (o.ts_utc >= now() - hours), rather than editing the view
    (which stays as the genuine unbounded/all-time computation, still
    used as-is by api_accuracy_variables). `hours` omitted or empty
    means "all time," matching the view's original unbounded behavior
    exactly -- verified the two produce identical numbers when no
    window is applied.
    """
    location = params.get("location")
    variable = params.get("variable")
    hours_param = params.get("hours", "168")
    hours = int(hours_param) if hours_param else None
    if not location or not variable:
        return {"error": "location and variable query params are required"}
    con = db()
    try:
        if hours is None:
            # All-time: identical to the pre-existing behavior, straight
            # from the unbounded view.
            cur = con.execute(
                """
                SELECT model, lead_bucket, n, mae, rmse
                FROM v_accuracy_by_lead_time
                WHERE location_id = ? AND variable = ?
                ORDER BY model, lead_bucket
                """,
                [location, variable],
            )
        else:
            cur = con.execute(
                """
                WITH candidates AS (
                    SELECT
                        fv.run_id,
                        fv.valid_time_utc,
                        fv.value AS forecast_value,
                        o.ts_utc AS obs_ts_utc,
                        o.value AS observed_value,
                        row_number() OVER (
                            PARTITION BY fv.run_id, fv.valid_time_utc
                            ORDER BY abs(epoch(fv.valid_time_utc) - epoch(o.ts_utc))
                        ) AS rn
                    FROM forecast_values fv
                    JOIN forecast_runs fr ON fr.run_id = fv.run_id
                    JOIN observations o
                        ON o.location_id = fr.location_id
                       AND o.variable = fv.variable
                       AND o.ts_utc BETWEEN fv.valid_time_utc - INTERVAL '7.5 minutes'
                                         AND fv.valid_time_utc + INTERVAL '7.5 minutes'
                    WHERE fr.location_id = ?
                      AND fv.variable = ?
                      AND o.ts_utc >= now() - (? * INTERVAL '1 hour')
                ),
                matched AS (
                    SELECT run_id, valid_time_utc, forecast_value, observed_value
                    FROM candidates
                    WHERE rn = 1
                ),
                with_lead AS (
                    SELECT
                        m.*,
                        fr.model,
                        date_diff('hour', fr.init_time_utc, m.valid_time_utc) AS lead_hours
                    FROM matched m
                    JOIN forecast_runs fr ON fr.run_id = m.run_id
                ),
                bucketed AS (
                    SELECT
                        model,
                        CASE
                            WHEN lead_hours >= 0 AND lead_hours < 6 THEN '0-6h'
                            WHEN lead_hours >= 6 AND lead_hours < 12 THEN '6-12h'
                            WHEN lead_hours >= 12 AND lead_hours < 24 THEN '12-24h'
                            WHEN lead_hours >= 24 AND lead_hours < 48 THEN '24-48h'
                            WHEN lead_hours >= 48 AND lead_hours < 72 THEN '48-72h'
                            WHEN lead_hours >= 72 THEN '72h+'
                            ELSE 'other'
                        END AS lead_bucket,
                        forecast_value - observed_value AS error
                    FROM with_lead
                )
                SELECT
                    model,
                    lead_bucket,
                    count(*) AS n,
                    round(avg(abs(error)), 3) AS mae,
                    round(sqrt(avg(error * error)), 3) AS rmse
                FROM bucketed
                GROUP BY model, lead_bucket
                ORDER BY model, lead_bucket
                """,
                [location, variable, hours],
            )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_accuracy_variables(params):
    """Which (location, variable) combos actually have accuracy rows, so
    the frontend only offers selectable options that have real data."""
    con = db()
    try:
        cur = con.execute(
            """
            SELECT DISTINCT location_id, variable
            FROM v_accuracy_by_lead_time
            ORDER BY location_id, variable
            """
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_forecast(params):
    """Forecast time series for a location/variable, spanning a past
    window (observed history + REAL historical model forecasts) through
    a future window (each model's current/latest prediction).

    HISTORY-STITCHING (added 2026-09-21, user: "i don't see the
    historical model data only the observed weather for 7 days" / "you
    should be storing the historical runs right"): confirmed the
    historical runs genuinely ARE stored (45-87 distinct runs per model
    going back to Sept 6-10) -- this endpoint just wasn't using them.
    The OLD query only ever looked at each model's single LATEST run,
    which only carries ~2 days of backfilled hindcast data (however far
    back its own `past_hours`/`past_days` ingestion window reaches) --
    so selecting "Last 7 days" showed 5 of those 7 days with NO model
    lines at all, even though real historical forecast data for that
    period existed in `forecast_runs`/`forecast_values` the whole time.

    Fix: for EVERY valid_time_utc in the requested window (past or
    future), pick each model's forecast value from whichever of ITS OWN
    runs has the LARGEST init_time_utc that is still <= valid_time_utc
    -- i.e. "the freshest forecast that model had actually issued as of
    that moment," not backfill/hindcast data from a run issued AFTER
    the fact. This is a genuine "what did the model actually predict"
    history, and it naturally unifies with the future window too: for
    any valid_time in the future, the only run with init_time <= it is
    each model's single latest run (nothing newer exists yet), so this
    one query correctly reduces to the old future-only behavior without
    needing a separate code path.
    """
    location = params.get("location")
    variable = params.get("variable")
    hours = int(params.get("hours", "72"))
    past_hours = int(params.get("past_hours", "0"))
    if not location or not variable:
        return {"error": "location and variable query params are required"}
    con = db()
    try:
        cur = con.execute(
            """
            WITH candidates AS (
                SELECT
                    fr.model,
                    fr.init_time_utc,
                    fv.valid_time_utc,
                    fv.value,
                    row_number() OVER (
                        PARTITION BY fr.model, fv.valid_time_utc
                        ORDER BY fr.init_time_utc DESC
                    ) AS rn
                FROM forecast_values fv
                JOIN forecast_runs fr ON fr.run_id = fv.run_id
                WHERE fr.location_id = ?
                  AND fv.variable = ?
                  AND fr.init_time_utc <= fv.valid_time_utc
                  AND fv.valid_time_utc <= now() + (? * INTERVAL '1 hour')
                  AND fv.valid_time_utc >= now() - (? * INTERVAL '1 hour')
            )
            SELECT model, init_time_utc, valid_time_utc, value
            FROM candidates
            WHERE rn = 1
            ORDER BY model, valid_time_utc
            """,
            [location, variable, hours, past_hours],
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_forecast_stability(params):
    """Day-over-day forecast stability: for each upcoming hour in the
    forecast window, how much has each model's prediction for that exact
    hour changed across its last few runs?

    This replaced the old Forecast Convergence endpoint entirely
    (removed 2026-09-14, per explicit user instruction: "if you don't
    use the backend convergence route, delete it" -- the frontend had
    already stopped calling it once this stability chart replaced that
    UI panel, so it was genuinely dead code, not just superseded).
    Convergence answered "how did predictions for ONE fixed target time
    evolve over the run history leading up to it"; this answers a
    different question -- "right now, looking at the whole upcoming
    forecast, which hours/models are the models still disagreeing with
    THEMSELVES run-to-run" -- i.e. a genuine forecast-confidence signal
    distinct from model accuracy (error vs. reality) or the Bayesian
    prediction panels.

    For each model, takes its last `num_runs` distinct init_time_utc
    runs (most recent first) and, for every valid_time_utc that appears
    in ALL of them, computes the spread (max-min) and stddev of the
    predicted value across those runs. A tight spread means the model
    has been predicting the same thing for that hour run after run
    (stable/high-confidence); a wide spread means its own story keeps
    changing (unstable/low-confidence), independent of whether it's
    actually accurate.
    """
    location = params.get("location")
    variable = params.get("variable")
    num_runs = int(params.get("num_runs", "3"))
    hours = int(params.get("hours", "72"))
    if not location or not variable:
        return {"error": "location and variable query params are required"}
    if num_runs < 2:
        return {"error": "num_runs must be >= 2 to measure any stability"}
    con = db()
    try:
        # MIN_RUNS_FOR_SPREAD (was: require count(*) == num_runs exactly)
        # -- FIXED 2026-09-24 (user: "hrdps shows more on the top graph
        # of forecast prediction than on forecast stability, seems
        # trimmed"). Root cause: HRDPS only forecasts ~48h out per run
        # (confirmed directly -- its runs' own max valid_time_utc is
        # consistently init_time + 48h, vs. ECMWF/GFS/etc reaching 14+
        # days), while this query previously required a target hour to
        # appear in ALL `num_runs` selected runs before showing ANY
        # spread for it. With num_runs=4 selected, any hour beyond
        # HRDPS's OLDEST of those 4 runs' 48h horizon got silently
        # dropped entirely -- even though 2-3 of the newer selected runs
        # DID cover that hour just fine. This truncated HRDPS's visible
        # stability range far short of its real forecast reach, and far
        # short of what the (unconstrained) Forecast Time Series chart
        # shows for the same model. Changed to only require at least 2
        # of the selected runs cover an hour (the minimum needed to
        # measure any spread at all) -- a short-horizon model's line now
        # extends as far as its own real data does, using however many
        # of the selected runs actually reach that far, rather than
        # being capped by whichever selected run happens to be shortest.
        MIN_RUNS_FOR_SPREAD = 2
        cur = con.execute(
            """
            WITH recent_runs AS (
                SELECT run_id, model, init_time_utc,
                       row_number() OVER (PARTITION BY model ORDER BY init_time_utc DESC) AS run_rank
                FROM forecast_runs
                WHERE location_id = ?
            ),
            chosen_runs AS (
                SELECT run_id, model, init_time_utc
                FROM recent_runs
                WHERE run_rank <= ?
            ),
            values_across_runs AS (
                SELECT cr.model, fv.valid_time_utc, fv.value, cr.init_time_utc
                FROM forecast_values fv
                JOIN chosen_runs cr ON cr.run_id = fv.run_id
                WHERE fv.variable = ?
                  AND fv.valid_time_utc >= now()
                  AND fv.valid_time_utc <= now() + (? * INTERVAL '1 hour')
            )
            SELECT model, valid_time_utc,
                   count(*) AS n_runs_with_this_hour,
                   min(value) AS min_value,
                   max(value) AS max_value,
                   round(max(value) - min(value), 3) AS spread,
                   round(stddev(value), 3) AS stddev_value,
                   round(avg(value), 3) AS avg_value
            FROM values_across_runs
            GROUP BY model, valid_time_utc
            HAVING count(*) >= ?
            ORDER BY model, valid_time_utc
            """,
            [location, num_runs, variable, hours, MIN_RUNS_FOR_SPREAD],
        )
        rows = rows_as_dicts(cur)

        # Also report which init_time_utc runs were actually used per
        # model, so the frontend/tooltip can say "based on the last 3
        # runs: 12:00, 18:00, 00:00" instead of just a bare number.
        runs_cur = con.execute(
            """
            WITH recent_runs AS (
                SELECT model, init_time_utc,
                       row_number() OVER (PARTITION BY model ORDER BY init_time_utc DESC) AS run_rank
                FROM forecast_runs
                WHERE location_id = ?
            )
            SELECT model, init_time_utc FROM recent_runs WHERE run_rank <= ?
            ORDER BY model, init_time_utc DESC
            """,
            [location, num_runs],
        )
        runs_used = {}
        for model, init_time in runs_cur.fetchall():
            runs_used.setdefault(model, []).append(
                init_time.isoformat() if hasattr(init_time, "isoformat") else init_time
            )

        return {
            "location": location,
            "variable": variable,
            "num_runs": num_runs,
            "runs_used_by_model": runs_used,
            "rows": rows,
        }
    finally:
        con.close()


def api_observations(params):
    """Raw historical observations for overlaying on the forecast chart.
    `hours` here means "how far back from the latest observation", used
    independently from the forecast's forward-looking window."""
    location = params.get("location")
    variable = params.get("variable")
    hours = int(params.get("hours", "72"))
    if not location or not variable:
        return {"error": "location and variable query params are required"}
    con = db()
    try:
        cur = con.execute(
            """
            SELECT ts_utc, value
            FROM observations
            WHERE location_id = ? AND variable = ?
              AND ts_utc >= (SELECT max(ts_utc) FROM observations WHERE location_id = ? AND variable = ?) - (? * INTERVAL '1 hour')
            ORDER BY ts_utc
            """,
            [location, variable, location, variable, hours],
        )
        return rows_as_dicts(cur)
    finally:
        con.close()


def api_variables_for_location(params):
    """Which observation variables exist for a given location (used to
    grey out / hide selector options that have no data)."""
    location = params.get("location")
    if not location:
        return {"error": "location query param is required"}
    con = db()
    try:
        cur = con.execute(
            """
            SELECT DISTINCT variable
            FROM observations
            WHERE location_id = ?
            ORDER BY variable
            """,
            [location],
        )
        return [r[0] for r in cur.fetchall()]
    finally:
        con.close()


FLAG_COLORS_ALL = ["red", "yellow", "green", "closed"]
WIND_BUCKET_WIDTH_KT = 4.0  # historical training data is sparse (~40 flag
                            # readings total), so buckets need to be wide
                            # enough for each to have a few examples


def _wind_bucket(value):
    """Buckets a wind speed into a fixed-width band (e.g. 8.3kt -> '8-12kt')
    for the historical training table below."""
    lo = int(value // WIND_BUCKET_WIDTH_KT) * WIND_BUCKET_WIDTH_KT
    return f"{int(lo)}-{int(lo + WIND_BUCKET_WIDTH_KT)}kt"


DAYPART_DAY_START_HOUR = 7   # local hour (America/New_York) day starts
DAYPART_DAY_END_HOUR = 19    # local hour (exclusive) day ends, night starts


def _daypart(local_hour):
    """Coarse day/night split by LOCAL hour (America/New_York), added
    2026-09-24 for api_wind_prediction's time-of-day conditioning (see
    that function's docstring for the full rationale/investigation).
    Deliberately just 2 buckets, not e.g. 4 (morning/afternoon/evening/
    night) or hourly -- with a training set of a few thousand pairs per
    model/location, finer daypart buckets would starve each
    (daypart, wind_bucket) cell of enough samples for the Dirichlet
    smoothing to mean much; day-vs-night was the actual regime split
    empirically confirmed in the investigation (thermal/convective
    effects roughly track daylight, not a finer schedule), so this is
    the coarsest split that still captures the real effect.
    """
    if local_hour is None:
        return None
    return "day" if DAYPART_DAY_START_HOUR <= local_hour < DAYPART_DAY_END_HOUR else "night"


WIND_PREDICTION_DIRECTIONS = 8  # 8-point compass (N/NE/E/SE/S/SW/W/NW, 45 deg each)
_COMPASS_SECTOR_LABELS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _compass_sector_label(deg, n=WIND_PREDICTION_DIRECTIONS):
    """Human-readable compass sector for api_wind_prediction's
    direction conditioning (added 2026-09-24, user: "add in wind
    direction to all the buckets"). Deliberately coarser (8 sectors,
    45deg each) than the Wind Rose panel's 16 sectors -- every extra
    conditioning dimension divides the same finite training data
    further, and with only ~400 deduped real events per model/location
    (see the training-pair dedup fix in api_wind_prediction), 16-way
    direction on top of 2-way daypart and 5-6 wind buckets would spread
    that data far too thin to mean anything. Reuses _compass_sector's
    bucketing math but only supports n=8 (hardcoded label table) since
    that's the only granularity this endpoint uses.
    """
    if n != WIND_PREDICTION_DIRECTIONS:
        raise ValueError("_compass_sector_label only supports the 8-sector case")
    idx = _compass_sector(deg, n=n)
    return _COMPASS_SECTOR_LABELS_8[idx]


def api_flag_prediction(params):
    """Bayesian flag-color prediction for CBI, driven by a selectable
    forecast model's wind-speed forecast.

    This is a SEPARATE panel from the existing Forecast/Accuracy charts
    (user explicit direction 2026-09-11: "do not ruin or mess up the
    existing charts... I want additional overlay or a separate chart").
    It answers a different question than MAE/accuracy does: not "how far
    off is the model on average" but "given this model's forecast wind
    at some future hour, what's the probability the flag will be each
    color at that time" -- i.e. P(flag | wind), computed via Bayes'
    theorem:

        P(flag=c | wind=w) = P(wind=w | flag=c) * P(flag=c) / P(wind=w)

    Implementation: rather than separately estimating the likelihood and
    marginal P(wind) (which would require picking a parametric wind
    distribution), we directly estimate the joint P(wind_bucket, flag)
    empirically from CBI's own history (matching each flag reading to
    the nearest real MIT Pavilion wind observation within 20 minutes --
    MIT is the venue's own on-site sensor, a few hundred meters from the
    CBI dockhouse on the same basin) and add a symmetric Dirichlet(alpha)
    prior (Laplace/additive smoothing) over the 4 flag colors within
    each wind bucket. This IS an explicit Bayesian posterior update
    (prior=uniform over colors, likelihood=observed per-bucket counts),
    not just a raw frequency table -- and the smoothing matters a lot
    here specifically because the training set is small (~130 flag
    readings as of 2026-09-15), so buckets with 0-2 real examples would
    otherwise give overconfident all-or-nothing probabilities. alpha=1
    (add-one smoothing) pulls sparse buckets toward the overall marginal
    P(flag=c) instead.

    BUCKETING SIGNAL is max(sustained wind_speed_kt, wind_gust_kt), not
    sustained wind alone (fixed 2026-09-15, user-reported: "i see red
    23% for today... but i don't think today has a chance of red at all
    personally from my experience"). Root cause found: matching each
    flag reading to the single NEAREST instantaneous wind_speed_kt
    sample is noisy right at the sample point -- a real Sept 14 evening
    had gusts of 16-17kt (correctly flagged red by the dockmaster) but
    the one sustained-wind sample nearest a couple of those flag
    timestamps happened to catch a brief lull (~4-6kt), so those red
    readings were polluting the LIGHT-wind bucket and inflating its red
    probability well above what a sailor's actual experience would
    suggest. Using max(wind, gust) as the bucketing key moves those
    examples into the (correctly gustier) 16-20kt bucket where they
    belong, and requires the same transform be applied to the forecast
    side too (a model's forecast gust, not just its forecast sustained
    wind) for the training and inference signals to stay consistent.
    This is a compromise vs. a full 2D (wind_bucket x gust_bucket) joint
    distribution, which would be strictly more faithful (gustiness is a
    real, distinct signal from peak wind) but isn't viable yet at ~130
    total examples -- most 2D cells would be too sparse for Dirichlet
    smoothing to reflect real signal rather than just falling back to
    the marginal prior. Revisit the full 2D grid once there's roughly
    200-350+ training examples (some months of accumulated CBI-open-hours
    history at the current hourly-poll rate).
    """
    model = params.get("model")
    hours = int(params.get("hours", "48"))
    location = "cbi_dockhouse"  # the only location with flag data
    variable = "wind_speed_kt"
    gust_variable = "wind_gust_kt"
    if not model:
        return {"error": "model query param is required"}
    con = db()
    try:
        # 1. Forecast wind AND gust for the selected model, forward-looking.
        forecast_rows = con.execute(
            """
            WITH latest_run AS (
                SELECT run_id, init_time_utc
                FROM forecast_runs
                WHERE location_id = ? AND model = ?
                QUALIFY row_number() OVER (ORDER BY init_time_utc DESC) = 1
            )
            SELECT lr.init_time_utc, fv_wind.valid_time_utc, fv_wind.value AS wind_kt,
                   fv_gust.value AS gust_kt
            FROM forecast_values fv_wind
            JOIN latest_run lr ON lr.run_id = fv_wind.run_id
            LEFT JOIN forecast_values fv_gust
                ON fv_gust.run_id = fv_wind.run_id
               AND fv_gust.valid_time_utc = fv_wind.valid_time_utc
               AND fv_gust.variable = ?
            WHERE fv_wind.variable = ?
              AND fv_wind.valid_time_utc <= now() + (? * INTERVAL '1 hour')
              AND fv_wind.valid_time_utc >= now()
            ORDER BY fv_wind.valid_time_utc
            """,
            [location, model, gust_variable, variable, hours],
        )
        forecast_points = rows_as_dicts(forecast_rows)

        # 2. Historical training pairs: each flag reading matched to the
        # nearest real MIT Pavilion wind AND gust observations within 20
        # minutes. (CBI itself has no wind sensor -- only flag-color
        # readings -- so MIT Pavilion, a few hundred meters away on the
        # same basin, stands in as the real wind ground-truth.)
        training_rows = con.execute(
            """
            SELECT f.flag_color,
                   (SELECT o.value FROM observations o
                    WHERE o.location_id = 'mit_pavilion' AND o.variable = ?
                      AND abs(epoch(o.ts_utc) - epoch(f.ts_utc)) <= 1200
                    ORDER BY abs(epoch(o.ts_utc) - epoch(f.ts_utc))
                    LIMIT 1) AS wind_kt,
                   (SELECT o.value FROM observations o
                    WHERE o.location_id = 'mit_pavilion' AND o.variable = ?
                      AND abs(epoch(o.ts_utc) - epoch(f.ts_utc)) <= 1200
                    ORDER BY abs(epoch(o.ts_utc) - epoch(f.ts_utc))
                    LIMIT 1) AS gust_kt
            FROM flags f
            WHERE f.location_id = ?
            """,
            [variable, gust_variable, location],
        ).fetchall()
        # Effective wind = max(sustained, gust) -- see docstring above.
        # Falls back to whichever of the two is actually available if
        # only one matched within the tolerance window.
        training_rows = [
            (color, max(w for w in (wind, gust) if w is not None))
            for (color, wind, gust) in training_rows
            if wind is not None or gust is not None
        ]

        # 3. Build the joint bucket->color count table, then the Bayesian
        # posterior P(flag=c | wind_bucket=b) with Dirichlet(alpha=1)
        # smoothing per bucket.
        bucket_counts = {}  # bucket -> {color: count}
        marginal_counts = {c: 0 for c in FLAG_COLORS_ALL}
        for color, effective_wind in training_rows:
            if color not in FLAG_COLORS_ALL:
                continue
            b = _wind_bucket(effective_wind)
            bucket_counts.setdefault(b, {c: 0 for c in FLAG_COLORS_ALL})
            bucket_counts[b][color] += 1
            marginal_counts[color] += 1

        alpha = 1.0
        n_colors = len(FLAG_COLORS_ALL)

        def posterior_for_bucket(bucket):
            counts = bucket_counts.get(bucket, {c: 0 for c in FLAG_COLORS_ALL})
            total = sum(counts.values()) + alpha * n_colors
            return {c: round((counts[c] + alpha) / total, 4) for c in FLAG_COLORS_ALL}

        # 4. Attach a probability distribution to every forecast point.
        # CBI only operates 9am-sunset (per user 2026-09-11); outside
        # that window it's ALWAYS closed regardless of wind, and during
        # open hours it should never show any "closed" probability at
        # all from this wind-based estimate (closed-during-open-hours is
        # a rare special-case closure the historical wind data can't
        # meaningfully predict, and letting it leak in just dilutes the
        # red/yellow/green signal that's actually useful for the times
        # you'd consider sailing). So: force certainty outside hours,
        # and zero-out-and-renormalize "closed" during hours.
        predictions = []
        for row in forecast_points:
            wind = row["wind_kt"]
            gust = row.get("gust_kt")
            effective_wind = max((v for v in (wind, gust) if v is not None), default=None)
            valid_dt = row["valid_time_utc"]
            if isinstance(valid_dt, str):
                valid_dt = datetime.fromisoformat(valid_dt)
            if valid_dt.tzinfo is None:
                valid_dt = valid_dt.replace(tzinfo=timezone.utc)

            if not is_cbi_open(valid_dt):
                # Outside 9am-sunset: always closed, no wind-based estimate.
                probs = {c: (1.0 if c == "closed" else 0.0) for c in FLAG_COLORS_ALL}
                bucket = _wind_bucket(effective_wind) if effective_wind is not None else None
                predictions.append({
                    "valid_time_utc": row["valid_time_utc"],
                    "wind_kt": wind,
                    "gust_kt": gust,
                    "wind_bucket": bucket,
                    "flag_probabilities": probs,
                    "most_likely_flag": "closed",
                })
                continue

            bucket = _wind_bucket(effective_wind) if effective_wind is not None else None
            probs = posterior_for_bucket(bucket) if bucket is not None else None
            if probs is not None:
                # During open hours: drop "closed" entirely and
                # renormalize the remaining 3 colors so they sum to 1.
                remaining = {c: probs[c] for c in FLAG_COLORS_ALL if c != "closed"}
                total_remaining = sum(remaining.values())
                if total_remaining > 0:
                    probs = {**{c: round(v / total_remaining, 4) for c, v in remaining.items()}, "closed": 0.0}
                else:
                    # degenerate case (shouldn't happen with Dirichlet smoothing,
                    # but guard anyway): fall back to uniform over the 3 open-hour colors
                    probs = {**{c: round(1 / 3, 4) for c in remaining}, "closed": 0.0}
            predictions.append({
                "valid_time_utc": row["valid_time_utc"],
                "wind_kt": wind,
                "gust_kt": gust,
                "wind_bucket": bucket,
                "flag_probabilities": probs,
                "most_likely_flag": max(probs, key=probs.get) if probs else None,
            })

        return {
            "model": model,
            "training_sample_size": len(training_rows),
            "wind_bucket_width_kt": WIND_BUCKET_WIDTH_KT,
            "marginal_flag_distribution": {
                c: round(marginal_counts[c] / max(1, sum(marginal_counts.values())), 4)
                for c in FLAG_COLORS_ALL
            },
            "predictions": predictions,
        }
    finally:
        con.close()


def api_wind_prediction(params):
    """Bayesian wind-speed-bucket prediction for locations with real wind
    observations (NDBC buoy, MIT Pavilion, KBOS/Logan) -- the wind-only
    counterpart to api_flag_prediction (CBI has no wind sensor, so it
    uses flag color instead; these locations DO have real observed wind,
    so they predict P(actual wind bucket | model's forecast wind bucket)
    instead of a flag color). Deliberately mirrors that endpoint's shape
    (stacked-probability-by-hour) per user direction 2026-09-11 ("show a
    very similar graph, but do wind prediction instead").

        P(actual_bucket=b | forecast_bucket=f, daypart=d, dir_sector=s)
            = P(forecast_bucket=f, daypart=d, dir_sector=s | actual_bucket=b) * P(actual_bucket=b)
              / P(forecast_bucket=f, daypart=d, dir_sector=s)

    ADDED time-of-day (daypart) conditioning 2026-09-24 (user: "how is it
    possible theres a chance of both 20knots and 0 knots, this isn't
    right", re: a genuinely bimodal-looking probability spread on this
    chart). Investigated with a real query at MIT Pavilion/NAM: when NAM
    forecasts 12-16kt, actual outcomes split into two very different
    regimes -- daytime hours (roughly 9am-7pm EDT) frequently come in
    MUCH lower (0-8kt), while nighttime/early-morning hours reliably
    match or exceed the forecast.

    DEDUPED training pairs by real event 2026-09-24 (see below).

    ADDED wind-direction conditioning 2026-09-24 (user: "add in wind
    direction to all the buckets", after investigating one of the real
    misses in this bucket directly: on 2026-09-14 14:00-15:00 UTC, NAM
    forecast 12-14kt and was actually a good REGIONAL forecast -- NDBC
    buoy showed 15.6kt and KBOS/Logan showed 17kt at the same time --
    but MIT Pavilion itself stayed nearly calm (2.4-3.3kt) the whole
    afternoon. MIT Pavilion sits tucked into the Charles River basin,
    sheltered by surrounding buildings/terrain in a way gridded models
    (13-25km+ resolution) can't resolve -- and that sheltering is
    strongly direction-dependent: wind from a sheltered quadrant gets
    blocked, wind from an open quadrant doesn't. Wind direction is a
    real, physically-motivated confound the same way daypart was.

    Fix: added an 8-sector compass-direction bucket (N/NE/E/SE/S/SW/W/NW,
    45 degrees each -- coarser than the 16-sector Wind Rose panel, since
    each extra dimension divides the same finite training data further;
    see the "how many more weeks of data" investigation this was paired
    with) as a third dimension in the joint count table, using the
    model's OWN forecast wind_dir_deg for historical training pairs (so
    the model is conditioned on what IT predicted the direction would
    be, matching how it'll be looked up for future predictions) and each
    future forecast point's own forecast wind_dir_deg the same way.

    Estimated the same way as the flag panel otherwise: build the joint
    (daypart, dir_sector, forecast_bucket, actual_bucket) count table
    directly from this location's own historical forecast-vs-observation
    pairs (same nearest-observation-within-7.5min matching used by
    v_accuracy_by_lead_time, one row per real observed EVENT rather than
    one row per forecast run that touched it -- see the dedup rationale
    below), add Dirichlet(alpha=1) smoothing per (daypart, dir_sector,
    forecast-bucket) row, then look up each future forecast point's own
    (daypart, dir_sector, bucket) in that table to get a probability
    distribution over what the REAL wind is likely to be -- a genuine
    uncertainty band around the model's raw number, not just the number
    itself.
    """
    location = params.get("location")
    model = params.get("model")
    hours = int(params.get("hours", "48"))
    variable = "wind_speed_kt"
    dir_variable = "wind_dir_deg"
    if not location or not model:
        return {"error": "location and model query params are required"}
    con = db()
    try:
        # 1. Forecast wind for the selected model, forward-looking. Also
        # pull each point's LOCAL hour (America/New_York, so it tracks
        # real daylight rather than a fixed UTC offset that would drift
        # across EDT/EST) to classify its daypart, and this SAME model's
        # OWN forecast wind direction at the same valid_time_utc (not
        # observed direction -- for a future point we only have the
        # model's own forecast to condition on, so training must use the
        # same signal for consistency).
        forecast_rows = con.execute(
            """
            WITH latest_run AS (
                SELECT run_id, init_time_utc
                FROM forecast_runs
                WHERE location_id = ? AND model = ?
                QUALIFY row_number() OVER (ORDER BY init_time_utc DESC) = 1
            )
            SELECT lr.init_time_utc, fv.valid_time_utc, fv.value,
                   extract(hour FROM (fv.valid_time_utc AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York')) AS local_hour,
                   fvdir.value AS forecast_dir_deg
            FROM forecast_values fv
            JOIN latest_run lr ON lr.run_id = fv.run_id
            LEFT JOIN forecast_values fvdir
                ON fvdir.run_id = fv.run_id
               AND fvdir.valid_time_utc = fv.valid_time_utc
               AND fvdir.variable = ?
            WHERE fv.variable = ?
              AND fv.valid_time_utc <= now() + (? * INTERVAL '1 hour')
              AND fv.valid_time_utc >= now()
            ORDER BY fv.valid_time_utc
            """,
            [location, model, dir_variable, variable, hours],
        )
        forecast_points = rows_as_dicts(forecast_rows)

        # 2. Historical training pairs: this model's own past forecast
        # values (wind speed AND direction) at this location, matched to
        # the nearest real observation within 7.5 minutes (same
        # tolerance as v_accuracy_by_lead_time), plus the observation's
        # own local hour for daypart classification.
        #
        # DEDUPED BY REAL EVENT 2026-09-24 (user: "it should be more
        # conclusive" / "why is 0 knots even showing up at all in
        # daypart" -- investigated and found a real methodological flaw:
        # a single real hour gets forecast by MANY different runs (a
        # 6h-ahead run, a 12h-ahead run, a 24h-ahead run, etc, all
        # predicting the SAME target valid_time_utc), and an earlier
        # version of this query counted every one of those as an
        # independent training example. Confirmed concretely: 41
        # "0-4kt actual" rows in the (day, 12-16kt-forecast) cell turned
        # out to be only 5 DISTINCT real hours, each re-forecast by ~8
        # different NAM runs -- a handful of real forecast busts on just
        # 2 calendar days was getting pseudo-replicated into 41 "pieces
        # of evidence". Fix: for each real observed event, keep only the
        # forecast from whichever run was CLOSEST to that valid_time
        # (shortest lead time = the run's last, most-informed guess
        # before the event), collapsing what used to be N rows (one per
        # run that touched this hour) into exactly 1 row per real event.
        training_rows = con.execute(
            """
            WITH nearest_obs AS (
                SELECT
                    fv.run_id,
                    fv.valid_time_utc,
                    fr.init_time_utc,
                    fv.value AS forecast_value,
                    fvdir.value AS forecast_dir_deg,
                    o.value AS observed_value,
                    extract(hour FROM (o.ts_utc AT TIME ZONE 'UTC' AT TIME ZONE 'America/New_York')) AS local_hour,
                    row_number() OVER (
                        PARTITION BY fv.run_id, fv.valid_time_utc
                        ORDER BY abs(epoch(fv.valid_time_utc) - epoch(o.ts_utc))
                    ) AS obs_rn
                FROM forecast_values fv
                JOIN forecast_runs fr ON fr.run_id = fv.run_id
                LEFT JOIN forecast_values fvdir
                    ON fvdir.run_id = fv.run_id
                   AND fvdir.valid_time_utc = fv.valid_time_utc
                   AND fvdir.variable = ?
                JOIN observations o
                    ON o.location_id = fr.location_id
                   AND o.variable = fv.variable
                   AND o.ts_utc BETWEEN fv.valid_time_utc - INTERVAL '7.5 minutes'
                                     AND fv.valid_time_utc + INTERVAL '7.5 minutes'
                WHERE fr.location_id = ? AND fr.model = ? AND fv.variable = ?
            ),
            one_per_real_event AS (
                SELECT
                    valid_time_utc, forecast_value, forecast_dir_deg, observed_value, local_hour,
                    row_number() OVER (
                        PARTITION BY valid_time_utc
                        ORDER BY init_time_utc DESC
                    ) AS event_rn
                FROM nearest_obs
                WHERE obs_rn = 1
            )
            SELECT forecast_value, forecast_dir_deg, observed_value, local_hour
            FROM one_per_real_event
            WHERE event_rn = 1
            """,
            [dir_variable, location, model, variable],
        ).fetchall()
        training_rows = [
            (f, d, o, h) for (f, d, o, h) in training_rows
            if f is not None and o is not None and h is not None
        ]

        # 3. Build joint (daypart, dir_sector, forecast_bucket) ->
        # {actual_bucket: count}, then compute the Dirichlet(alpha=1)-
        # smoothed posterior per (daypart, dir_sector, forecast bucket)
        # tuple. A missing/null forecast direction (LEFT JOIN found no
        # matching wind_dir_deg row for that run/valid_time) falls back
        # to a dedicated "unknown" sector rather than being dropped, so
        # a model that's missing direction data for some runs doesn't
        # lose those speed pairs entirely.
        bucket_counts = {}  # (daypart, dir_sector, forecast_bucket) -> {actual_bucket: count}
        all_actual_buckets = set()
        for f_val, dir_deg, o_val, local_hour in training_rows:
            dp = _daypart(local_hour)
            sector = _compass_sector_label(dir_deg) if dir_deg is not None else "unknown"
            fb = _wind_bucket(f_val)
            ob = _wind_bucket(o_val)
            all_actual_buckets.add(ob)
            key = (dp, sector, fb)
            bucket_counts.setdefault(key, {})
            bucket_counts[key][ob] = bucket_counts[key].get(ob, 0) + 1

        all_actual_buckets = sorted(all_actual_buckets, key=lambda b: int(b.split("-")[0]))
        alpha = 1.0
        n_buckets = max(1, len(all_actual_buckets))

        def posterior_for(daypart, sector, fb):
            counts = bucket_counts.get((daypart, sector, fb), {})
            total = sum(counts.values()) + alpha * n_buckets
            return {
                b: round((counts.get(b, 0) + alpha) / total, 4)
                for b in all_actual_buckets
            }

        # 4. Attach a probability distribution (over ACTUAL wind buckets)
        # to every forecast point, using ITS OWN local-hour daypart and
        # ITS OWN forecast direction sector.
        predictions = []
        for row in forecast_points:
            forecast_wind = row["value"]
            forecast_dir = row["forecast_dir_deg"]
            fb = _wind_bucket(forecast_wind) if forecast_wind is not None else None
            dp = _daypart(row["local_hour"]) if row["local_hour"] is not None else None
            sector = _compass_sector_label(forecast_dir) if forecast_dir is not None else "unknown"
            probs = posterior_for(dp, sector, fb) if (fb is not None and dp is not None and all_actual_buckets) else None
            predictions.append({
                "valid_time_utc": row["valid_time_utc"],
                "forecast_wind_kt": forecast_wind,
                "forecast_wind_bucket": fb,
                "daypart": dp,
                "forecast_dir_deg": forecast_dir,
                "dir_sector": sector,
                "actual_wind_probabilities": probs,
                "most_likely_actual_bucket": max(probs, key=probs.get) if probs else None,
            })

        return {
            "location": location,
            "model": model,
            "training_sample_size": len(training_rows),
            "wind_bucket_width_kt": WIND_BUCKET_WIDTH_KT,
            "wind_prediction_directions": WIND_PREDICTION_DIRECTIONS,
            "actual_wind_buckets": all_actual_buckets,
            "predictions": predictions,
        }
    finally:
        con.close()


WIND_ROSE_DIRECTIONS = 16  # standard 16-point compass rose (22.5-degree sectors)
WIND_ROSE_SPEED_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, None)]  # kt


def _compass_sector(deg, n=WIND_ROSE_DIRECTIONS):
    """Maps a direction in degrees to one of n compass sectors, each
    centered on its own heading (sector 0 = N, centered on 0 deg)."""
    sector_width = 360.0 / n
    return int(((deg % 360) + sector_width / 2) // sector_width) % n


def _speed_bin_label(speed_kt):
    for lo, hi in WIND_ROSE_SPEED_BINS:
        if hi is None or speed_kt < hi:
            if speed_kt >= lo:
                return f"{lo}-{hi}kt" if hi is not None else f"{lo}kt+"
    return f"{WIND_ROSE_SPEED_BINS[-1][0]}kt+"


def api_wind_rose(params):
    """Wind rose data: frequency of (direction sector, speed bin) pairs,
    for OBSERVED wind and (optionally) a selected model's FORECAST wind
    at the same location, so the two roses can be compared side by side.
    Only meaningful for locations with real wind_dir_deg/wind_speed_kt
    observations (user 2026-09-11: "wind rose comparison... but only on
    places with observed wind data") -- CBI has neither (no wind sensor),
    so this endpoint is not offered there.
    """
    location = params.get("location")
    model = params.get("model")  # optional; if omitted, observed-only
    hours = int(params.get("hours", "720"))  # default 30 days of history
    if not location:
        return {"error": "location query param is required"}
    con = db()
    try:
        obs_rows = con.execute(
            """
            SELECT o_dir.value AS dir_deg, o_spd.value AS speed_kt
            FROM observations o_dir
            JOIN observations o_spd
                ON o_spd.location_id = o_dir.location_id
               AND o_spd.variable = 'wind_speed_kt'
               AND o_spd.ts_utc BETWEEN o_dir.ts_utc - INTERVAL '5 minutes' AND o_dir.ts_utc + INTERVAL '5 minutes'
            WHERE o_dir.location_id = ? AND o_dir.variable = 'wind_dir_deg'
              AND o_dir.ts_utc >= now() - (? * INTERVAL '1 hour')
            """,
            [location, hours],
        ).fetchall()

        def bucket_rows(rows):
            grid = {}  # (sector, speed_bin) -> count
            total = 0
            for dir_deg, speed_kt in rows:
                if dir_deg is None or speed_kt is None:
                    continue
                sector = _compass_sector(dir_deg)
                sbin = _speed_bin_label(speed_kt)
                key = (sector, sbin)
                grid[key] = grid.get(key, 0) + 1
                total += 1
            return grid, total

        obs_grid, obs_total = bucket_rows(obs_rows)

        forecast_grid, forecast_total = {}, 0
        if model:
            forecast_rows = con.execute(
                """
                WITH latest_run AS (
                    SELECT run_id
                    FROM forecast_runs
                    WHERE location_id = ? AND model = ?
                    QUALIFY row_number() OVER (ORDER BY init_time_utc DESC) = 1
                )
                SELECT fv_dir.value AS dir_deg, fv_spd.value AS speed_kt
                FROM forecast_values fv_dir
                JOIN latest_run lr ON lr.run_id = fv_dir.run_id
                JOIN forecast_values fv_spd
                    ON fv_spd.run_id = fv_dir.run_id
                   AND fv_spd.valid_time_utc = fv_dir.valid_time_utc
                   AND fv_spd.variable = 'wind_speed_kt'
                WHERE fv_dir.variable = 'wind_dir_deg'
                """,
                [location, model],
            ).fetchall()
            forecast_grid, forecast_total = bucket_rows(forecast_rows)

        def to_rows(grid, total):
            out = []
            for sector in range(WIND_ROSE_DIRECTIONS):
                for lo, hi in WIND_ROSE_SPEED_BINS:
                    sbin = f"{lo}-{hi}kt" if hi is not None else f"{lo}kt+"
                    count = grid.get((sector, sbin), 0)
                    out.append({
                        "sector": sector,
                        "sector_deg": sector * (360.0 / WIND_ROSE_DIRECTIONS),
                        "speed_bin": sbin,
                        "count": count,
                        "frequency": round(count / total, 4) if total else 0.0,
                    })
            return out

        return {
            "location": location,
            "model": model,
            "observed": {"total_samples": obs_total, "rows": to_rows(obs_grid, obs_total)},
            "forecast": {"total_samples": forecast_total, "rows": to_rows(forecast_grid, forecast_total)} if model else None,
            "speed_bins": [f"{lo}-{hi}kt" if hi is not None else f"{lo}kt+" for lo, hi in WIND_ROSE_SPEED_BINS],
        }
    finally:
        con.close()


def api_gust_factor(params):
    """Gustiness data for a location: BOTH the absolute gust delta
    (gust_kt - sustained_kt) and the ratio (gust_kt / sustained_kt),
    for a location's wind_speed_kt/wind_gust_kt observation pairs.

    REDESIGNED 2026-09-19 per user: "let's talk about some better
    graphs to represent the gustiness factor because 10x gust is
    different if wind is 10 knots vs 1 knot base." Confirmed
    empirically against MIT Pavilion's own data: the ratio metric is
    systematically misleading at low wind purely because dividing by a
    small number inflates it -- 0-3kt wind showed avg ratio 3.04x (looks
    "extremely gusty") for an absolute gust of only +2.6kt, while 6-10kt
    wind showed a calmer-looking 1.40x ratio for a very similar +2.8kt
    absolute gust. The ratio was telling two nearly-identical real
    gustiness events "extremely gusty" and "not very gusty" respectively,
    purely as an artifact of the denominator.

    Fix: `gust_delta_kt` (gust - sustained, in knots) is now the primary
    metric -- directly answers "how many extra knots could hit me,"
    unaffected by low-wind division blowup. `gust_factor` (the ratio) is
    still always computed and returned (user 2026-09-19: "I like colors
    on the gust factor, do not null it at all at low wind" -- explicitly
    wants the ratio visible/colored across the full wind range, low-wind
    caveats notwithstanding); only genuinely undefined at sustained_kt=0
    (division by zero), which is left as None since there's no ratio to
    report, not because the wind was "too light."

    ONE-DAY FORWARD EXTENSION (added 2026-09-19, user: "I need to be
    able to see a one day forward looking addition to the charts, with
    past data using observed data only, and future data using a model
    of my choice"): if `forecast_model` is given, appends up to 24h of
    FORECAST-derived gustiness points (that model's own wind_speed_kt
    and wind_gust_kt forecasts, same location, joined on valid_time_utc
    from its single latest run) after the observed history. Returned as
    a SEPARATE `forecast` array (not merged into `observed`) so the
    frontend can render them with a visually distinct treatment (dashed
    line / different color) -- this is a genuine response-shape change
    from the old flat-array return, both charts' JS updated accordingly.
    If `forecast_model` is omitted, `forecast` is always `[]` and
    `observed` behaves exactly like the old flat-array response did.
    """
    location = params.get("location")
    hours = int(params.get("hours", "72"))
    forecast_model = params.get("forecast_model")
    forecast_hours = int(params.get("forecast_hours", "24"))
    if not location:
        return {"error": "location query param is required"}
    con = db()
    try:
        rows = con.execute(
            """
            SELECT o_spd.ts_utc, o_spd.value AS sustained_kt, o_gst.value AS gust_kt
            FROM observations o_spd
            JOIN observations o_gst
                ON o_gst.location_id = o_spd.location_id
               AND o_gst.variable = 'wind_gust_kt'
               AND o_gst.ts_utc = o_spd.ts_utc
            WHERE o_spd.location_id = ? AND o_spd.variable = 'wind_speed_kt'
              AND o_spd.ts_utc >= now() - (? * INTERVAL '1 hour')
              AND o_spd.value >= 0
            ORDER BY o_spd.ts_utc
            """,
            [location, hours],
        ).fetchall()
        observed = []
        for ts_utc, sustained_kt, gust_kt in rows:
            if sustained_kt is None or gust_kt is None or sustained_kt < 0:
                continue
            gust_factor = round(gust_kt / sustained_kt, 3) if sustained_kt > 0 else None
            observed.append({
                "ts_utc": ts_utc.isoformat(),
                "sustained_kt": sustained_kt,
                "gust_kt": gust_kt,
                "gust_delta_kt": round(gust_kt - sustained_kt, 1),
                "gust_factor": gust_factor,
            })

        forecast = []
        if forecast_model:
            fc_rows = con.execute(
                """
                WITH latest_run AS (
                    SELECT run_id
                    FROM forecast_runs
                    WHERE location_id = ? AND model = ?
                    QUALIFY row_number() OVER (ORDER BY init_time_utc DESC) = 1
                )
                SELECT fv_spd.valid_time_utc, fv_spd.value AS sustained_kt, fv_gst.value AS gust_kt
                FROM forecast_values fv_spd
                JOIN latest_run lr ON lr.run_id = fv_spd.run_id
                LEFT JOIN forecast_values fv_gst
                    ON fv_gst.run_id = fv_spd.run_id
                   AND fv_gst.valid_time_utc = fv_spd.valid_time_utc
                   AND fv_gst.variable = 'wind_gust_kt'
                WHERE fv_spd.variable = 'wind_speed_kt'
                  AND fv_spd.valid_time_utc >= now()
                  AND fv_spd.valid_time_utc <= now() + (? * INTERVAL '1 hour')
                ORDER BY fv_spd.valid_time_utc
                """,
                [location, forecast_model, forecast_hours],
            ).fetchall()
            for valid_time_utc, sustained_kt, gust_kt in fc_rows:
                if sustained_kt is None or gust_kt is None or sustained_kt < 0:
                    continue
                gust_factor = round(gust_kt / sustained_kt, 3) if sustained_kt > 0 else None
                forecast.append({
                    "ts_utc": valid_time_utc.isoformat(),
                    "sustained_kt": sustained_kt,
                    "gust_kt": gust_kt,
                    "gust_delta_kt": round(gust_kt - sustained_kt, 1),
                    "gust_factor": gust_factor,
                })

        return {"observed": observed, "forecast": forecast, "forecast_model": forecast_model}
    finally:
        con.close()


ROUTES = {
    "/api/locations": api_locations,
    "/api/current-conditions": api_current_conditions,
    "/api/flags-latest": api_flags_latest,
    "/api/flags-history": api_flags_history,
    "/api/flag-prediction": api_flag_prediction,
    "/api/wind-prediction": api_wind_prediction,
    "/api/wind-rose": api_wind_rose,
    "/api/gust-factor": api_gust_factor,
    "/api/accuracy": api_accuracy,
    "/api/accuracy-variables": api_accuracy_variables,
    "/api/forecast": api_forecast,
    "/api/forecast-stability": api_forecast_stability,
    "/api/observations": api_observations,
    "/api/variables-for-location": api_variables_for_location,
}


CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webmanifest": "application/manifest+json; charset=utf-8",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Slightly quieter default logging
        print("%s - %s" % (self.address_string(), format % args))

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path):
        if not os.path.isfile(path):
            self._send_json({"error": "not found"}, 404)
            return
        ext = os.path.splitext(path)[1]
        ctype = CONTENT_TYPES.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            body = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}

        if parsed.path in ROUTES:
            try:
                result = ROUTES[parsed.path](params)
                self._send_json(result)
            except Exception as e:  # surface errors as JSON, not a stack trace to client
                self._send_json({"error": str(e)}, 500)
            return

        # static file serving
        rel = parsed.path
        if rel == "/":
            rel = "/index.html"
        safe_rel = os.path.normpath(rel).lstrip(os.sep)
        full_path = os.path.join(STATIC_DIR, safe_rel)
        # prevent path traversal outside STATIC_DIR
        if not os.path.abspath(full_path).startswith(os.path.abspath(STATIC_DIR)):
            self._send_json({"error": "forbidden"}, 403)
            return
        self._send_file(full_path)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Sailing Weather Dashboard listening on http://localhost:{PORT}")
    print(f"DB path: {os.path.abspath(DB_PATH)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
