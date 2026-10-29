#!/usr/bin/env python3
"""NDA -- the "weighted voting" ensemble forecast model.

ADDED 2026-10-17 (user: "for every metric for every location for every
graph, i need a new model, the Nchinda model that does linear
regression across predictions for all models to give its own answer,
weighted voting" -> named "NDA" per follow-up). For every (location,
variable) combo that has enough real history, fits a genuine linear
regression against real ground-truth observations, then writes its own
OWN forecast_runs/forecast_values rows (model='nda') for the CURRENT
run of each of the 6 real models, exactly like any other model would.
This is a deliberate architectural choice: by writing real rows into
the same tables every other model uses, NDA automatically shows up in
EVERY existing chart (Forecast Time Series, Forecast Stability, Model
Accuracy, Wind Rose, Gustiness, Wind/Gust Prediction, Flag Prediction)
with zero special-casing needed in any of those 6+ separate
chart-rendering code paths.

REWRITTEN 2026-10-22 (user, in direct response to the quadratic-through
-zero instability bug -- see git history for that incident):
    "1. go back to linear model
     2. make sure database for every model is storing the updates,
        both the time the model was check AND the time the model was
        predicted for
     3. calculate NDA and store it as you store models, but dont just
        store a single number, store the history of predictions for
        the times in the future
     4. the linear regression should have time as an input because the
        regression might be different for now vs 72 hours in the future
     5. dont include models like HRRR in the future calculation if
        they don't have data, dont take an average"

Addressed as follows:

(1) LINEAR, NOT QUADRATIC -- reverted entirely. The quadratic-through-
zero design (shipped for the Saturday-overshoot complaint) turned out
to be numerically unstable with small training samples: a documented
real case predicted 199kt of wind when the actual observation was
16.7kt, traced to the walk-forward backfill refitting on as few as 199
historical rows early in this project's history (a 12-parameter
quadratic is far more prone to runaway extrapolation on thin data than
a 7-parameter linear fit). Reverting removes that risk entirely; the
Saturday-style small-intercept overshoot this was meant to fix is a
much smaller, more bounded problem than an occasional 199kt prediction.

(2) STORAGE OF BOTH TIMESTAMPS -- already true, confirmed by re-reading
db/schema.sql rather than assuming: forecast_runs.init_time_utc is
"the time the model was checked" (when this run's data was pulled/
generated) and forecast_values.valid_time_utc is "the time the model
was predicted for" (one row per future hour). This was NOT a gap that
needed fixing -- every real model (GFS/ECMWF/etc) and NDA have always
stored both. Documented explicitly here since the user's instruction
implied it might be missing.

(3) FULL PREDICTION HISTORY, NOT A SINGLE NUMBER -- also already true:
NDA writes one forecast_values row per valid_time_utc for every hour
each of the 6 real models currently forecasts (same as any other
model), and every run of this script creates a NEW forecast_runs row
(not an update to a prior one), so historical NDA runs are preserved
exactly like real models' run history -- this is what already lets
Forecast Stability compare NDA's last-N-runs the same way it compares
any real model. Also not a gap; documented for the same reason as (2).

(4) TIME AS A REGRESSION INPUT -- genuinely new. Confirmed empirically
first (not just taking the user's intuition on faith): a regression
trained ONLY on 0-12h lead-time data and then applied unchanged out to
48-72h showed real, measurable degradation (clean CV MAE 1.64kt at
0-12h vs 1.99kt at 24-48h when using the SAME fixed line) -- the
relationship between the 6 models' forecasts and the real outcome
genuinely does shift with lead time, confirming the user's suspicion.
Addressed by adding lead_hours (hours between "now" and the valid_time
being predicted) as an explicit regression feature: NDA = beta0 +
beta_lead*lead_hours + sum(beta_m * model_m for m in whichever models
have real data for that hour). This lets the fitted relationship
genuinely vary continuously with distance into the future, rather than
either (a) one fixed line for every horizon (the old design) or (b)
hard lead-time buckets with arbitrary boundaries and less training
data per bucket.

(5) NO AVERAGING-IN FOR MISSING MODELS, EVER -- genuinely new, and
replaces the single biggest source of the OLD design's error at long
lead times. Previously, when a short-horizon model (HRRR ~48h, NAM
~60h, HRDPS ~42h real coverage -- see ingest_forecasts.py's MODELS
comment) had no data for an hour, predict_ensemble() substituted the
MEAN of whichever models WERE present in that missing model's slot --
this fed the regression a fabricated value in place of real data,
every single time, for every hour past each short-horizon model's own
limit (confirmed live: HRDPS/HRRR/NAM essentially have zero real
coverage past 48h, meaning almost every "72h+" NDA prediction was
quietly substituting 2-3 fabricated inputs). Now: for every distinct
VALID_TIME, the actual set of models with real data at that hour is
computed directly (no hardcoded horizon assumptions -- this adapts
automatically if a model's real coverage ever changes), and a
SEPARATE regression is fit (or reused from cache) for that EXACT set
of available models, trained only on historical rows where that same
set of models was genuinely present. A missing model is never
backfilled with a substitute value of any kind -- it's simply excluded
from both the regression's inputs and its training data for that
specific active-model combination.

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
    - kbos, ndbc_44013, courageous_sailing, coops_8443970: use their
      own real sensors directly.

NEVER INCLUDES ITSELF: per explicit user instruction ("as long as NDA
isn't computed to include itself"), every regression is fit ONLY from
the 6 real NWP models' (gfs, ecmwf, hrrr, icon, nam, hrdps) historical
forecast values -- this script never reads back its own prior 'nda'
rows as a regression input.

DIRECTION (wind_dir_deg) IS CIRCULAR -- per explicit user direction
("regress the sin/cos components separately ... then reconstruct the
angle -- keeps it genuinely 'linear regression', handles wraparound
correctly"): two SEPARATE linear regressions are fit (now also
per-active-model-set + lead_hours, same as every other variable), one
predicting sin(observed_direction), one predicting cos(...), then
NDA's own direction is reconstructed via atan2(predicted_sin,
predicted_cos).

MINIMUM DATA BAR: MIN_TRAINING_ROWS=20 (mirrors
website/wind_regression_model.py's threshold), applied PER
(ACTIVE-MODEL-SET, LEAD-TIME-BUCKET) -- a model-combination/lead-range
with too few historical examples is skipped entirely (no prediction
written for those hours) rather than falling back to a less-specific
fit, which would reintroduce exactly the kind of implicit substitution
point (5) was meant to eliminate.

LEAD-TIME BUCKETS -- ADDED 2026-10-27 (user: "NDA model should extend
out as far as other models go, but over time eg. 10 days out if there
is only one model to do the regression with only use that model...
there should be a lead time parameter of sorts in NDA so that it can
use different weights when predicting today from 10 days from now").
lead_hours was already a continuous regression FEATURE (2026-10-22),
but a real bug meant NDA could never actually reach GFS/ECMWF's full
~384h range: fetch_bucket_training_rows's "freshest run wins" dedup
kept only each (model, valid_time)'s SHORTEST ever-seen lead_hours,
which structurally starves long-lead training examples (confirmed:
GFS alone has 45,000+ real rows with lead_hours in 264-500h sitting in
this database, essentially none of which survived that dedup). Fixed
by fitting a genuinely SEPARATE regression per (active-model-set,
LEAD_BUCKETS range) rather than per active-model-set alone -- each
bucket's fit is trained ONLY on real examples whose lead_hours
genuinely falls in that bucket, so e.g. the 168h+ bucket for a 1-model
(GFS-only) active set is trained on real 168h+ GFS-vs-observed
examples, not short-lead data mislabeled with a long lead_hours value.
This simultaneously implements "extend as far as other models go" (an
hour only gets a prediction if SOME real model + enough matching
training data exists for its exact active-set+bucket, same as always,
but the training data itself is no longer artificially capped) and
"only use that one model at 10 days out" (point 5's no-averaging
principle, now correctly paired with genuine long-lead training data).

CADENCE: run on the SAME cron cadence as the main Forecast Ingester
(see sw_ingest_nda.sh), right after it.
"""
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, "/home/firescar96/.hermes/projects/sailing-weather/db")
from common import get_connection

REAL_MODELS = ["gfs", "ecmwf", "hrrr", "icon", "nam", "hrdps"]
NDA_MODEL_NAME = "nda"
MIN_TRAINING_ROWS = 20  # mirrors website/wind_regression_model.py's threshold
WIND_SPEED_VARS = {"wind_speed_kt", "wind_gust_kt"}  # which variables get the WIND_SANITY_RANGE_KT output clamp -- see predict_linear()'s docstring

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


# Lead-time buckets NDA fits a SEPARATE regression per (active-model-
# set, bucket) combination within -- ADDED 2026-10-27 (user: "NDA model
# should extend out as far as other models go, but over time eg. 10
# days out if there is only one model to do the regression with only
# use that model... there should be a lead time parameter of sorts in
# NDA so that it can use different weights when predicting today from
# 10 days from now").
#
# ROOT CAUSE this fixes: lead_hours was already a continuous regression
# FEATURE (added 2026-10-22), but fetch_bucket_training_rows's old
# "freshest run wins" dedup (row_number() partitioned by model+
# valid_time, keeping only rn=1, i.e. the SHORTEST lead_hours any run
# ever forecast that hour at) meant a GFS-only fit's training set was
# starved down to ~29h max lead -- even though GFS alone has 45,000+
# real (model, valid_time) rows with lead_hours in the 264-500h range
# in this database (confirmed via direct query). The dedup was
# discarding nearly all of that real long-lead data in favor of a
# LATER run's short-lead re-forecast of the exact same hour, which is
# correct behavior for "what's our best current estimate of this
# specific hour" but wrong for "how does the GFS-alone relationship
# behave at long lead times" -- it needs its OWN lead-matched examples,
# not short-lead ones mislabeled with a long lead_hours value.
#
# FIX: bucket both training AND prediction by lead_hours range, and
# scope the "freshest within this range" dedup to each bucket
# separately -- so the 168h+ bucket's fit is trained on real, genuine
# 168h+ examples (of which there are tens of thousands for GFS/ECMWF),
# not whatever happened to survive a global freshest-run filter.
# FIX 2026-10-27 (found via direct testing after first implementing
# LEAD_BUCKETS): an open-ended final bucket (168, None) still clustered
# its training data near 168h, because the "freshest run wins" dedup
# (shortest lead_hours survives) picks rows closest to the bucket's OWN
# lower edge first -- confirmed live: the (gfs,)/(168, None) fit's
# training data range was only 168-179h even though GFS has real rows
# out to ~500h, so predictions at 250h+ were wild extrapolations from a
# fit that had never actually seen anything past 179h. Fixed by closing
# off every bucket (no open-ended tail) so each one's dedup pulls from
# a representative, bounded spread of real training examples
# throughout its own range, matching GFS/ECMWF's real ~384h ceiling.
LEAD_BUCKETS = [(0, 24), (24, 48), (48, 96), (96, 168), (168, 264), (264, 400)]


def lead_bucket_for(lead_hours):
    """Which LEAD_BUCKETS entry a given lead_hours falls into."""
    for lo, hi in LEAD_BUCKETS:
        if lead_hours >= lo and (hi is None or lead_hours < hi):
            return (lo, hi)
    return LEAD_BUCKETS[-1]


def fetch_bucket_training_rows(con, location_id, variable, gt_location, active_models, lead_bucket):
    """Training data for a SPECIFIC set of active models (point 5: a
    regression is only ever fit/applied using models that genuinely
    have data for the hours in question -- never a fabricated
    substitute for a missing one) AND a SPECIFIC lead-time bucket (see
    LEAD_BUCKETS module comment for why this is essential, not optional
    -- without it, long-lead fits are starved of real long-lead data).

    Returns a list of (lead_hours, [value for m in active_models],
    observed_value) tuples. A historical valid_time is included as
    long as EVERY model in active_models has a real value there, AT A
    LEAD TIME WITHIN lead_bucket -- other real models not in
    active_models are ignored entirely for this fit (their presence/
    absence doesn't matter), which is exactly why this naturally reuses
    near-term rows (where all 6 models are present) as training data
    for every active-set, not just the smaller active-sets that only
    ever occur at long lead times.

    Within the bucket, "freshest run" dedup still applies (keeps the
    shortest-lead example WITHIN this bucket, not globally) -- this
    keeps each training row representative of "a real forecast made at
    roughly this distance out" rather than letting multiple re-issued
    runs for the same hour all count as separate rows.

    lead_hours is computed PER ROW as the average, across the active
    models, of (valid_time - that model's own init_time) -- i.e. "how
    far ahead was this forecast made", consistent with how lead_hours
    is computed for a live prediction (see predict-time call sites
    below, which use "now" as the reference init_time since NDA's own
    run represents the current moment)."""
    lo, hi = lead_bucket
    model_list_sql = ", ".join("?" for _ in active_models)
    value_cols = ", ".join(
        f"max(CASE WHEN model = ? THEN value END) AS m{i}" for i in range(len(active_models))
    )
    lead_cols = ", ".join(
        f"max(CASE WHEN model = ? THEN lead_hours END) AS l{i}" for i in range(len(active_models))
    )
    not_null_clause = " AND ".join(f"p.m{i} IS NOT NULL" for i in range(len(active_models)))
    avg_lead_expr = "(" + " + ".join(f"p.l{i}" for i in range(len(active_models))) + f") / {len(active_models)}.0"
    value_select = ", ".join(f"p.m{i}" for i in range(len(active_models)))
    lead_filter = "epoch(fv.valid_time_utc - fr.init_time_utc) / 3600.0 >= ?"
    lead_params = [lo]
    if hi is not None:
        lead_filter += " AND epoch(fv.valid_time_utc - fr.init_time_utc) / 3600.0 < ?"
        lead_params.append(hi)

    query = f"""
        WITH per_model AS (
            SELECT fv.valid_time_utc, fr.model, fv.value,
                   epoch(fv.valid_time_utc - fr.init_time_utc) / 3600.0 AS lead_hours,
                   row_number() OVER (
                       PARTITION BY fr.model, fv.valid_time_utc
                       ORDER BY epoch(fv.valid_time_utc - fr.init_time_utc)
                   ) AS rn
            FROM forecast_values fv
            JOIN forecast_runs fr ON fr.run_id = fv.run_id
            WHERE fr.location_id = ? AND fv.variable = ?
              AND fr.model IN ({model_list_sql})
              AND fv.valid_time_utc >= fr.init_time_utc
              AND {lead_filter}
        ),
        pivoted AS (
            SELECT valid_time_utc, {value_cols}, {lead_cols}
            FROM per_model
            WHERE rn = 1
            GROUP BY valid_time_utc
        )
        SELECT {avg_lead_expr} AS lead_hours, {value_select}, o.value AS observed_value
        FROM pivoted p
        JOIN observations o
            ON o.location_id = ? AND o.variable = ?
           AND o.ts_utc BETWEEN p.valid_time_utc - INTERVAL '7.5 minutes'
                             AND p.valid_time_utc + INTERVAL '7.5 minutes'
        WHERE {not_null_clause}
    """
    params = (
        [location_id, variable]
        + list(active_models)
        + lead_params
        + list(active_models)
        + list(active_models)
        + [gt_location, variable]
    )
    rows = con.execute(query, params).fetchall()
    return rows


def fit_linear(rows, n_active_models):
    """Plain OLS with intercept: y ~= beta0 + beta_lead*lead_hours +
    sum(beta_m * model_m). `rows` is (lead_hours, model_val_1, ...,
    model_val_k, observed_value) tuples -- the lead_hours column comes
    FIRST so it's always feature index 0 regardless of how many models
    are active, keeping this one fit function usable for every
    active-model-set size (1 through 6). Returns None if there's not
    enough data (point 5's "don't fall back to a less-specific fit"
    principle applies here too -- too little data for one specific
    active-set means no prediction for those hours, not a guess).

    MINIMUM ROWS SCALES WITH PARAMETER COUNT -- ADDED 2026-10-24 after
    finding a real problem during backfill testing: a flat
    MIN_TRAINING_ROWS=20 for every active-set size let 1-model fits
    (2 parameters: intercept + lead_hours + 1 model = effectively a
    plane in 3D) run with barely more rows than parameters, which is
    exactly the small-sample-instability regime that caused the
    quadratic model's 199kt explosion earlier in this project's
    history. Confirmed live: courageous_sailing's backfill produced
    ~7% of its wind_speed_kt predictions as physically impossible
    (negative or 100+kt), ALL traced to n_models=1 fits with minimal
    walk-forward training data early in that location's history.
    Scaling the floor by (2 + n_active_models) -- the real parameter
    count (intercept + lead_hours + one coefficient per active model)
    -- means a 1-model fit now needs more real support before being
    trusted, while a full 6-model fit (8 parameters) isn't penalized
    beyond the original threshold in proportion."""
    min_rows = max(MIN_TRAINING_ROWS, (2 + n_active_models) * 10)
    if len(rows) < min_rows:
        return None
    arr = np.array(rows, dtype=float)
    X = arr[:, :-1]  # lead_hours + all active model values
    y = arr[:, -1]
    X1 = np.hstack([np.ones((len(X), 1)), X])
    beta, _residuals, _rank, _sv = np.linalg.lstsq(X1, y, rcond=None)
    # ADDED 2026-10-24 (user: "shouldn't be 80 knot predictions...unless
    # which models said the same?") -- found that even after fixing the
    # negative-lead_hours query bug (see fetch_bucket_training_rows's
    # new "fv.valid_time_utc >= fr.init_time_utc" filter), a fit trained
    # ONLY on short lead times (e.g. a 1-model active-set that's rarely
    # the sole survivor except at very long lead times, so its training
    # rows end up clustered at 0-5h) was still being asked to predict at
    # 300+ hours -- wild extrapolation FAR outside the fit's actual
    # training range, producing nonsense (verified: -9kt for a case
    # whose only real input, GFS, said an ordinary 10.5kt). The WIDE
    # sanity clamp from the same date was hiding this by capping the
    # output number, but never addressed why the regression was wrong
    # in the first place. Real fix (per user's explicit choice): track
    # the training data's own lead_hours range here and have
    # predict_linear() refuse to extrapolate meaningfully beyond it.
    max_train_lead = float(X[:, 0].max())
    return beta, max_train_lead


# Floor-only sanity guard for wind variables -- REMOVED the upper
# bound 2026-10-24 (user: "remove your wind clamps at the top end").
# The real defense against nonsense high values is predict_linear()'s
# lead_hours-extrapolation-range check below (MAX_EXTRAPOLATION_FACTOR),
# not an arbitrary output ceiling -- a ceiling can never distinguish a
# genuinely high but real wind/gust value from a regression artifact,
# it just chops off anything above the line regardless of cause. Only
# the 0 floor remains, since negative wind speed/gust is never
# physically meaningful under any circumstance.
WIND_SANITY_RANGE_KT = (0.0, None)

# How far beyond a fit's own training lead_hours range it's allowed to
# extrapolate before predict_linear() refuses outright (returns None)
# instead of guessing -- ADDED 2026-10-24, see fit_linear()'s docstring
# for the investigation that motivated this (a 1-model fit trained on
# 0-5h data was extrapolating to 300+h and producing nonsense). A
# fixed multiplier rather than a fixed hour count so a fit with a wider
# natural training range (e.g. a 2-model set that's commonly active out
# to 72h) isn't penalized the same as a 1-model set that's only ever
# seen 0-5h.
MAX_EXTRAPOLATION_FACTOR = 2.0


def predict_linear(beta_and_range, lead_hours, model_values, clamp_range=None):
    """model_values: list of floats, same order/length as the
    active_models the beta was fit against. No None-filling/averaging
    here at all (point 5) -- by construction, the caller only ever
    passes the models that genuinely have real data for this hour.
    clamp_range, if given, clips the output to (lo, hi) -- used for
    wind_speed_kt/wind_gust_kt only, see WIND_SANITY_RANGE_KT (now a
    last-resort safety net, not the primary defense -- see
    MAX_EXTRAPOLATION_FACTOR for the real fix).

    beta_and_range is fit_linear()'s (beta, max_train_lead) return
    value -- refuses to predict (returns None) when lead_hours is more
    than MAX_EXTRAPOLATION_FACTOR x beyond what this specific fit was
    ever trained on, rather than silently extrapolating into nonsense
    and relying on the output clamp to hide it."""
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


def fetch_bucket_direction_rows(con, location_id, gt_location, active_models, lead_bucket):
    """Same shape as fetch_bucket_training_rows but for wind_dir_deg --
    returns raw (not yet sin/cos-transformed) rows so the caller can
    build the sin and cos training sets separately. See module
    docstring's "DIRECTION IS CIRCULAR" section."""
    return fetch_bucket_training_rows(con, location_id, "wind_dir_deg", gt_location, active_models, lead_bucket)


def fit_direction(rows):
    """Fits 2 separate linear regressions (sin and cos components) from
    the same raw-degree rows fetch_bucket_direction_rows returns.
    Returns (beta_sin, beta_cos), either/both None if too little data."""
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
    """model_degrees: list of raw direction forecasts (degrees), same
    order/length as the active_models the betas were fit against --
    no None-filling, mirrors predict_linear()."""
    if beta_sin is None or beta_cos is None:
        return None
    sins = [np.sin(np.radians(d)) for d in model_degrees]
    coss = [np.cos(np.radians(d)) for d in model_degrees]
    pred_sin = float(np.array([1.0, lead_hours] + sins) @ beta_sin)
    pred_cos = float(np.array([1.0, lead_hours] + coss) @ beta_cos)
    return float(np.degrees(np.arctan2(pred_sin, pred_cos)) % 360)


def get_latest_run_forecast(con, location_id, model, variable):
    """Returns {valid_time_utc: value} for the latest run of (location,
    model, variable).

    FIXED 2026-10-24 (same bug as backfill_nda.py's prefetch_all_runs
    and fetch_full_rows_any_lead -- user: "shouldn't be 80 knot
    predictions...unless which models said the same?"): without the
    valid_time_utc >= init_time_utc filter, a model's bundled pre-init
    analysis hours were being included as if they were real forward-
    looking forecast values, corrupting which active-model-set a given
    hour was grouped into live."""
    rows = con.execute(
        """
        WITH latest_run AS (
            SELECT run_id, init_time_utc FROM forecast_runs
            WHERE location_id = ? AND model = ?
            ORDER BY init_time_utc DESC LIMIT 1
        )
        SELECT fv.valid_time_utc, fv.value
        FROM forecast_values fv JOIN latest_run lr ON lr.run_id = fv.run_id
        WHERE fv.variable = ? AND fv.valid_time_utc >= lr.init_time_utc
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
            # Query-then-branch rather than ON CONFLICT ... DO UPDATE --
            # see backfill_nda.py's identical fix for the full
            # rationale (a real DuckDB 1.5.5 limitation: UPSERT-ing a
            # row that already has forecast_values children raises a
            # spurious ConstraintException). `now` differs every run
            # down to the microsecond so this conflict essentially
            # never fires live in practice, but the fix is applied here
            # too for consistency/future-proofing with backfill_nda.py.
            existing_run = con.execute(
                "SELECT run_id FROM forecast_runs WHERE model = ? AND location_id = ? AND init_time_utc = ?",
                [NDA_MODEL_NAME, location_id, now],
            ).fetchone()
            if existing_run:
                run_id = existing_run[0]
                con.execute("UPDATE forecast_runs SET retrieved_at = ? WHERE run_id = ?", [now, run_id])
            else:
                run_id = con.execute(
                    """
                    INSERT INTO forecast_runs (model, location_id, init_time_utc, retrieved_at)
                    VALUES (?, ?, ?, ?)
                    RETURNING run_id
                    """,
                    [NDA_MODEL_NAME, location_id, now, now],
                ).fetchone()[0]

            for variable in VARIABLES:
                gt_loc = ground_truth_location(location_id, variable)

                per_model_forecasts = {
                    m: get_latest_run_forecast(con, location_id, m, variable) for m in REAL_MODELS
                }
                all_valid_times = sorted(set().union(*[set(d.keys()) for d in per_model_forecasts.values()]))
                if not all_valid_times:
                    continue

                # Group valid_times by their EXACT active-model-set
                # (point 5) AND lead-time bucket (see LEAD_BUCKETS
                # module comment -- "different weights when predicting
                # today from 10 days from now") -- a given hour's
                # active set is whichever real models genuinely have a
                # value there, nothing more; its bucket is which
                # LEAD_BUCKETS range its own lead_hours (relative to
                # THIS run's "now") falls into. Most hours within a
                # model's real coverage share the same (active-set,
                # bucket) pair, so this naturally collapses to a
                # manageable number of distinct fits per location/
                # variable, not one fit per hour.
                groups = {}
                for vt in all_valid_times:
                    active = tuple(m for m in REAL_MODELS if per_model_forecasts[m].get(vt) is not None)
                    if not active:
                        continue
                    lead_hours_for_vt = (vt - now.replace(tzinfo=None)).total_seconds() / 3600.0
                    bucket = lead_bucket_for(lead_hours_for_vt)
                    groups.setdefault((active, bucket), []).append(vt)

                rows_written = 0
                skipped_insufficient_data = 0
                fit_cache = {}  # (active_models, bucket) tuple -> beta (or (beta_sin, beta_cos) for direction)
                for (active_models, bucket), vts in groups.items():
                    if (active_models, bucket) not in fit_cache:
                        if variable == "wind_dir_deg":
                            train_rows = fetch_bucket_direction_rows(con, location_id, gt_loc, active_models, bucket)
                            fit_cache[(active_models, bucket)] = fit_direction(train_rows)
                        else:
                            train_rows = fetch_bucket_training_rows(con, location_id, variable, gt_loc, active_models, bucket)
                            fit_cache[(active_models, bucket)] = fit_linear(train_rows, len(active_models))

                    fitted = fit_cache[(active_models, bucket)]
                    is_missing = (fitted is None) if variable != "wind_dir_deg" else (fitted[0] is None)
                    if is_missing:
                        skipped_insufficient_data += len(vts)
                        continue

                    for vt in vts:
                        # DuckDB TIMESTAMP columns come back as naive
                        # datetimes (no tz info) -- now is tz-aware, so
                        # strip its tzinfo for a consistent subtraction
                        # (every timestamp in this DB is already
                        # implicitly UTC, matching the project's
                        # existing convention elsewhere).
                        lead_hours = (vt - now.replace(tzinfo=None)).total_seconds() / 3600.0
                        model_values = [per_model_forecasts[m][vt] for m in active_models]
                        if variable == "wind_dir_deg":
                            beta_sin, beta_cos = fitted
                            pred = predict_direction(beta_sin, beta_cos, lead_hours, model_values)
                        else:
                            clamp = WIND_SANITY_RANGE_KT if variable in WIND_SPEED_VARS else None
                            pred = predict_linear(fitted, lead_hours, model_values, clamp_range=clamp)
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
                note = f" ({skipped_insufficient_data} hours skipped, insufficient training data for their active-model-set)" if skipped_insufficient_data else ""
                print(f"NDA {location_id}/{variable}: fit on ground truth {gt_loc}, wrote {rows_written} points across {len(fit_cache)} active-model-set(s){note}")
    finally:
        con.close()
    print(f"NDA ensemble: {total_rows_written} total forecast points written across all location/variable combos")


if __name__ == "__main__":
    sys.exit(main())
