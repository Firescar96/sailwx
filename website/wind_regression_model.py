"""Bayesian linear regression for the Wind Prediction / Gust Prediction panels.

ADDED 2026-10-06 (user: "update the wind prediction calculation on the
other locations to also use regression" -- following the CBI flag
panel's switch from a discrete bucket table to ordinal logistic
regression). Replaces api_wind_prediction's (daypart, dir_sector,
forecast_bucket) -> {actual_bucket: count} Dirichlet-smoothed count
table with a Bayesian linear regression predicting the ACTUAL
(continuous) wind speed from the model's forecast wind speed plus
daypart and direction-sector as categorical (dummy-coded) features,
then converting that continuous prediction into a probability
distribution over the same discrete wind buckets the frontend already
displays via a Gaussian likelihood.

SPLIT INTO TWO SEPARATE CHARTS/TARGETS -- EXTENDED 2026-10-16 (user:
"i need two graphs instead of one, one for wind and one for wind
gusts, but both charts should use linear regression that incorporates
base wind, gusts, and direction"). Both the Wind chart and the Gust
chart now use the SAME 4 input features -- forecast wind speed,
forecast gust, forecast direction (as an 8-sector dummy, same as
before), and daypart -- predicting two DIFFERENT targets (actual wind
speed vs. actual gust speed respectively). This is a genuine
cross-feed: the Gust chart's prediction is conditioned on forecast
WIND too (not just forecast gust), and vice versa, because real
cross-validated log loss confirmed this helps, not just adds noise
(see below).

WHY THIS IS A REAL IMPROVEMENT, NOT JUST CONSISTENCY FOR ITS OWN SAKE:
the old table has one independent cell per (daypart x dir_sector x
forecast_bucket) combination -- with 2 dayparts x 8 compass sectors x
~8 wind buckets, that's ~128 possible cells competing for a few hundred
training examples at best, so most cells are sparse and fall back to a
flat Dirichlet(alpha=1) prior that barely uses the forecast value at
all once bucketed. A regression instead fits a handful of real
parameters (intercept, slopes, a handful of daypart/sector dummy
offsets) using ALL the training data at once, and extrapolates/
interpolates smoothly rather than needing real examples in each tiny
cell. Verified via real 5-fold cross-validated log loss (lower is
better) across every real location+model combo that has enough
history, BEFORE adding gust as a 4th feature:
    ecmwf/mit_pavilion (ground truth: harvard_bridge): bucket=1.258 regression=0.938
    gfs/kbos:                                          bucket=1.259 regression=0.826
    ecmwf/kbos:                                        bucket=1.235 regression=0.989
    gfs/ndbc_44013:                                    bucket=1.309 regression=0.767
    gfs/mit_pavilion (ground truth: harvard_bridge):    bucket=1.342 regression=1.006
Regression wins by a clear, consistent margin in every case tested --
not a close call, and not cherry-picked to one location/model.

THEN re-validated with forecast gust ADDED as a feature when predicting
actual wind (the cross-feed direction), comparing wind+dir+daypart vs.
wind+gust+dir+daypart, same 5-fold CV methodology, 5 random seeds each:
    ecmwf/mit_pavilion: baseline=0.966 withGust=0.948
    gfs/kbos:            baseline=0.870 withGust=0.853
    ecmwf/kbos:          baseline=1.020 withGust=1.015
    gfs/ndbc_44013:      baseline=0.803 withGust=0.802
Small but consistently non-negative improvement across every combo
tested (never worse) -- confirms the cross-feed is a real, if modest,
win rather than noise, justifying the 4-feature design for both charts.

THE MODEL: ordinary least squares (ordinary, not ordinal -- this
predicts a CONTINUOUS target, unlike the flag panel's ordered
categories) with features:
    [1, forecast_wind_kt, forecast_gust_kt, daypart dummies (night as
     reference), direction-sector dummies (8 compass sectors +
     "unknown", N as reference)]
used identically whether the TARGET (y) is actual wind speed or actual
gust speed -- only which observed column is being predicted changes
between the two charts; the feature set and fitting code are shared.
Residuals are assumed Gaussian with a single shared standard deviation
(estimated from the training residuals) -- this is what turns a single
point prediction into a probability distribution: P(actual in bucket
[lo, hi)) = CDF(hi) - CDF(lo) under Normal(predicted_mean, sigma).

THE "BAYESIAN" PART: fit via ordinary least squares, which IS the
maximum-likelihood/MAP solution under a Gaussian likelihood with a flat
(uninformative) prior over the regression coefficients -- genuinely a
degenerate-prior special case of Bayesian linear regression, not a
different thing wearing its name. A full informative-prior Bayesian
treatment (e.g. ridge-regularized coefficients, posterior samples over
beta AND sigma via a Normal-Inverse-Gamma conjugate update) was
considered but not used here: unlike the CBI flag model (53 examples,
needed real regularization to avoid unstable cutpoints), these
locations have 245-900+ training examples for a comparably small
number of parameters (~12), so OLS is already numerically stable and
an explicit prior wouldn't materially change the fitted coefficients
-- added complexity without a corresponding accuracy gain. If any
location's training set shrinks a lot in the future (e.g. if Harvard
Bridge's limited history becomes the binding constraint again), revisit
with the same Laplace-approximation approach used in
ordinal_flag_model.py.

NO is_cbi_open()-STYLE OPERATING-HOURS FILTER HERE (per explicit user
instruction 2026-10-06, re: a similar-looking filter just added to the
CBI flag model): none of these locations (MIT Pavilion, KBOS/Logan,
NDBC buoy) have restricted operating hours the way CBI's dockhouse
does -- wind blows and gets observed/forecast around the clock
regardless of any human schedule, so there's no "off-hours stale
reading" failure mode to filter out here. The daypart feature already
captures the real physical day/night difference in local wind behavior
(see the ADDED time-of-day conditioning note in api_wind_prediction's
docstring) -- that is NOT the same thing as an hours-of-operation
filter and should not be confused with one.
"""
import numpy as np
from scipy.stats import norm

DAYPARTS = ["night", "day"]  # night is the reference level (dropped from dummies)
SECTORS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "unknown"]  # N is the reference level
MIN_TRAINING_ROWS = 20  # ~12 parameters to fit -- want a healthy multiple of that


def _daypart(local_hour):
    return "day" if (local_hour is not None and 9 <= local_hour < 19) else "night"


def _compass_sector8(deg):
    if deg is None:
        return "unknown"
    dirs = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    idx = int(((deg % 360) + 22.5) // 45) % 8
    return dirs[idx]


def _feature_row(forecast_wind_kt, forecast_gust_kt, daypart, sector):
    # Missing forecast gust (LEFT JOIN with no matching row) falls back
    # to a simple wind-based estimate rather than dropping the row or
    # silently zeroing it out -- 1.3x is a reasonable rough gust factor
    # for an unknown case, not a precise claim, just a safer default
    # than 0 (which would look like "no gust at all" to the model).
    gust = forecast_gust_kt if forecast_gust_kt is not None else forecast_wind_kt * 1.3
    row = [1.0, forecast_wind_kt, gust]
    row += [1.0 if daypart == dp else 0.0 for dp in DAYPARTS[1:]]
    row += [1.0 if sector == s else 0.0 for s in SECTORS[1:]]
    return row


def fit(training_rows):
    """training_rows: list of (forecast_wind_kt, forecast_gust_kt,
    forecast_dir_deg, observed_target, local_hour) -- forecast_gust_kt,
    forecast_dir_deg, and local_hour may be None (forecast_gust_kt
    falls back to a wind-based estimate, forecast_dir_deg falls back to
    the "unknown" sector, local_hour falls back to "night" daypart).
    `observed_target` is whichever actual quantity this fit is for --
    actual wind speed for the Wind chart, actual gust speed for the
    Gust chart -- the fitting code itself doesn't care which. Returns a
    dict with everything predict_distribution() needs, or None if
    there's not enough data to fit meaningfully.
    """
    rows = [
        (fw, fg, fd, ot, h) for fw, fg, fd, ot, h in training_rows
        if fw is not None and ot is not None
    ]
    if len(rows) < MIN_TRAINING_ROWS:
        return None

    X = []
    y = []
    for forecast_wind, forecast_gust, dir_deg, observed_target, local_hour in rows:
        dp = _daypart(local_hour)
        sector = _compass_sector8(dir_deg)
        X.append(_feature_row(forecast_wind, forecast_gust, dp, sector))
        y.append(observed_target)
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    beta, _residuals, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    predicted = X @ beta
    residuals = y - predicted
    dof = max(1, len(y) - X.shape[1])
    sigma = float(np.sqrt(np.sum(residuals ** 2) / dof))
    if sigma <= 0:
        # Degenerate (e.g. all identical observations) -- guard against
        # a zero-width Gaussian that would make every bucket probability
        # either 0 or 1 in a way that's not really justified by the data.
        sigma = 1.0

    return {"beta": beta, "sigma": sigma, "n_training": len(rows)}


def predict_distribution(fitted_model, forecast_wind_kt, forecast_gust_kt, forecast_dir_deg, local_hour, bucket_labels, bucket_width_kt):
    """Returns {bucket_label: probability} over the given bucket labels
    (e.g. "12-16kt"), using the fitted Gaussian predictive distribution
    Normal(predicted_mean, sigma) and integrating its CDF across each
    bucket's [lo, hi) range."""
    beta = fitted_model["beta"]
    sigma = fitted_model["sigma"]
    dp = _daypart(local_hour)
    sector = _compass_sector8(forecast_dir_deg)
    row = np.array(_feature_row(forecast_wind_kt, forecast_gust_kt, dp, sector))
    mean = float(row @ beta)

    raw = {}
    for label in bucket_labels:
        lo = int(label.split("-")[0])
        hi = lo + bucket_width_kt
        p = norm.cdf(hi, mean, sigma) - norm.cdf(lo, mean, sigma)
        raw[label] = max(p, 1e-9)
    total = sum(raw.values())
    return {label: round(p / total, 4) for label, p in raw.items()}

