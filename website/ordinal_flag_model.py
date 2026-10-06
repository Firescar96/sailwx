"""Bayesian ordinal logistic regression for CBI flag-color prediction.

ADDED 2026-10-06 (user shared a Claude conversation about this exact
problem -- https://claude.ai/share/8a3712e1-68aa-44c6-9799-f999221ebb11
-- then said: "use logistical regression with baysian prediction
inside"). Replaces the discrete 4kt-wide wind-bucket + Dirichlet-
smoothing approach for the RED/YELLOW/GREEN part of the flag
prediction (CLOSED stays handled separately by CBI's operating hours,
same as before -- see api_flag_prediction in app.py).

WHY THIS IS A REAL IMPROVEMENT, NOT JUST A DIFFERENT TASTE:
the old bucket table treats wind speed as a categorical variable with
independent per-bucket distributions, so a bucket with zero real
training examples of some color (confirmed real case: the 20-24kt
bucket had ZERO green examples) falls back to a flat Dirichlet(alpha=1)
floor that gives that color an artificially nonzero probability no
matter how implausible. Flag color vs wind is actually a smooth,
ORDERED relationship (calmer -> more green, windier -> more red) -- so
ordinal logistic regression can correctly extrapolate/interpolate
between training points using a single fitted surface, rather than
treating each 4kt bucket as its own independent experiment.

TWO SEPARATE FEATURES (wind, gust), NOT max(wind, gust) -- CHANGED
2026-10-06 (user: "base wind + gust together predict the flag", after
the earlier version collapsed both into one "effective wind" number).
A single combined number throws away real information: a steady 15kt
with 16kt gusts (barely gusty) and a steady 8kt with 18kt gusts (very
gusty, puffy conditions) would both collapse to the same "effective
wind" of ~15-18kt under max(), even though a dockmaster plausibly
treats them differently. Using (wind, gust) as 2 separate regression
inputs lets the model learn their actual separate contributions (and
fit 2 independent slopes, not force one shared one) instead of
discarding whichever of the two is smaller. Re-validated via 5-fold
cross-validated log loss on real CBI/Harvard Bridge training data
(lower is better):
  naive baseline (ignore wind/gust):      0.993
  old bucket + Dirichlet(alpha=1) table:  0.708
  1-feature ordinal (max(wind,gust)):     0.554
  2-feature ordinal (wind, gust separate):0.574
The 2-feature version is marginally higher CV log loss than the
1-feature version on the CURRENT small sample (53 examples) -- with
this little data, 4 parameters (2 slopes + 2 cutpoints) are harder to
pin down than 3 (1 slope + 2 cutpoints), so there's a real bias-
variance tradeoff here. Kept the 2-feature version anyway per the
user's explicit instruction, since it's the more honest model of the
real physical signal (wind and gust ARE separately meaningful to a
dockmaster) and the CV gap is small; it should also improve as more
training data accumulates and the extra parameter is better supported.

THE MODEL: 2 inputs (sustained wind_kt, gust_kt) predict one of 3
ORDERED outcomes (green < yellow < red) via 2 cutpoints c1 < c2 on a
logistic "danger score" eta = beta_wind*wind + beta_gust*gust:
    P(<= c1) = sigmoid(c1 - eta)   = P(green)
    P(<= c2) = sigmoid(c2 - eta)   = P(green or yellow)
    P(red)    = 1 - P(green or yellow)
    P(yellow) = P(green or yellow) - P(green)
4 parameters total (beta_wind, beta_gust, c1, c2) -- still nowhere near
enough data to need a neural net (the first question in the user's
shared conversation), just a slightly richer simple model.

THE "BAYESIAN PREDICTION INSIDE" PART (per the user's explicit
request): fit via MAP (regularized maximum likelihood -- a weak
N(0, 5^2) Gaussian prior on each parameter) then a LAPLACE
APPROXIMATION to the posterior: the Hessian of the negative log
posterior at the MAP estimate gives an approximate Gaussian posterior
covariance over (beta_wind, beta_gust, c1, c2). Predictions average
over many samples drawn from that posterior Gaussian -- a genuine (if
approximate) Bayesian posterior-predictive average, not just a point
estimate. Chosen over a full MCMC sampler (e.g. PyMC) since PyMC's
compiled backends aren't installed and would be much heavier than
needed for a model this small.

REMOVED 2026-10-06: the Markov persistence blend (a first-order Markov
chain on the flag's own past color history, "P(next hour | current
hour)") that had been layered on top of this module's output via a
product-of-experts combination. User explicitly said: "you don't need
the past 2 hours to predict next 2 hours logic you had added without
asking me" -- removed entirely, without being asked to justify or
soften it first. This module's predict() output is now used directly,
with no further blending step in api_flag_prediction.

ALSO FIXED 2026-10-06 (same conversation, user: "cbi it's closed after
certain hours and those wind levels shouldn't affect calculations"):
confirmed via a direct query that 121 of the training flags were
recorded OUTSIDE CBI's real 9am-sunset operating window (e.g. a stale
"red" or "green" reading at 1am) -- predating a 2026-09-11 ingester fix
that forces off-hours readings to "closed", so these are leftover
stale/pre-fix rows still sitting in the historical data. Training on
them would let a 1am wind reading (which has nothing to do with
whether CBI is actually open) leak into the red/yellow/green curve.
Fixed in api_flag_prediction's query by filtering training rows to
only timestamps where is_cbi_open(ts) is true, so the ordinal model
only ever learns from real open-hours flag decisions.
"""
import numpy as np
from scipy.optimize import minimize

ORDER = ["green", "yellow", "red"]  # must stay in this order -- ordinal position matters
N_POSTERIOR_SAMPLES = 2000
PRIOR_SD = 5.0  # weak regularizing prior on (beta_wind, beta_gust, c1, delta_raw)
MIN_TRAINING_ROWS = 10  # 4 parameters to fit -- need a bit more data than the 1-feature version


def fit(training_rows):
    """training_rows: list of (wind_kt, gust_kt, flag_color) where
    flag_color is one of ORDER (CLOSED rows, and any rows from outside
    CBI's operating hours, must already be filtered out by the caller).
    wind_kt/gust_kt may individually be None (falls back to using
    whichever is present; a row is dropped only if BOTH are missing).
    Returns a dict with everything predict() needs, or None if there's
    not enough data/color variety to fit (caller should fall back to
    the marginal frequency in that case).
    """
    rows = [
        (w, g, c) for w, g, c in training_rows
        if c in ORDER and (w is not None or g is not None)
    ]
    if len(rows) < MIN_TRAINING_ROWS:
        return None
    # If one of wind/gust is missing for a given row, fall back to the
    # other value for both features rather than dropping the row --
    # keeps every real flag reading usable even when only one sensor
    # matched within the time-tolerance window.
    wind = np.array([w if w is not None else g for w, g, _ in rows], dtype=float)
    gust = np.array([g if g is not None else w for w, g, _ in rows], dtype=float)
    y = np.array([ORDER.index(c) for _, _, c in rows], dtype=int)
    if len(set(y.tolist())) < 2:
        # All training examples are the same color -- nothing to fit.
        return None

    w_mean, w_std = wind.mean(), wind.std()
    g_mean, g_std = gust.mean(), gust.std()
    if w_std == 0 or g_std == 0:
        return None
    ws = (wind - w_mean) / w_std
    gs = (gust - g_mean) / g_std

    def neg_log_posterior(params):
        beta_w, beta_g, c1, delta_raw = params
        c2 = c1 + np.exp(delta_raw)  # enforce c2 > c1 structurally
        eta = beta_w * ws + beta_g * gs
        p_le_c1 = 1.0 / (1.0 + np.exp(-(c1 - eta)))
        p_le_c2 = 1.0 / (1.0 + np.exp(-(c2 - eta)))
        probs = np.stack([p_le_c1, p_le_c2 - p_le_c1, 1 - p_le_c2], axis=1)
        probs = np.clip(probs, 1e-9, 1.0)
        log_likelihood = np.sum(np.log(probs[np.arange(len(y)), y]))
        log_prior = -0.5 * (beta_w ** 2 + beta_g ** 2 + c1 ** 2 + delta_raw ** 2) / (PRIOR_SD ** 2)
        return -(log_likelihood + log_prior)

    result = minimize(neg_log_posterior, x0=np.array([1.0, 1.0, 0.0, 0.0]), method="BFGS")
    if not result.success:
        return None

    mean_params = result.x
    cov = result.hess_inv
    try:
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal(mean_params, cov, size=N_POSTERIOR_SAMPLES)
    except np.linalg.LinAlgError:
        # Degenerate/non-PSD covariance (can happen with very sparse or
        # near-perfectly-separated data) -- fall back to a point
        # estimate (zero-width posterior) rather than crashing.
        samples = np.tile(mean_params, (N_POSTERIOR_SAMPLES, 1))

    return {
        "w_mean": w_mean, "w_std": w_std,
        "g_mean": g_mean, "g_std": g_std,
        "posterior_samples": samples,
        "n_training": len(rows),
    }


def predict(fitted_model, wind_kt, gust_kt):
    """Returns {color: probability} for the 3 ORDER colors, averaged
    over the Laplace-approximated posterior samples (genuine Bayesian
    posterior-predictive average, not a single point estimate). Falls
    back to using whichever of wind_kt/gust_kt is available for both
    features if only one is given (mirrors fit()'s handling)."""
    if wind_kt is None and gust_kt is None:
        raise ValueError("predict() requires at least one of wind_kt/gust_kt")
    w = wind_kt if wind_kt is not None else gust_kt
    g = gust_kt if gust_kt is not None else wind_kt

    ws = (w - fitted_model["w_mean"]) / fitted_model["w_std"]
    gs = (g - fitted_model["g_mean"]) / fitted_model["g_std"]
    samples = fitted_model["posterior_samples"]
    beta_w_s, beta_g_s, c1_s, delta_s = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
    c2_s = c1_s + np.exp(delta_s)
    eta = beta_w_s * ws + beta_g_s * gs
    p_le_c1 = 1.0 / (1.0 + np.exp(-(c1_s - eta)))
    p_le_c2 = 1.0 / (1.0 + np.exp(-(c2_s - eta)))
    p_green = p_le_c1
    p_yellow = p_le_c2 - p_le_c1
    p_red = 1 - p_le_c2

    return {
        "green": float(np.clip(p_green.mean(), 0.0, 1.0)),
        "yellow": float(np.clip(p_yellow.mean(), 0.0, 1.0)),
        "red": float(np.clip(p_red.mean(), 0.0, 1.0)),
    }
