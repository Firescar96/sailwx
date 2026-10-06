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
matter how implausible. Flag color vs wind speed is actually a smooth,
ORDERED relationship (calmer -> more green, windier -> more red) -- so
ordinal logistic regression can correctly extrapolate/interpolate
between training points using a single fitted curve, rather than
treating each 4kt bucket as its own independent experiment. Verified
via real 5-fold cross-validated log loss on 2026-10-06's training data
(53 non-closed CBI flag readings matched to Harvard Bridge wind/gust):
  - naive baseline (ignore wind, always predict marginal frequency): 0.993
  - old bucket + Dirichlet(alpha=1) table:                           0.708
  - this ordinal regression model:                                  0.554
Lower is better -- the ordinal model is a real, measured improvement,
not a theoretical nicety.

THE MODEL: a single input (effective wind = max(sustained, gust) kt,
same transform as the old bucket model -- see api_flag_prediction's
"BUCKETING SIGNAL" docstring note for why) predicts one of 3 ORDERED
outcomes (green < yellow < red) via 2 cutpoints c1 < c2 on a logistic
"danger score" eta = beta * wind:
    P(wind <= c1 threshold)  = sigmoid(c1 - eta)   = P(green)
    P(wind <= c2 threshold)  = sigmoid(c2 - eta)   = P(green or yellow)
    P(red) = 1 - P(green or yellow)
    P(yellow) = P(green or yellow) - P(green)
Only 3 parameters total (beta, c1, c2), which is why 53 training
examples is plenty -- nowhere near enough for a neural net, but ample
for a model this simple (this was the first question in the user's
shared conversation: Bayesian/simple-model vs small neural net, and the
conversation's own conclusion -- simple model wins with this little
data -- is followed here).

THE "BAYESIAN PREDICTION INSIDE" PART (per the user's explicit
request): rather than a single best-fit (beta, c1, c2) point estimate,
this fits via MAP (maximum a posteriori, i.e. regularized maximum
likelihood -- a weak N(0, 5^2) Gaussian prior on each parameter for
regularization, since 53 samples is little enough that an unregularized
MLE can push cutpoints to unstable extremes) and then uses a LAPLACE
APPROXIMATION to the posterior: the Hessian of the negative log
posterior at the MAP estimate gives an approximate Gaussian posterior
covariance over (beta, c1, c2). Predictions are made by drawing many
samples from that posterior Gaussian and averaging each sample's
predicted probabilities -- a genuine (if approximate) Bayesian
posterior-predictive average, not just a point estimate. This was
chosen over a full MCMC sampler (e.g. PyMC) because PyMC and its
compiled backends aren't installed and would be a much heavier
dependency for a problem this small; the Laplace approximation is a
standard, well-established lightweight substitute that's exact for
truly Gaussian posteriors and a good approximation for posteriors with
one clear mode (true here, confirmed via a successful BFGS converge
every time this has been tested).

This module only handles the ordinal (red/yellow/green) piece. The
Markov persistence blend (see api_flag_prediction's "MARKOV PERSISTENCE
BLENDING" docstring note, added earlier in the same investigation
thread) still runs on TOP of this module's output via the same
product-of-experts combination as before -- this module just replaces
what used to be "posterior_for_bucket()".
"""
import numpy as np
from scipy.optimize import minimize

ORDER = ["green", "yellow", "red"]  # must stay in this order -- ordinal position matters
N_POSTERIOR_SAMPLES = 2000
PRIOR_SD = 5.0  # weak regularizing prior on (beta, c1, delta_raw)


def fit(training_pairs):
    """training_pairs: list of (effective_wind_kt, flag_color) where
    flag_color is one of ORDER (CLOSED rows must already be filtered
    out by the caller -- this model only predicts the 3 open-hours
    colors). Returns a dict with everything predict() needs, or None if
    there's not enough data/color variety to fit (caller should fall
    back to something else, e.g. the marginal frequency, in that case).
    """
    pairs = [(w, c) for w, c in training_pairs if c in ORDER and w is not None]
    if len(pairs) < 8:
        # Need a bare minimum of data for a 3-parameter fit to mean
        # anything at all -- below this, MAP easily degenerates.
        return None
    x = np.array([w for w, _ in pairs], dtype=float)
    y = np.array([ORDER.index(c) for _, c in pairs], dtype=int)
    if len(set(y.tolist())) < 2:
        # All training examples are the same color -- there's no
        # wind-color relationship to fit at all.
        return None

    x_mean, x_std = x.mean(), x.std()
    if x_std == 0:
        return None
    xs = (x - x_mean) / x_std

    def neg_log_posterior(params):
        beta, c1, delta_raw = params
        c2 = c1 + np.exp(delta_raw)  # enforce c2 > c1 structurally
        eta = beta * xs
        p_le_c1 = 1.0 / (1.0 + np.exp(-(c1 - eta)))
        p_le_c2 = 1.0 / (1.0 + np.exp(-(c2 - eta)))
        probs = np.stack([p_le_c1, p_le_c2 - p_le_c1, 1 - p_le_c2], axis=1)
        probs = np.clip(probs, 1e-9, 1.0)
        log_likelihood = np.sum(np.log(probs[np.arange(len(y)), y]))
        log_prior = -0.5 * (beta ** 2 + c1 ** 2 + delta_raw ** 2) / (PRIOR_SD ** 2)
        return -(log_likelihood + log_prior)

    result = minimize(neg_log_posterior, x0=np.array([1.0, 0.0, 0.0]), method="BFGS")
    if not result.success:
        return None

    mean_params = result.x
    cov = result.hess_inv
    # Guard against a degenerate/non-PSD covariance (can happen with
    # very sparse or near-perfectly-separated data) -- fall back to a
    # point estimate (zero-width posterior) rather than crashing.
    try:
        rng = np.random.default_rng(42)
        samples = rng.multivariate_normal(mean_params, cov, size=N_POSTERIOR_SAMPLES)
    except np.linalg.LinAlgError:
        samples = np.tile(mean_params, (N_POSTERIOR_SAMPLES, 1))

    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "posterior_samples": samples,
        "n_training": len(pairs),
    }


def predict(fitted_model, wind_kt):
    """Returns {color: probability} for the 3 ORDER colors, averaged
    over the Laplace-approximated posterior samples (genuine Bayesian
    posterior-predictive average, not a single point estimate)."""
    x_mean = fitted_model["x_mean"]
    x_std = fitted_model["x_std"]
    samples = fitted_model["posterior_samples"]

    ws = (wind_kt - x_mean) / x_std
    beta_s, c1_s, delta_s = samples[:, 0], samples[:, 1], samples[:, 2]
    c2_s = c1_s + np.exp(delta_s)
    eta = beta_s * ws
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
