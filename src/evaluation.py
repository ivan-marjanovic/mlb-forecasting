"""Evaluation: benchmarks, significance, calibration.

A Brier improvement of a few thousandths is easy to report and hard to defend.
This module exists so that every number in the README arrives with an error bar
and a baseline underneath it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def brier(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p, eps: float = 1e-12) -> float:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_decomposition(y, p, bins: int = 10):
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    Reliability is miscalibration and should be near zero. Resolution is how far
    the forecasts move away from the base rate, and it is the only term a model
    can improve. Uncertainty is the base rate variance, fixed by the sport.
    Reporting these separately says whether a small edge comes from calibration
    or from genuine discrimination.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    base = y.mean()
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)

    reliability = resolution = 0.0
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        w = m.sum() / len(y)
        reliability += w * (p[m].mean() - y[m].mean()) ** 2
        resolution += w * (y[m].mean() - base) ** 2

    return {
        "reliability": reliability,      # lower is better
        "resolution": resolution,        # higher is better
        "uncertainty": base * (1 - base),
        "brier": brier(y, p),
    }


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------

def brier_diff_test(y, p_baseline, p_model):
    """Paired test on the difference in Brier score.

    Games are independent, so the per-game difference in squared error is an iid
    sample and its mean has an ordinary standard error. This is the
    Diebold-Mariano test specialised to squared-error loss. A positive mean says
    the model beat the baseline; the t statistic says whether to believe it.
    """
    y = np.asarray(y, float)
    d = (np.asarray(p_baseline, float) - y) ** 2 - (np.asarray(p_model, float) - y) ** 2
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    return {
        "improvement": mean,
        "std_error": se,
        "t_stat": mean / se if se > 0 else np.nan,
        "ci95": (mean - 1.96 * se, mean + 1.96 * se),
        "n": n,
    }


def bootstrap_diff(y, p_baseline, p_model, draws: int = 10_000, seed: int = 0):
    """Bootstrap the same difference. Agrees with the analytic SE when the
    per-game differences are well behaved; disagreement is worth investigating."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    d = (np.asarray(p_baseline, float) - y) ** 2 - (np.asarray(p_model, float) - y) ** 2
    n = len(d)
    means = d[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "ci95": (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))),
        "p_worse_than_baseline": float((means <= 0).mean()),
    }


# ---------------------------------------------------------------------------
# The feature ladder
# ---------------------------------------------------------------------------

def feature_ladder(df, target, ladder, fit_predict, baseline_rate=None, verbose=True):
    """Score nested feature sets on one identical out-of-sample split.

    `ladder` is an ordered list of (label, feature_list). Each rung is scored
    against the constant baseline and against the rung below it, so the table
    shows what each block of features is actually worth rather than only the
    final number.
    """
    y = df[target].values
    base = float(y.mean()) if baseline_rate is None else baseline_rate
    p_base = np.full(len(y), base)

    rows = [{
        "model": f"constant base rate ({base:.4f})",
        "n_features": 0,
        "brier": brier(y, p_base),
        "vs_base": 0.0,
        "t_vs_base": np.nan,
        "vs_previous": np.nan,
    }]

    prev = p_base
    for label, feats in ladder:
        if verbose:
            print(f"    fitting: {label} ({len(feats)} features)", flush=True)
        p = fit_predict(df, feats, target)
        mask = ~np.isnan(p)
        stats = brier_diff_test(y[mask], p_base[mask], p[mask])
        step = brier_diff_test(y[mask], prev[mask], p[mask])
        rows.append({
            "model": label,
            "n_features": len(feats),
            "brier": brier(y[mask], p[mask]),
            "vs_base": stats["improvement"],
            "t_vs_base": stats["t_stat"],
            "vs_previous": step["improvement"],
        })
        prev = p

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def reliability_table(y, p, bins: int = 10):
    """Binned observed frequency against mean forecast, with counts.

    The counts column matters: a reliability curve that looks perfect in a bin
    holding nine games is telling you nothing.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(p.min(), p.max(), bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({
            "bin": b,
            "n": int(m.sum()),
            "mean_forecast": float(p[m].mean()),
            "observed": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(out)


def fit_isotonic_forward(df, prob_col, target, split=0.7, order_col="date"):
    """Isotonic recalibration fit on the earlier portion, applied to the later.

    Fitting and evaluating a calibrator on the same rows produces a perfect
    reliability curve that means nothing, so the split is chronological and the
    reported curve comes only from the held-out tail.
    """
    from sklearn.isotonic import IsotonicRegression

    df = df.sort_values(order_col).reset_index(drop=True)
    cut = int(len(df) * split)
    train, test = df.iloc[:cut], df.iloc[cut:].copy()

    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(train[prob_col], train[target])
    test["calibrated"] = iso.predict(test[prob_col])
    return test, iso


# ---------------------------------------------------------------------------

def search_budget_note(n_configs: int, n_feature_sets: int) -> str:
    """State how many configurations were evaluated on the reported split.

    Selecting hyperparameters on the same out-of-sample window that produces the
    headline number biases that number optimistically. Reporting the count does
    not remove the bias, but it lets a reader size it, and almost nobody does it.
    """
    total = n_configs * n_feature_sets
    return (
        f"{n_configs} hyperparameter configurations across {n_feature_sets} feature "
        f"sets ({total} evaluations) were scored on the reported walk-forward split. "
        f"The headline Brier is therefore mildly optimistic; a fully clean estimate "
        f"would require a season held out and touched once."
    )


# ---------------------------------------------------------------------------
# Forecast encompassing
# ---------------------------------------------------------------------------

def encompassing_test(y, p_model, p_benchmark, n_boot: int = 400, seed: int = 0,
                      clip: float = 0.01):
    r"""Does the model carry information the benchmark does not?

    A worse Brier score than the market does not by itself mean a forecast is
    useless. Two forecasts can each be individually inferior and still combine
    into something better, if each sees something the other misses. The test is
    to put both on the log-odds scale and regress the outcome on the pair:

        logit P(y=1) = a + b_benchmark * logit(p_benchmark) + b_model * logit(p_model)

    If the model has no incremental information, b_model is zero. If b_model is
    significantly negative, the model is anti-informative conditional on the
    benchmark: where it disagrees, the benchmark is right in a predictable
    direction, which is a stronger and more useful statement than "it scored
    worse".

    Standard errors are bootstrapped rather than taken from the fitted
    information matrix, because the two regressors are strongly collinear by
    construction and asymptotic errors are optimistic there.

    The `combined_brier` field is fitted in sample and is reported as a
    diagnostic only. It is not evidence that combining helps out of sample, and
    it should never be quoted as a headline.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    pm = np.clip(np.asarray(p_model, dtype=float), clip, 1 - clip)
    pb = np.clip(np.asarray(p_benchmark, dtype=float), clip, 1 - clip)
    logit = lambda q: np.log(q / (1.0 - q))
    X = np.column_stack([logit(pb), logit(pm)])

    def fit(idx):
        return LogisticRegression(C=1e6, max_iter=4000).fit(X[idx], y[idx])

    full = fit(np.arange(len(y)))
    coef = full.coef_[0]

    rng = np.random.default_rng(seed)
    draws = np.array([fit(rng.integers(0, len(y), len(y))).coef_[0] for _ in range(n_boot)])
    se = draws.std(axis=0, ddof=1)

    return {
        "benchmark_coef": float(coef[0]),
        "benchmark_se": float(se[0]),
        "benchmark_t": float(coef[0] / se[0]) if se[0] > 0 else np.nan,
        "model_coef": float(coef[1]),
        "model_se": float(se[1]),
        "model_t": float(coef[1] / se[1]) if se[1] > 0 else np.nan,
        "combined_brier_in_sample": float(brier(y, full.predict_proba(X)[:, 1])),
        "benchmark_brier": float(brier(y, pb)),
        "model_brier": float(brier(y, pm)),
        "n": int(len(y)),
    }
