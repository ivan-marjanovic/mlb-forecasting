"""Walk-forward forecasting.

Training on an expanding window of the past and predicting only forward is the
whole discipline here. Random k-fold on time-ordered data lets a model learn from
September to predict April, which inflates every metric and is invisible in the
output. There is no way to detect it after the fact from a score alone, which is
why the split is structural rather than a parameter.

Splits fall on date boundaries rather than row counts so a single day's slate is
never divided between train and test. Games are independent given their features
so this is not strictly required, but a boundary that respects the clock is one
fewer thing to argue about.

LightGBM is the intended model. Scikit-learn's histogram gradient booster is used
when LightGBM is unavailable, so the repository runs for someone who clones it
without a full stack. The backend actually used is reported rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _backend():
    try:
        import lightgbm  # noqa: F401
        return "lightgbm"
    except ImportError:
        return "sklearn"


BACKEND = _backend()


def _fit_one(X, y, cfg: Config, model: str = "gbm"):
    """Fit one model on one training window.

    Two families are offered because it is not obvious a priori which is right.
    Gradient boosting is the default reflex for tabular data, but when the
    strongest single correlation is around 0.13 a tree will happily split on
    noise, and the relationship between talent, pitching and win probability is
    close to additive in the log-odds. Fitting both and reporting both is
    cheaper than arguing about it.
    """
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=cfg.logistic_C, max_iter=2000, random_state=cfg.seed),
        ).fit(X, y)

    if BACKEND == "lightgbm":
        import lightgbm as lgb
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": cfg.learning_rate,
            "num_leaves": cfg.num_leaves,
            "max_depth": cfg.max_depth,
            "feature_fraction": cfg.feature_fraction,
            "min_data_in_leaf": cfg.min_data_in_leaf,
            "lambda_l2": cfg.lambda_l2,
            "verbose": -1,
            "seed": cfg.seed,
        }
        return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=cfg.n_rounds)

    from sklearn.ensemble import HistGradientBoostingClassifier
    estimator = HistGradientBoostingClassifier(
        learning_rate=cfg.learning_rate,
        max_leaf_nodes=cfg.num_leaves,
        max_depth=cfg.max_depth,
        max_iter=cfg.n_rounds,
        min_samples_leaf=cfg.min_data_in_leaf,
        l2_regularization=cfg.lambda_l2,
        early_stopping=False,
        random_state=cfg.seed,
    )
    return estimator.fit(X, y)


def _predict_one(fitted, X, model: str = "gbm"):
    if model == "gbm" and BACKEND == "lightgbm":
        return fitted.predict(X)
    return fitted.predict_proba(X)[:, 1]


def walk_forward(panel: pd.DataFrame, features: list[str], cfg: Config,
                 target: str = "home_win", model: str = "gbm") -> np.ndarray:
    """Out-of-sample probabilities, NaN over the initial training window.

    Returns one prediction per row of `panel`, in `panel` order. Rows before the
    first split have no honest prediction and are left as NaN rather than filled,
    so that every downstream metric is computed on genuinely out-of-sample rows.
    """
    panel = panel.reset_index(drop=True)
    out = np.full(len(panel), np.nan)

    dates = panel["date"].to_numpy()
    n = len(panel)
    start = cfg.initial_train

    while start < n:
        # extend the boundary to the end of its calendar day
        boundary_date = dates[start - 1]
        while start < n and dates[start] == boundary_date:
            start += 1
        if start >= n:
            break

        stop = min(start + cfg.step, n)
        stop_date = dates[stop - 1]
        while stop < n and dates[stop] == stop_date:
            stop += 1

        train = panel.iloc[:start]
        test = panel.iloc[start:stop]

        fitted = _fit_one(train[features].to_numpy(), train[target].to_numpy(), cfg, model)
        out[start:stop] = _predict_one(fitted, test[features].to_numpy(), model)

        start = stop

    return out


def permutation_check(panel: pd.DataFrame, features: list[str], cfg: Config,
                      target: str = "home_win", seed: int = 0,
                      model: str = "gbm") -> dict:
    """Rerun the identical pipeline with the target shuffled.

    With the outcome destroyed, no feature can carry information about its own
    row, so out-of-sample Brier must land at the variance of the shuffled target,
    near 0.25. A score meaningfully below that means some feature is contaminated
    by the result it is supposed to predict. This catches leakage that no amount
    of reading the feature code reliably will, and it is the single most useful
    test in the suite.
    """
    rng = np.random.default_rng(seed)
    shuffled = panel.copy()
    shuffled[target] = rng.permutation(shuffled[target].to_numpy())

    preds = walk_forward(shuffled, features, cfg, target=target, model=model)
    mask = ~np.isnan(preds)
    y = shuffled[target].to_numpy()[mask]
    p = preds[mask]

    return {
        "brier": float(np.mean((p - y) ** 2)),
        "expected": float(y.mean() * (1 - y.mean())),
        "n": int(mask.sum()),
        "backend": BACKEND,
    }
