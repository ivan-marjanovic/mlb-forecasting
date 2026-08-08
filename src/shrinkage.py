"""Empirical Bayes shrinkage.

A team 15-5 after twenty games is not a .750 talent, and a starter with a 1.20
ERA over three outings is mostly luck. Both are the same problem: an estimate
from a small sample should be pulled toward the population it was drawn from, by
an amount the data itself determines.

Two likelihoods appear here because the two quantities are different kinds of
thing. Wins are bounded counts out of a known number of trials, so the natural
conjugate pair is Beta-Binomial. Runs allowed is an unbounded count over an
exposure, so it is Gamma-Poisson. Using a Beta prior on runs, or a Gamma prior on
a win rate, would give shrinkage that is directionally sensible and quantitatively
wrong.

Both posteriors are closed form, which matters more than elegance: the MCMC
version of the win-rate model takes about a minute per refit, so a season of
daily updates is an hour. The closed form is two milliseconds, which makes daily
refits free and removes the reason the original version updated only every two
weeks.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln, gammaln


# ---------------------------------------------------------------------------
# Beta-Binomial: win rates
# ---------------------------------------------------------------------------

def fit_kappa(wins, games, phi: float = 0.5, fallback: float = 60.0,
              cap: float = 400.0, floor: float = 20.0, min_units: int = 5,
              min_games_per_unit: int = 20) -> float:
    r"""Fit the Beta-Binomial concentration by marginal likelihood.

    The prior is Beta(phi*kappa, (1-phi)*kappa), so kappa is interpretable as a
    number of prior games: a fitted kappa of 60 means each team's record is
    shrunk as though sixty .500 games sat behind it.

    phi is fixed at 0.5 for team win rates rather than estimated, because the
    league-wide rate is exactly .500 by construction. Every game produces one
    winner and one loser, so there is nothing to learn there and estimating it
    only adds variance.

    The cap and floor are not cosmetic, and the guard is on games *per team*
    rather than league-wide for a reason that cost a real bug. Two days into a
    season every team is 1-0 or 0-1. League-wide that is sixty games, which looks
    like plenty; per team it is two. The all-or-nothing spread reads to the
    marginal likelihood as enormous true variance, so the maximiser drives kappa
    to its lower bound and applies almost no shrinkage at the exact moment the
    records are least informative. The per-unit guard holds the fallback until
    each unit has real exposure, and the floor stops the same failure recurring
    mid-season.

    The cap handles the opposite pathology: when observed spread is no wider than
    binomial noise alone would produce, the likelihood is flat toward complete
    pooling and the maximiser runs off toward infinity.
    """
    w = np.asarray(wins, dtype=float)
    n = np.asarray(games, dtype=float)
    mask = n > 0
    w, n = w[mask], n[mask]

    if len(w) < min_units or np.median(n) < min_games_per_unit:
        return float(fallback)

    def negative_log_marginal(log_kappa: float) -> float:
        kappa = np.exp(log_kappa)
        alpha, beta = phi * kappa, (1.0 - phi) * kappa
        return -np.sum(betaln(w + alpha, n - w + beta) - betaln(alpha, beta))

    result = minimize_scalar(negative_log_marginal, bounds=(0.0, 8.0), method="bounded")
    return float(np.clip(np.exp(result.x), floor, cap))


def implied_talent_sd(kappa: float, phi: float = 0.5) -> float:
    """Prior standard deviation of true talent implied by a fitted kappa.

    sd = sqrt(phi*(1-phi)/(kappa+1)). Worth reporting alongside kappa because it
    is checkable against outside knowledge: true MLB team win rates spread with a
    standard deviation near .045 to .050, which corresponds to kappa near 100. A
    fitted kappa of 5 or of 400 is not a number, it is a symptom.
    """
    return float(np.sqrt(phi * (1.0 - phi) / (kappa + 1.0)))


def shrink_binomial(wins, games, kappa: float, phi: float = 0.5) -> np.ndarray:
    """Posterior mean of a Beta-Binomial: (w + phi*kappa) / (n + kappa)."""
    w = np.asarray(wins, dtype=float)
    n = np.asarray(games, dtype=float)
    return (w + phi * kappa) / (n + kappa)


# ---------------------------------------------------------------------------
# Gamma-Poisson: runs allowed
# ---------------------------------------------------------------------------

def fit_gamma_poisson(total_runs, starts, prior_mean: float = 4.5,
                      fallback_b: float = 5.0, min_units: int = 5,
                      min_starts: int = 50) -> tuple[float, float]:
    r"""Fit a Gamma(a, b) prior on runs allowed per start by marginal likelihood.

    With runs ~ Poisson(lambda * starts) and lambda ~ Gamma(a, b), the marginal
    is negative binomial and the posterior mean is (runs + a) / (starts + b).
    Here b plays the same role kappa does above: an effective number of prior
    starts. Returns (a, b) with a = prior_mean * b, so the prior is centred on
    the league run environment and only its strength is estimated.
    """
    y = np.asarray(total_runs, dtype=float)
    n = np.asarray(starts, dtype=float)
    mask = n > 0
    y, n = y[mask], n[mask]

    if len(y) < min_units or n.sum() < min_starts:
        return prior_mean * fallback_b, fallback_b

    def negative_log_marginal(log_b: float) -> float:
        b = np.exp(log_b)
        a = prior_mean * b
        return -np.sum(
            gammaln(y + a) - gammaln(a) - gammaln(y + 1.0)
            + a * np.log(b / (b + n))
            + y * np.log(n / (b + n))
        )

    result = minimize(negative_log_marginal, x0=np.array([np.log(5.0)]),
                      bounds=[(np.log(0.05), np.log(200.0))], method="L-BFGS-B")
    b = float(np.exp(result.x[0]))
    return prior_mean * b, b


def shrink_poisson(total_runs, starts, a: float, b: float) -> np.ndarray:
    """Posterior mean of a Gamma-Poisson: (runs + a) / (starts + b)."""
    y = np.asarray(total_runs, dtype=float)
    n = np.asarray(starts, dtype=float)
    return (y + a) / (n + b)


# ---------------------------------------------------------------------------
# Validation against MCMC
# ---------------------------------------------------------------------------

def posterior_means_mcmc(wins, games, draws: int = 2000, tune: int = 2000,
                         chains: int = 4, target_accept: float = 0.95, seed: int = 0):
    """Same model, sampled rather than solved, as a check on the closed form.

    Returns None when PyMC is not installed so the rest of the pipeline still
    runs. This exists to be run once and compared, not to sit in the hot path.
    """
    try:
        import pymc as pm
    except ImportError:
        return None

    wins = np.asarray(wins, dtype=int)
    games = np.asarray(games, dtype=int)

    with pm.Model():
        phi = 0.5
        kappa = pm.Deterministic("kappa", pm.math.exp(pm.Exponential("log_kappa", lam=0.2)))
        theta = pm.Beta("theta", alpha=phi * kappa, beta=(1 - phi) * kappa, shape=len(wins))
        pm.Binomial("obs", n=games, p=theta, observed=wins)
        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=seed,
                          progressbar=False)

    return idata.posterior["theta"].mean(dim=["chain", "draw"]).values


def compare_closed_form_to_mcmc(wins, games, **kwargs) -> dict | None:
    """Run both estimators on the same records and report the disagreement.

    Agreement to two or three decimals is what licenses replacing the sampler
    with the closed form in the pipeline. Disagreement would mean the analytic
    posterior does not match the model actually being sampled, which is worth
    knowing before either number is reported.
    """
    mcmc = posterior_means_mcmc(wins, games, **kwargs)
    if mcmc is None:
        return None
    kappa = fit_kappa(wins, games)
    closed = shrink_binomial(wins, games, kappa)
    diff = np.abs(closed - mcmc)
    return {
        "kappa": kappa,
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "correlation": float(np.corrcoef(closed, mcmc)[0, 1]),
        "closed_form": closed,
        "mcmc": mcmc,
    }
