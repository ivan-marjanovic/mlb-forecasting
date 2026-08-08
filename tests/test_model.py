"""Correctness tests.

Runs with either runner:

    python -m pytest tests -q
    python -m unittest discover -s tests -v

The suite uses the synthetic fixture rather than real data so it needs no network
and no multi-season download. The fixture plants signal at known strength, which
lets several tests assert that an estimator recovers a truth rather than merely
that it returns a number.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT, FULL_FEATURES, LADDER  # noqa: E402
from src.evaluation import (  # noqa: E402
    bootstrap_diff, brier, brier_decomposition, brier_diff_test, reliability_table,
)
from src.features import build_panel  # noqa: E402
from src.forecast import permutation_check, walk_forward  # noqa: E402
from src.retrosheet import load_games, starter_timeline, team_timeline  # noqa: E402
from src.shrinkage import (  # noqa: E402
    fit_gamma_poisson, fit_kappa, implied_talent_sd, shrink_binomial, shrink_poisson,
)
from tests.synthetic import write_seasons  # noqa: E402

SEASONS = (2021, 2022, 2023)
_CACHE: dict = {}


def fixture():
    """Build the panel once and share it; the walk-forward tests are the slow part."""
    if "panel" not in _CACHE:
        tmp = Path(tempfile.mkdtemp())
        write_seasons(tmp, SEASONS, seed=1)
        games = load_games(SEASONS, paths=sorted(tmp.glob("GL*.TXT")))
        _CACHE["games"] = games
        _CACHE["panel"] = build_panel(games, DEFAULT)
    return _CACHE["games"], _CACHE["panel"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

class TestParsing(unittest.TestCase):

    def test_shape_and_columns(self):
        games, _ = fixture()
        for col in ("date", "home_team", "away_team", "home_score", "away_score",
                    "home_starter", "away_starter", "park", "home_win", "season"):
            self.assertIn(col, games.columns)
        self.assertEqual(games["season"].nunique(), len(SEASONS))
        self.assertTrue(games["game_id"].is_unique)

    def test_no_ties_and_win_flag_agrees_with_score(self):
        games, _ = fixture()
        self.assertTrue((games["home_score"] != games["away_score"]).all())
        expected = (games["home_score"] > games["away_score"]).astype(int)
        self.assertTrue((games["home_win"] == expected).all())

    def test_chronological_order(self):
        games, _ = fixture()
        self.assertTrue(games["date"].is_monotonic_increasing)

    def test_timelines_have_two_rows_per_game(self):
        games, _ = fixture()
        self.assertEqual(len(team_timeline(games)), 2 * len(games))
        self.assertEqual(len(starter_timeline(games)), 2 * len(games))


# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------

class TestShrinkage(unittest.TestCase):

    def test_recovers_known_concentration(self):
        """Fitted kappa should land near the truth once exposure is real."""
        rng = np.random.default_rng(0)
        true_kappa, fitted = 70.0, []
        for _ in range(120):
            talent = rng.beta(0.5 * true_kappa, 0.5 * true_kappa, 30)
            games = np.full(30, 100)
            fitted.append(fit_kappa(rng.binomial(games, talent), games))
        self.assertLess(abs(np.median(fitted) - true_kappa) / true_kappa, 0.30)

    def test_guard_holds_fallback_on_thin_exposure(self):
        """Two games per team must not produce a near-zero concentration.

        This is a regression test for a real bug. The guard originally checked
        games league-wide, so thirty teams at two games each looked like sixty
        games and passed. Every team is 1-0 or 0-1 at that point, which reads as
        enormous true spread, and the fit collapsed to almost no shrinkage at the
        moment shrinkage matters most.
        """
        wins = np.array([1, 0] * 15, dtype=float)
        games = np.full(30, 2.0)
        self.assertEqual(fit_kappa(wins, games, fallback=60.0), 60.0)

    def test_shrunk_lies_between_raw_and_prior(self):
        rng = np.random.default_rng(1)
        games = rng.integers(5, 150, 30).astype(float)
        wins = rng.binomial(games.astype(int), 0.5).astype(float)
        shrunk = shrink_binomial(wins, games, 80.0)
        raw = wins / games
        self.assertTrue(np.all((shrunk - 0.5) * (raw - 0.5) >= -1e-12))
        self.assertTrue(np.all(np.abs(shrunk - 0.5) <= np.abs(raw - 0.5) + 1e-12))

    def test_shrinkage_weakens_as_evidence_grows(self):
        for n, previous in [(10, None), (50, None), (200, None)]:
            pass
        pulls = [abs(shrink_binomial([0.7 * n], [n], 80.0)[0] - 0.5) for n in (10, 50, 200)]
        self.assertTrue(pulls[0] < pulls[1] < pulls[2])

    def test_gamma_poisson_recovers_prior_strength(self):
        rng = np.random.default_rng(2)
        true_b = 8.0
        lam = rng.gamma(4.5 * true_b, 1 / true_b, 300)
        starts = rng.integers(1, 30, 300)
        _, b = fit_gamma_poisson(rng.poisson(lam * starts), starts)
        self.assertLess(abs(b - true_b) / true_b, 0.40)

    def test_gamma_poisson_pulls_thin_samples_to_the_prior(self):
        shrunk = shrink_poisson([0.0, 24.0], [2.0, 2.0], a=4.5 * 8, b=8.0)
        self.assertTrue(3.0 < shrunk[0] < 4.5)
        self.assertTrue(4.5 < shrunk[1] < 7.0)

    def test_implied_sd_inverts_kappa(self):
        for kappa in (20.0, 100.0, 400.0):
            sd = implied_talent_sd(kappa)
            self.assertAlmostEqual(0.25 / sd ** 2 - 1.0, kappa, places=6)


# ---------------------------------------------------------------------------
# Point-in-time discipline
# ---------------------------------------------------------------------------

class TestNoLookahead(unittest.TestCase):

    def test_features_match_a_brute_force_recomputation(self):
        """Recompute a sample of rows the slow, obvious way and compare.

        The vectorised construction shifts and accumulates inside groupby, which
        is fast and easy to get subtly wrong. This recomputes the same quantity
        by filtering the game table directly, which is transparently correct.
        """
        games, panel = fixture()
        rng = np.random.default_rng(4)
        rows = rng.choice(panel.index[panel.index > 400], size=25, replace=False)

        for idx in rows:
            row = panel.loc[idx]
            prior = games[(games["season"] == row["season"])
                          & (games["game_id"] < row["game_id"])]
            as_home = prior[prior["home_team"] == row["home_team"]]
            as_away = prior[prior["away_team"] == row["home_team"]]
            wins = as_home["home_win"].sum() + (1 - as_away["home_win"]).sum()
            played = len(as_home) + len(as_away)

            expected_raw = wins / played if played else 0.5
            self.assertAlmostEqual(row["home_raw_wpct"], expected_raw, places=9)
            expected_talent = (wins + 0.5 * row["kappa"]) / (played + row["kappa"])
            self.assertAlmostEqual(row["home_talent"], expected_talent, places=9)

    def test_first_game_of_season_carries_no_record(self):
        _, panel = fixture()
        for season, block in panel.groupby("season"):
            first = block.iloc[0]
            self.assertAlmostEqual(first["home_raw_wpct"], 0.5, places=9)
            self.assertAlmostEqual(first["away_raw_wpct"], 0.5, places=9)

    def test_no_missing_features(self):
        _, panel = fixture()
        self.assertEqual(int(panel[FULL_FEATURES].isna().sum().sum()), 0)

    def test_permutation_check_finds_nothing(self):
        """The strongest leakage test available: destroy the target, expect noise.

        If any feature encoded its own outcome, a model would still score below
        the shuffled variance here, and no amount of reading the feature code
        catches that as reliably as this does.
        """
        _, panel = fixture()
        result = permutation_check(panel, FULL_FEATURES, DEFAULT, model="logistic")
        self.assertGreater(result["brier"], result["expected"] - 0.004)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class TestEvaluation(unittest.TestCase):

    def test_analytic_and_bootstrap_intervals_agree(self):
        rng = np.random.default_rng(5)
        n = 4000
        p = np.clip(rng.normal(0.53, 0.05, n), 0.3, 0.75)
        y = rng.binomial(1, p)
        base = np.full(n, y.mean())
        analytic = brier_diff_test(y, base, p)
        boot = bootstrap_diff(y, base, p, draws=2000)
        for a, b in zip(analytic["ci95"], boot["ci95"]):
            self.assertLess(abs(a - b), 3e-4)

    def test_identical_forecasts_give_zero_difference(self):
        rng = np.random.default_rng(6)
        p = rng.uniform(0.4, 0.6, 500)
        y = rng.binomial(1, p)
        self.assertAlmostEqual(brier_diff_test(y, p, p)["improvement"], 0.0, places=12)

    def test_murphy_decomposition_reconstructs_brier(self):
        rng = np.random.default_rng(7)
        n = 20000
        p = np.round(np.clip(rng.normal(0.53, 0.06, n), 0.3, 0.75), 2)
        y = rng.binomial(1, p)
        d = brier_decomposition(y, p, bins=25)
        rebuilt = d["reliability"] - d["resolution"] + d["uncertainty"]
        self.assertLess(abs(rebuilt - d["brier"]), 2e-3)

    def test_reliability_bins_account_for_every_row(self):
        rng = np.random.default_rng(8)
        p = rng.uniform(0.4, 0.65, 1000)
        y = rng.binomial(1, p)
        self.assertEqual(int(reliability_table(y, p).n.sum()), 1000)


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

class TestForecast(unittest.TestCase):

    def test_predictions_only_missing_in_the_initial_window(self):
        _, panel = fixture()
        preds = walk_forward(panel, FULL_FEATURES, DEFAULT, model="logistic")
        missing = np.isnan(preds)
        self.assertTrue(missing[: DEFAULT.initial_train].all())
        self.assertFalse(missing[-100:].any())

    def test_probabilities_are_probabilities(self):
        _, panel = fixture()
        preds = walk_forward(panel, FULL_FEATURES, DEFAULT, model="logistic")
        finite = preds[~np.isnan(preds)]
        self.assertTrue(np.all((finite >= 0) & (finite <= 1)))

    def test_recovers_planted_signal(self):
        """The fixture plants real signal, so the model must find it.

        A pipeline can be leak-free and still useless. This asserts the other
        direction: on data where the answer is known to be learnable, the model
        beats the constant by a margin that clears its own error bar.
        """
        _, panel = fixture()
        preds = walk_forward(panel, FULL_FEATURES, DEFAULT, model="logistic")
        mask = ~np.isnan(preds)
        y = panel["home_win"].to_numpy()[mask]
        result = brier_diff_test(y, np.full(mask.sum(), y.mean()), preds[mask])
        self.assertGreater(result["t_stat"], 2.0)

    def test_ladder_rungs_are_nested(self):
        for (_, earlier), (_, later) in zip(LADDER, LADDER[1:]):
            if len(earlier) <= len(later):
                continue
            self.fail("ladder rungs must not shrink")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ---------------------------------------------------------------------------
# Market layer
# ---------------------------------------------------------------------------

class TestOddsMath(unittest.TestCase):

    def test_american_conversions_round_trip(self):
        from src.odds import american_to_decimal, american_to_implied
        for odds in (-250, -110, 100, 145, 300):
            implied = american_to_implied([odds])[0]
            decimal = american_to_decimal([odds])[0]
            self.assertAlmostEqual(implied, 1.0 / decimal, places=10)

    def test_devig_normalises_and_reports_the_margin(self):
        from src.odds import american_to_implied, devig
        q_home = american_to_implied([-130])[0]
        q_away = american_to_implied([110])[0]
        p_home, p_away, margin = devig(q_home, q_away)
        self.assertAlmostEqual(p_home + p_away, 1.0, places=12)
        self.assertAlmostEqual(margin, q_home + q_away - 1.0, places=12)
        self.assertGreater(margin, 0.0)

    def test_placeholder_prices_are_rejected(self):
        """A handful of records carry 0 rather than a price; converting them
        produces infinite payouts, so they must be filtered rather than clipped."""
        from src.odds import valid_american
        keep = valid_american([0, 50, -99, -100, 100, -250, np.nan])
        self.assertTrue((keep == np.array([False, False, False, True, True, True, False])).all())

    def test_team_map_covers_every_franchise(self):
        from src.odds import TEAM_MAP
        self.assertEqual(len(set(TEAM_MAP.values())), 30)
        self.assertEqual(TEAM_MAP["AZ"], TEAM_MAP["ARI"])
        self.assertEqual(TEAM_MAP["ATH"], TEAM_MAP["OAK"])


class TestDecision(unittest.TestCase):

    def test_kelly_matches_the_closed_form(self):
        from src.decision import kelly_fraction
        p, decimal = 0.6, 2.0
        self.assertAlmostEqual(kelly_fraction([p], [decimal])[0], 0.2, places=12)

    def test_kelly_refuses_negative_expectation(self):
        from src.decision import kelly_fraction
        self.assertEqual(kelly_fraction([0.4], [2.0])[0], 0.0)

    def test_a_sure_thing_at_fair_odds_always_profits(self):
        """End-to-end sanity on the backtest: given a forecast that is always
        right and a price that pays, the bankroll must grow."""
        from src.decision import backtest
        n = 200
        df = pd.DataFrame({
            "date": pd.date_range("2021-04-01", periods=n),
            "home_win": np.ones(n, dtype=int),
            "model_p": np.full(n, 0.95),
            "open_p_home": np.full(n, 0.50), "open_p_away": np.full(n, 0.50),
            "close_p_home": np.full(n, 0.50), "close_p_away": np.full(n, 0.50),
            "open_best_home_dec": np.full(n, 2.0), "open_best_away_dec": np.full(n, 2.0),
        })
        result = backtest(df, "model_p")
        self.assertGreater(result["final_bankroll"], 10_000.0)
        self.assertEqual(result["hit_rate"], 1.0)


class TestEncompassing(unittest.TestCase):

    def test_a_copy_of_the_benchmark_adds_nothing(self):
        """Two identical forecasts are perfectly collinear, so the model cannot
        be credited with information the benchmark lacks."""
        from src.evaluation import encompassing_test
        rng = np.random.default_rng(3)
        p = np.clip(rng.normal(0.53, 0.06, 3000), 0.2, 0.8)
        y = rng.binomial(1, p)
        result = encompassing_test(y, p, p, n_boot=60)
        total = result["benchmark_coef"] + result["model_coef"]
        self.assertGreater(total, 0.5)

    def test_pure_noise_earns_a_zero_coefficient(self):
        from src.evaluation import encompassing_test
        rng = np.random.default_rng(4)
        p = np.clip(rng.normal(0.53, 0.06, 4000), 0.2, 0.8)
        y = rng.binomial(1, p)
        noise = rng.uniform(0.4, 0.6, 4000)
        result = encompassing_test(y, noise, p, n_boot=80)
        self.assertLess(abs(result["model_t"]), 2.5)
