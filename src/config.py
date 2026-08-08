"""Configuration.

Retrosheet game logs are 161 comma-separated fields with no header, so the
column map lives here rather than as magic numbers scattered through the parser.
Field numbers below are zero-indexed; Retrosheet's own documentation numbers them
from one, so field 102 in their guide is index 101 here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

RETROSHEET_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
N_FIELDS = 161

# zero-indexed positions in the game log
COLUMNS = {
    "date": 0,
    "game_number": 1,        # 0 single, 1 first of doubleheader, 2 second
    "away_team": 3,
    "home_team": 6,
    "away_score": 9,
    "home_score": 10,
    "day_night": 12,
    "park": 16,
    "attendance": 17,
    "away_starter": 101,
    "home_starter": 103,
}


@dataclass(frozen=True)
class Config:
    seasons: tuple[int, ...] = (2021, 2022, 2023)

    # --- shrinkage -------------------------------------------------------
    kappa_cap: float = 400.0        # ceiling on the Beta-Binomial concentration
    kappa_floor: float = 20.0       # never shrink less than this in a noisy domain
    kappa_fallback: float = 60.0    # used before enough games exist to fit
    min_teams_for_fit: int = 5
    min_games_per_unit: int = 20    # per team, not league-wide: see fit_kappa

    # --- features --------------------------------------------------------
    form_window: int = 10
    rest_cap: int = 7
    league_runs_per_game: float = 4.5

    # --- walk-forward ----------------------------------------------------
    initial_train: int = 500
    step: int = 100

    # --- model -----------------------------------------------------------
    # Tree settings are deliberately severe. They were chosen on synthetic data
    # with matched signal-to-noise (see tests/synthetic.py), not by searching the
    # real out-of-sample window, so they cost no selection bias on the reported
    # number. Deeper trees scored worse on synthetic: at these correlations a
    # tree splits on noise.
    model: str = "gbm"              # "gbm" or "logistic"
    learning_rate: float = 0.05
    num_leaves: int = 2
    max_depth: int = 1
    n_rounds: int = 300
    feature_fraction: float = 0.8
    min_data_in_leaf: int = 200
    lambda_l2: float = 10.0
    logistic_C: float = 0.1

    # --- reporting -------------------------------------------------------
    calibration_split: float = 0.7
    holdout_season: int | None = None   # set to e.g. 2023 for a clean single-shot estimate
    seed: int = 0

    # How many configurations were evaluated on the reported split. Update this
    # honestly; it is what makes the headline number interpretable.
    n_hyperparameter_configs: int = 4
    n_feature_sets: int = 6


# Nested feature sets, scored on one identical out-of-sample split. Each rung
# adds a block, so the table reports what each block is worth rather than only
# the final number.
LADDER: list[tuple[str, list[str]]] = [
    ("raw team win pct",        ["home_raw_wpct", "away_raw_wpct"]),
    ("+ EB shrunk talent",      ["home_talent", "away_talent"]),
    ("+ form and rest",         ["home_talent", "away_talent",
                                 "home_form", "away_form",
                                 "home_rest", "away_rest"]),
    ("+ starting pitcher",      ["home_talent", "away_talent",
                                 "home_form", "away_form",
                                 "home_rest", "away_rest",
                                 "home_starter_ra", "away_starter_ra",
                                 "home_starter_starts", "away_starter_starts"]),
    ("+ park and prior season", ["home_talent", "away_talent",
                                 "home_form", "away_form",
                                 "home_rest", "away_rest",
                                 "home_starter_ra", "away_starter_ra",
                                 "home_starter_starts", "away_starter_starts",
                                 "park_factor", "home_prior_season", "away_prior_season"]),
]

FULL_FEATURES = LADDER[-1][1]

DEFAULT = Config()
