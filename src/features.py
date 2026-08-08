"""Feature construction, all strictly point-in-time.

Every feature on a given game row is computed from games that finished before it.
The mechanism is the same throughout: sort within a group, shift by one, then
accumulate. The shift is what does the work, and removing it anywhere would
produce a model that looks excellent and predicts nothing.

Two deliberate asymmetries are worth flagging because they look inconsistent.

Team records reset at the season boundary. Rosters turn over, and a 2021 record
is weak evidence about a 2023 team, so carrying it forward mostly adds stale
signal. Prior-season finish is instead offered as its own feature, which lets the
model decide what it is worth rather than baking in an assumption.

Pitcher records do not reset. A starter is the same person across seasons, and
his history is the best available evidence about him, so discarding it every
April in order to be symmetric with teams would throw away real information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .retrosheet import starter_timeline, team_timeline
from .shrinkage import fit_gamma_poisson, fit_kappa, shrink_binomial, shrink_poisson


# ---------------------------------------------------------------------------
# Team features
# ---------------------------------------------------------------------------

def team_features(games: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Point-in-time win rate, shrunk talent, rolling form, and rest."""
    tl = team_timeline(games)
    grp = tl.groupby(["season", "team"], sort=False)["win"]

    # cumulative record strictly before the current game
    tl["prior_w"] = grp.transform(lambda s: s.shift(1).expanding().sum()).fillna(0.0)
    tl["prior_n"] = grp.transform(lambda s: s.shift(1).expanding().count()).fillna(0.0)

    tl["raw_wpct"] = np.where(tl["prior_n"] > 0,
                              tl["prior_w"] / tl["prior_n"].replace(0, np.nan), 0.5)
    tl["raw_wpct"] = tl["raw_wpct"].fillna(0.5)

    # Refit the concentration every day on the cross-section available that day.
    # Using only records that already exist keeps this point-in-time, and the
    # closed form makes the cost of ~500 fits per season negligible.
    kappa_by_date = {}
    for date, block in tl.groupby("date", sort=True):
        kappa_by_date[date] = fit_kappa(
            block["prior_w"].to_numpy(), block["prior_n"].to_numpy(),
            fallback=cfg.kappa_fallback, cap=cfg.kappa_cap, floor=cfg.kappa_floor,
            min_units=cfg.min_teams_for_fit, min_games_per_unit=cfg.min_games_per_unit,
        )
    tl["kappa"] = tl["date"].map(kappa_by_date)
    tl["talent"] = shrink_binomial(tl["prior_w"], tl["prior_n"], tl["kappa"].to_numpy())

    # rolling form over the team's own schedule, both venues
    tl["form"] = grp.transform(
        lambda s: s.shift(1).rolling(cfg.form_window, min_periods=1).mean()
    ).fillna(0.5)

    # days of rest, reset at the season boundary
    tl["prev_date"] = tl.groupby(["season", "team"], sort=False)["date"].shift(1)
    tl["rest"] = (tl["date"] - tl["prev_date"]).dt.days.sub(1)
    tl["rest"] = tl["rest"].fillna(cfg.rest_cap).clip(lower=0, upper=cfg.rest_cap)

    # prior season finishing win rate, as its own feature
    season_totals = (tl.groupby(["season", "team"], sort=False)["win"]
                       .agg(["sum", "count"]).reset_index())
    season_totals["final_wpct"] = season_totals["sum"] / season_totals["count"]
    season_totals["season"] = season_totals["season"] + 1     # becomes the *prior* for next year
    tl = tl.merge(season_totals[["season", "team", "final_wpct"]],
                  on=["season", "team"], how="left")
    tl = tl.rename(columns={"final_wpct": "prior_season"})
    tl["prior_season"] = tl["prior_season"].fillna(0.5)

    return tl[["game_id", "side", "raw_wpct", "talent", "form", "rest", "prior_season", "kappa"]]


# ---------------------------------------------------------------------------
# Starting pitcher features
# ---------------------------------------------------------------------------

def pitcher_features(games: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Shrunk runs allowed per start, and start count as an experience signal."""
    tl = starter_timeline(games)
    grp = tl.groupby("pitcher", sort=False)["runs_allowed"]

    tl["prior_runs"] = grp.transform(lambda s: s.shift(1).expanding().sum()).fillna(0.0)
    tl["prior_starts"] = grp.transform(lambda s: s.shift(1).expanding().count()).fillna(0.0)

    # Gamma-Poisson prior refit daily on the cross-section available that day
    ab_by_date = {}
    for date, block in tl.groupby("date", sort=True):
        ab_by_date[date] = fit_gamma_poisson(
            block["prior_runs"].to_numpy(), block["prior_starts"].to_numpy(),
            prior_mean=cfg.league_runs_per_game,
        )
    params = tl["date"].map(ab_by_date)
    tl["a"] = [p[0] for p in params]
    tl["b"] = [p[1] for p in params]

    tl["starter_ra"] = shrink_poisson(tl["prior_runs"], tl["prior_starts"], tl["a"], tl["b"])
    tl["starter_starts"] = tl["prior_starts"]

    return tl[["game_id", "side", "starter_ra", "starter_starts"]]


# ---------------------------------------------------------------------------
# Park
# ---------------------------------------------------------------------------

def park_features(games: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Point-in-time run factor for the park, relative to the league to date.

    Shrunk toward a neutral 1.0 by the same logic as everywhere else: a park with
    four games behind it should not be treated as a confirmed launching pad.
    """
    df = games[["game_id", "date", "park", "total_runs"]].copy().sort_values("date")

    league_runs = df["total_runs"].shift(1).expanding().sum()
    league_games = df["total_runs"].shift(1).expanding().count()
    league_rate = (league_runs / league_games).fillna(cfg.league_runs_per_game * 2)

    grp = df.groupby("park", sort=False)["total_runs"]
    park_runs = grp.transform(lambda s: s.shift(1).expanding().sum()).fillna(0.0)
    park_games = grp.transform(lambda s: s.shift(1).expanding().count()).fillna(0.0)

    prior_strength = 30.0     # games of neutral park evidence
    df["park_factor"] = ((park_runs + prior_strength * league_rate)
                         / (park_games + prior_strength)) / league_rate
    df["park_factor"] = df["park_factor"].replace([np.inf, -np.inf], 1.0).fillna(1.0)
    return df[["game_id", "park_factor"]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_panel(games: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Join every feature onto the game table, one row per game.

    Merges are explicit on (game_id, side) rather than positional, because the
    timelines are sorted by team and by pitcher rather than by date and any
    alignment by position would silently mismatch rows.
    """
    panel = games.copy()

    team = team_features(games, cfg)
    pitch = pitcher_features(games, cfg)
    park = park_features(games, cfg)

    for side in ("home", "away"):
        t = (team[team["side"] == side]
             .drop(columns="side")
             .rename(columns={c: f"{side}_{c}" for c in
                              ["raw_wpct", "talent", "form", "rest", "prior_season"]}))
        panel = panel.merge(t.drop(columns="kappa"), on="game_id", how="left")

        p = (pitch[pitch["side"] == side]
             .drop(columns="side")
             .rename(columns={c: f"{side}_{c}" for c in ["starter_ra", "starter_starts"]}))
        panel = panel.merge(p, on="game_id", how="left")

    panel = panel.merge(park, on="game_id", how="left")
    panel = panel.merge(team[team["side"] == "home"][["game_id", "kappa"]],
                        on="game_id", how="left")

    defaults = {
        "raw_wpct": 0.5, "talent": 0.5, "form": 0.5, "rest": cfg.rest_cap,
        "prior_season": 0.5, "starter_ra": cfg.league_runs_per_game, "starter_starts": 0.0,
    }
    for side in ("home", "away"):
        for name, value in defaults.items():
            col = f"{side}_{name}"
            if col in panel:
                panel[col] = panel[col].fillna(value)
    panel["park_factor"] = panel["park_factor"].fillna(1.0)

    return panel.sort_values(["date", "game_number", "home_team"]).reset_index(drop=True)
