"""Market odds: parsing, devigging, and joining to the game panel.

The file is keyed by date, then a list of games, each carrying a `gameView` and an
`odds` block. The odds block holds moneyline, pointspread and totals, and every
moneyline record carries both an opening and a current line for each sportsbook.
Having both is what makes closing line value measurable: a bet is placed at the
opening price and marked against the close.

A note on provenance, because it is the weakest assumption in the project.
`currentLine` is the last price the scraper captured. For a completed game that is
the closing line or very near it, but it is not stamped as such, so the CLV
figures inherit whatever staleness the collection introduced. Anyone rebuilding
this should confirm the collection cadence before treating CLV as exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Sportsbook shortNames to Retrosheet team codes. Two franchises appear under two
# names across the sample because they rebranded or relocated mid-file.
TEAM_MAP = {
    "ARI": "ARI", "AZ": "ARI",
    "ATH": "OAK", "OAK": "OAK",
    "ATL": "ATL", "BAL": "BAL", "BOS": "BOS",
    "CHC": "CHN", "CHW": "CHA",
    "CIN": "CIN", "CLE": "CLE", "COL": "COL", "DET": "DET", "HOU": "HOU",
    "KC": "KCA", "LAA": "ANA", "LAD": "LAN", "MIA": "MIA", "MIL": "MIL",
    "MIN": "MIN", "NYM": "NYN", "NYY": "NYA", "PHI": "PHI", "PIT": "PIT",
    "SD": "SDN", "SEA": "SEA", "SF": "SFN", "STL": "SLN", "TB": "TBA",
    "TEX": "TEX", "TOR": "TOR", "WAS": "WAS",
}

# All-star rosters, not clubs. Dropped rather than mapped.
NON_TEAMS = {"AL", "NL"}


def valid_american(odds) -> np.ndarray:
    """American odds are undefined between -100 and +100 exclusive.

    A handful of records in the file carry 0 or a small value, which are
    placeholders rather than prices. Silently converting them produces infinite
    payouts, so they are filtered rather than clipped.
    """
    o = np.asarray(odds, dtype=float)
    return np.isfinite(o) & (np.abs(o) >= 100.0)


def american_to_implied(odds) -> np.ndarray:
    """American moneyline to implied probability, vig included."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, 100.0 / (o + 100.0), np.abs(o) / (np.abs(o) + 100.0))


def american_to_decimal(odds) -> np.ndarray:
    """American moneyline to decimal payout, so a win returns stake * decimal."""
    o = np.asarray(odds, dtype=float)
    return np.where(o > 0, o / 100.0 + 1.0, 100.0 / np.abs(o) + 1.0)


def devig(q_home, q_away):
    """Remove the bookmaker margin by proportional normalisation.

    The two implied probabilities sum to more than one, and the excess is the
    margin. Dividing each by the total is the standard treatment and assumes the
    margin is applied proportionally across both sides. It is not the only
    choice: the power and Shin methods assume the favourite carries more of the
    margin, and on lopsided games they disagree with this by a point or two.
    Proportional is used here for transparency, and the effect on anything
    downstream is small because most MLB games are close to even.
    """
    q_home = np.asarray(q_home, dtype=float)
    q_away = np.asarray(q_away, dtype=float)
    total = q_home + q_away
    return q_home / total, q_away / total, total - 1.0


def load_odds(path: str | Path, seasons=None, game_type: str = "R") -> pd.DataFrame:
    """Flatten the JSON into one row per game, consensus and best price.

    Two prices are carried for a reason. The consensus across books is the better
    estimate of the market's true belief, so it is what the model is benchmarked
    against. The best available price is what a bettor could actually have taken,
    so it is what any simulated position is filled at. Conflating them either
    flatters the strategy or understates the benchmark.
    """
    with open(path) as handle:
        payload = json.load(handle)

    rows = []
    for date_str, games in payload.items():
        year = int(date_str[:4])
        if seasons is not None and year not in seasons:
            continue

        for game in games:
            view = game.get("gameView", {})
            if game_type is not None and view.get("gameType") != game_type:
                continue

            home = view.get("homeTeam", {}).get("shortName")
            away = view.get("awayTeam", {}).get("shortName")
            if home in NON_TEAMS or away in NON_TEAMS:
                continue
            if home not in TEAM_MAP or away not in TEAM_MAP:
                continue

            moneyline = (game.get("odds") or {}).get("moneyline") or []
            record = {
                "date": pd.Timestamp(date_str),
                "home_team": TEAM_MAP[home],
                "away_team": TEAM_MAP[away],
                "start": view.get("startDate"),
                "home_score": view.get("homeTeamScore"),
                "away_score": view.get("awayTeamScore"),
            }

            for label, key in (("open", "openingLine"), ("close", "currentLine")):
                h = [o[key]["homeOdds"] for o in moneyline
                     if o.get(key, {}).get("homeOdds") is not None]
                a = [o[key]["awayOdds"] for o in moneyline
                     if o.get(key, {}).get("awayOdds") is not None]
                h = list(np.asarray(h, dtype=float)[valid_american(h)]) if h else []
                a = list(np.asarray(a, dtype=float)[valid_american(a)]) if a else []
                if not h or not a:
                    record[f"{label}_n_books"] = 0
                    continue

                qh, qa, margin = devig(american_to_implied(h).mean(),
                                       american_to_implied(a).mean())
                record[f"{label}_p_home"] = float(qh)
                record[f"{label}_p_away"] = float(qa)
                record[f"{label}_vig"] = float(margin)
                # best price = longest odds available on that side
                record[f"{label}_best_home_dec"] = float(american_to_decimal(h).max())
                record[f"{label}_best_away_dec"] = float(american_to_decimal(a).max())
                record[f"{label}_n_books"] = len(h)

            rows.append(record)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["close_p_home", "open_p_home"])
    df["start"] = pd.to_datetime(df["start"], errors="coerce", utc=True)
    df = df.sort_values(["date", "start", "home_team"]).reset_index(drop=True)
    # doubleheaders share a date, so order within the day is the tie-break
    df["slot"] = df.groupby(["date", "home_team", "away_team"]).cumcount()
    return df


def games_from_odds(odds: pd.DataFrame) -> pd.DataFrame:
    """Build a game table from the odds file alone, for validation without a download.

    The odds file carries real dates, teams and final scores, so the team-level
    half of the pipeline can be exercised against real MLB before the Retrosheet
    logs are fetched. It has no starting pitchers or park IDs, so the pitcher and
    park rungs of the ladder are unavailable here; this is a check on the
    machinery against reality, not a substitute for the real panel.
    """
    df = odds.copy()
    df["season"] = df["date"].dt.year
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_runs"] = df["home_score"] + df["away_score"]
    df["game_number"] = df["slot"]
    df["park"] = df["home_team"] + "00"
    df["home_starter"] = ""
    df["away_starter"] = ""
    df = df.sort_values(["date", "game_number", "home_team"]).reset_index(drop=True)
    df["game_id"] = np.arange(len(df))
    df["has_odds"] = df["close_p_home"].notna()
    return df


def attach_odds(panel: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Left-join odds onto the panel on date, both teams, and doubleheader slot."""
    panel = panel.copy()
    panel["slot"] = panel.groupby(["date", "home_team", "away_team"]).cumcount()

    keep = [c for c in odds.columns
            if c.startswith(("open_", "close_")) or c in ("date", "home_team", "away_team", "slot")]
    merged = panel.merge(odds[keep], on=["date", "home_team", "away_team", "slot"], how="left")
    merged["has_odds"] = merged["close_p_home"].notna()
    return merged


def coverage_report(merged: pd.DataFrame) -> dict:
    """How much of the panel actually found a price, which gates every claim."""
    total = len(merged)
    matched = int(merged["has_odds"].sum())
    by_season = (merged.groupby("season")["has_odds"].mean().round(4).to_dict()
                 if "season" in merged else {})
    return {
        "games": total,
        "matched": matched,
        "match_rate": matched / total if total else 0.0,
        "by_season": by_season,
        "mean_closing_vig": float(merged.loc[merged["has_odds"], "close_vig"].mean())
        if matched else np.nan,
    }
