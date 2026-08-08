"""Log a frozen model's forecasts forward, one day at a time.

A backtest is a claim about the past that the person making it also designed. A
frozen model logging predictions before outcomes exist is a claim nobody can
tune after the fact, which is why it is worth more than any in-sample number.

Two sources, both free:

  MLB StatsAPI (statsapi.mlb.com) for schedule and completed results. Retrosheet
  publishes at season end, so it cannot drive a live feature set.
  The Odds API (the-odds-api.com) for today's moneylines. Needs ODDS_API_KEY.

Each run appends one row per game to `live_predictions.csv` and never rewrites a
past row. Grading happens in a separate pass once results exist, so a prediction
and its outcome are always written by different invocations and the file cannot
be quietly improved after the fact.

    python -m src.live --dry-run     # fetch and print, write nothing
    python -m src.live               # append today's forecasts
    python -m src.live --grade       # fill outcomes for past rows
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT, ROOT
from .features import team_features
from .odds import TEAM_MAP, american_to_implied, devig, valid_american

LOG = ROOT / "live_predictions.csv"
STATS_API = "https://statsapi.mlb.com/api/v1/schedule"
ODDS_API = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"

FIELDS = ["logged_at", "game_date", "away_team", "home_team", "model_p_home",
          "market_p_home", "best_home_dec", "best_away_dec", "n_books",
          "edge_home", "kelly_stake", "home_win", "graded_at"]

# StatsAPI uses its own three-letter codes; map them to Retrosheet.
STATS_TO_RETRO = dict(TEAM_MAP)
STATS_TO_RETRO.update({"CWS": "CHA", "WSH": "WAS", "SFG": "SFN", "SDP": "SDN",
                       "TBR": "TBA", "KCR": "KCA"})


def _get(url: str, params: dict) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(full, timeout=30) as response:
        return json.loads(response.read())


def season_results(season: int, through: dt.date) -> pd.DataFrame:
    """Completed regular season games this year, for building features."""
    payload = _get(STATS_API, {
        "hydrate": "team", "sportId": 1, "season": season, "gameType": "R",
        "startDate": f"{season}-03-01", "endDate": through.isoformat(),
    })
    rows = []
    for day in payload.get("dates", []):
        for game in day.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            teams = game["teams"]
            home = STATS_TO_RETRO.get(teams["home"]["team"].get("abbreviation"))
            away = STATS_TO_RETRO.get(teams["away"]["team"].get("abbreviation"))
            if not home or not away:
                continue
            hs, as_ = teams["home"].get("score"), teams["away"].get("score")
            if hs is None or as_ is None or hs == as_:
                continue
            rows.append({"date": pd.Timestamp(day["date"]), "game_number": 0,
                         "season": season, "home_team": home, "away_team": away,
                         "home_score": hs, "away_score": as_,
                         "home_win": int(hs > as_), "total_runs": hs + as_})
    df = pd.DataFrame(rows).sort_values(["date", "home_team"]).reset_index(drop=True)
    df["game_id"] = np.arange(len(df))
    return df


def todays_odds(api_key: str) -> pd.DataFrame:
    """Current moneylines, consensus devigged and best price across books."""
    payload = _get(ODDS_API, {"apiKey": api_key, "regions": "us",
                              "markets": "h2h", "oddsFormat": "american"})
    rows = []
    for event in payload:
        home = STATS_TO_RETRO.get(_abbrev(event.get("home_team", "")))
        away = STATS_TO_RETRO.get(_abbrev(event.get("away_team", "")))
        if not home or not away:
            continue
        h, a = [], []
        for book in event.get("bookmakers", []):
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    if not valid_american([price])[0]:
                        continue
                    if outcome.get("name") == event.get("home_team"):
                        h.append(price)
                    elif outcome.get("name") == event.get("away_team"):
                        a.append(price)
        if not h or not a:
            continue
        qh, _, _ = devig(american_to_implied(h).mean(), american_to_implied(a).mean())
        rows.append({
            "game_date": pd.Timestamp(event["commence_time"]).date().isoformat(),
            "home_team": home, "away_team": away,
            "market_p_home": float(qh), "n_books": len(h),
            "best_home_dec": float(max(_dec(x) for x in h)),
            "best_away_dec": float(max(_dec(x) for x in a)),
        })
    return pd.DataFrame(rows)


def _dec(odds: float) -> float:
    return odds / 100.0 + 1.0 if odds > 0 else 100.0 / abs(odds) + 1.0


_NAME_TO_ABBREV = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU", "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Athletics": "OAK", "Athletics": "OAK", "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD",
    "Seattle Mariners": "SEA", "San Francisco Giants": "SF", "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB", "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}


def _abbrev(full_name: str) -> str:
    return _NAME_TO_ABBREV.get(full_name, full_name)


def forecast_today(cfg=DEFAULT, api_key: str | None = None) -> pd.DataFrame:
    """Build features from season to date, price today's slate, size positions."""
    api_key = api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("set ODDS_API_KEY")

    today = dt.date.today()
    history = season_results(today.year, today - dt.timedelta(days=1))
    if len(history) < 200:
        raise RuntimeError(f"only {len(history)} completed games this season; too early")

    board = todays_odds(api_key)
    if board.empty:
        return board

    # Extend history with today's fixtures so features are computed for them,
    # then keep only the fixture rows. Outcomes for today are unknown and unused.
    fixtures = board[["game_date", "home_team", "away_team"]].copy()
    fixtures["date"] = pd.to_datetime(fixtures["game_date"])
    fixtures["game_number"] = 0
    fixtures["season"] = today.year
    fixtures["home_win"] = 0            # placeholder, never read: it is shifted out
    fixtures["total_runs"] = 0
    combined = pd.concat([history, fixtures], ignore_index=True)
    combined = combined.sort_values(["date", "game_number", "home_team"]).reset_index(drop=True)
    combined["game_id"] = np.arange(len(combined))

    tf = team_features(combined, cfg)
    panel = combined.copy()
    for side in ("home", "away"):
        block = (tf[tf["side"] == side].drop(columns=["side", "kappa"])
                 .rename(columns={c: f"{side}_{c}" for c in
                                  ["raw_wpct", "talent", "form", "rest", "prior_season"]}))
        panel = panel.merge(block, on="game_id", how="left")

    slate = panel[panel["date"].dt.date == today].copy()
    features = ["home_talent", "away_talent", "home_form", "away_form",
                "home_rest", "away_rest", "home_prior_season", "away_prior_season"]

    from .forecast import _fit_one, _predict_one
    train = panel[panel["date"].dt.date < today]
    model = _fit_one(train[features].to_numpy(), train["home_win"].to_numpy(),
                     cfg, "logistic")
    slate["model_p_home"] = _predict_one(model, slate[features].to_numpy(), "logistic")

    out = slate.merge(board, on=["home_team", "away_team"], how="inner")
    out["edge_home"] = out["model_p_home"] - out["market_p_home"]

    from .decision import kelly_fraction
    take_home = out["edge_home"] >= (1 - out["model_p_home"]) - (1 - out["market_p_home"])
    prob = np.where(take_home, out["model_p_home"], 1 - out["model_p_home"])
    price = np.where(take_home, out["best_home_dec"], out["best_away_dec"])
    edge = np.abs(out["edge_home"])
    out["kelly_stake"] = np.where(edge > 0.02,
                                  np.minimum(0.25 * kelly_fraction(prob, price), 0.02), 0.0)
    out["logged_at"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    out["home_win"] = ""
    out["graded_at"] = ""
    print("\n--- FINAL DATA DEBUG ---")
    print("Total matched games:", len(out))
    print("Available columns:", out.columns.tolist())
    print("------------------------\n")
    out['game_date'] = out.get('date', out.get('game_date_x'))
    return out[FIELDS]


def append(rows: pd.DataFrame, path: Path = LOG) -> int:
    """Append only games not already logged. Existing rows are never rewritten."""
    seen = set()
    if path.exists():
        prior = pd.read_csv(path)
        seen = set(zip(prior["game_date"], prior["home_team"], prior["away_team"]))

    fresh = rows[~rows.apply(
        lambda r: (r["game_date"], r["home_team"], r["away_team"]) in seen, axis=1)]
    if fresh.empty:
        return 0

    write_header = not path.exists()
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for _, row in fresh.iterrows():
            writer.writerow(row.to_dict())
    return len(fresh)


def grade(path: Path = LOG) -> int:
    """Fill outcomes for logged games that have since finished."""
    if not path.exists():
        return 0
    log = pd.read_csv(path)
    pending = log["home_win"].isna() | (log["home_win"].astype(str) == "")
    if not pending.any():
        return 0

    filled = 0
    for season in sorted({int(d[:4]) for d in log.loc[pending, "game_date"]}):
        results = season_results(season, dt.date.today())
        key = {(r["date"].date().isoformat(), r["home_team"], r["away_team"]): r["home_win"]
               for _, r in results.iterrows()}
        for idx in log.index[pending]:
            row = log.loc[idx]
            outcome = key.get((row["game_date"], row["home_team"], row["away_team"]))
            if outcome is not None:
                log.at[idx, "home_win"] = int(outcome)
                log.at[idx, "graded_at"] = dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds")
                filled += 1
    log.to_csv(path, index=False)
    return filled


def summarise(path: Path = LOG) -> dict:
    """Score the live log so far, against the base rate and against the market."""
    if not path.exists():
        return {}
    log = pd.read_csv(path)
    done = log[log["home_win"].notna() & (log["home_win"].astype(str) != "")].copy()
    if len(done) < 30:
        return {"graded": len(done), "note": "fewer than 30 graded games"}

    y = done["home_win"].astype(int).to_numpy()
    from .evaluation import brier, brier_diff_test
    base = np.full(len(y), y.mean())
    model = brier_diff_test(y, base, done["model_p_home"].to_numpy())
    return {
        "graded": len(done),
        "base_rate_brier": brier(y, base),
        "model_brier": brier(y, done["model_p_home"].to_numpy()),
        "market_brier": brier(y, done["market_p_home"].to_numpy()),
        "improvement_vs_base": model["improvement"],
        "t_stat": model["t_stat"],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--grade", action="store_true")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    if args.grade:
        print(f"graded {grade()} games")
    elif args.summary:
        print(json.dumps(summarise(), indent=2, default=float))
    else:
        board = forecast_today()
        if board.empty:
            print("no games priced today")
        elif args.dry_run:
            print(board.to_string(index=False))
        else:
            print(f"appended {append(board)} forecasts")
