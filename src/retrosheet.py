"""Retrosheet game log ingestion.

The game logs are the whole data source. They carry, for every regular season
game since 1871, the date, teams, score, park, and both starting pitchers, which
is everything this project needs. No scraping, no rate limits, no API key.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import COLUMNS, DATA, N_FIELDS, RETROSHEET_URL


def download(seasons, dest: Path = DATA) -> list[Path]:
    """Fetch and extract game logs, skipping any season already present.

    Retrosheet ships each season zipped, and the file inside is sometimes
    uppercase and sometimes not, so the search is case-insensitive rather than
    assuming either.
    """
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for year in seasons:
        found = list(dest.glob(f"[gG][lL]{year}.[tT][xX][tT]"))
        if found:
            paths.append(found[0])
            continue
        url = RETROSHEET_URL.format(year=year)
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".txt"):
                    archive.extract(name, dest)
                    paths.append(dest / name)
                    break
    return paths


def _read_one(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, header=None, names=[f"f{i}" for i in range(N_FIELDS)],
                      dtype=str, low_memory=False)
    take = {name: raw[f"f{i}"] for name, i in COLUMNS.items()}
    return pd.DataFrame(take)


def load_games(seasons, dest: Path = DATA, paths=None) -> pd.DataFrame:
    """Return one row per game, chronologically ordered, with a stable game_id.

    Ordering matters more than it looks: every point-in-time feature downstream
    is built by shifting within a group, so a wrong sort silently leaks the
    future. Doubleheaders share a date, which is why game_number is part of the
    sort key rather than an afterthought.
    """
    paths = paths if paths is not None else download(seasons, dest)
    frames = [_read_one(Path(p)) for p in paths]
    df = pd.concat(frames, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df["game_number"] = pd.to_numeric(df["game_number"], errors="coerce").fillna(0).astype(int)
    for col in ("away_score", "home_score", "attendance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("away_team", "home_team", "away_starter", "home_starter", "park"):
        df[col] = df[col].astype(str).str.strip().str.strip('"')

    df = df.dropna(subset=["date", "away_score", "home_score"])
    df = df[df["away_score"] != df["home_score"]]        # ties are not scored games

    df["season"] = df["date"].dt.year
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["total_runs"] = df["home_score"] + df["away_score"]

    df = (df.sort_values(["date", "game_number", "home_team"])
            .reset_index(drop=True))
    df["game_id"] = np.arange(len(df))
    return df


def team_timeline(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, both sides stacked.

    Almost every team-level feature is a rolling or expanding statistic over a
    team's own schedule regardless of venue. Building that from the game table
    directly is where the original version went wrong: grouping by home_team
    produces a rolling window over home games only, which is a different and
    much noisier quantity than recent form.
    """
    home = games[["game_id", "date", "game_number", "season", "home_team", "home_win"]].copy()
    home.columns = ["game_id", "date", "game_number", "season", "team", "win"]
    home["side"] = "home"

    away = games[["game_id", "date", "game_number", "season", "away_team", "home_win"]].copy()
    away.columns = ["game_id", "date", "game_number", "season", "team", "win"]
    away["win"] = 1 - away["win"]
    away["side"] = "away"

    return (pd.concat([home, away], ignore_index=True)
              .sort_values(["team", "date", "game_number"])
              .reset_index(drop=True))


def starter_timeline(games: pd.DataFrame) -> pd.DataFrame:
    """One row per starting pitcher per start, with runs allowed by his team.

    Runs allowed is the opposing side's score, so it charges the starter for the
    bullpen and the defence as well. That is a real limitation and it is stated
    in the README; play-by-play files would give earned runs against the starter
    alone, at the cost of a much heavier ingest.
    """
    home = games[["game_id", "date", "game_number", "season", "home_starter", "away_score"]].copy()
    home.columns = ["game_id", "date", "game_number", "season", "pitcher", "runs_allowed"]
    home["side"] = "home"

    away = games[["game_id", "date", "game_number", "season", "away_starter", "home_score"]].copy()
    away.columns = ["game_id", "date", "game_number", "season", "pitcher", "runs_allowed"]
    away["side"] = "away"

    out = pd.concat([home, away], ignore_index=True)
    out = out[out["pitcher"].astype(bool) & (out["pitcher"] != "nan")]
    return out.sort_values(["pitcher", "date", "game_number"]).reset_index(drop=True)
