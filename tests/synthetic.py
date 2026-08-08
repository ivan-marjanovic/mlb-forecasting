"""Generate a synthetic season in exact Retrosheet game-log format.

The test suite needs data. Downloading three real seasons to run tests makes the
suite slow, network-dependent, and impossible to run offline, so instead this
writes files in the real 161-field layout with signal planted at known strength.

That buys something the real files cannot: ground truth. Team talent, pitcher
quality, and park effects are all known here, so a test can assert that the
shrinkage recovers them and that the model finds the signal that was planted
rather than merely producing a number.
"""

from __future__ import annotations

import numpy as np

from src.config import N_FIELDS

TEAMS = ["ANA", "ARI", "ATL", "BAL", "BOS", "CHA", "CHN", "CIN", "CLE", "COL",
         "DET", "HOU", "KCA", "LAN", "MIA", "MIL", "MIN", "NYA", "NYN", "OAK",
         "PHI", "PIT", "SDN", "SEA", "SFN", "SLN", "TBA", "TEX", "TOR", "WAS"]

STARTERS_PER_TEAM = 5
HOME_FIELD_LOGIT = 0.16          # about 4 points of win probability
TALENT_SD_LOGIT = 0.22           # spreads true win rates roughly .40 to .60
PITCHER_SD_LOGIT = 0.20


def _row(date, game_num, away, home, away_score, home_score, park,
         away_sp, home_sp) -> str:
    f = ["" for _ in range(N_FIELDS)]
    f[0] = date.strftime("%Y%m%d")
    f[1] = str(game_num)
    f[2] = date.strftime("%a")
    f[3], f[6] = away, home
    f[4], f[7] = "NL", "NL"
    f[9], f[10] = str(away_score), str(home_score)
    f[12] = "N"
    f[16] = park
    f[17] = "30000"
    f[101], f[102] = away_sp, f"SP {away_sp}"
    f[103], f[104] = home_sp, f"SP {home_sp}"
    return ",".join(f'"{v}"' for v in f)


def write_season(path, season: int, seed: int = 0, games_per_team: int = 162):
    """Write one season and return the ground truth used to generate it."""
    import datetime as dt

    rng = np.random.default_rng(seed + season)
    talent = {t: rng.normal(0, TALENT_SD_LOGIT) for t in TEAMS}
    park_effect = {t: rng.normal(0, 0.15) for t in TEAMS}
    rotations = {t: [f"{t.lower()}p{i}" for i in range(STARTERS_PER_TEAM)] for t in TEAMS}
    pitcher_skill = {p: rng.normal(0, PITCHER_SD_LOGIT)
                     for t in TEAMS for p in rotations[t]}

    opening = dt.date(season, 4, 1)
    rows, truth = [], []
    counters = {t: 0 for t in TEAMS}

    n_days = games_per_team
    for day in range(n_days):
        date = opening + dt.timedelta(days=day)
        order = list(TEAMS)
        rng.shuffle(order)
        for i in range(0, len(order) - 1, 2):
            away, home = order[i], order[i + 1]
            away_sp = rotations[away][counters[away] % STARTERS_PER_TEAM]
            home_sp = rotations[home][counters[home] % STARTERS_PER_TEAM]
            counters[away] += 1
            counters[home] += 1

            logit = (HOME_FIELD_LOGIT
                     + (talent[home] - talent[away])
                     + (pitcher_skill[home_sp] - pitcher_skill[away_sp]))
            p_home = 1.0 / (1.0 + np.exp(-logit))
            home_wins = rng.random() < p_home

            base = 4.4 * np.exp(park_effect[home])
            home_runs = rng.poisson(base * np.exp(-pitcher_skill[away_sp]))
            away_runs = rng.poisson(base * np.exp(-pitcher_skill[home_sp]))
            if home_wins and home_runs <= away_runs:
                home_runs = away_runs + 1 + rng.poisson(0.5)
            if not home_wins and away_runs <= home_runs:
                away_runs = home_runs + 1 + rng.poisson(0.5)

            rows.append(_row(date, 0, away, home, away_runs, home_runs,
                             f"{home}01", away_sp, home_sp))
            truth.append({"date": date, "home": home, "away": away,
                          "p_home": p_home, "home_win": int(home_wins)})

    path = str(path)
    with open(path, "w") as handle:
        handle.write("\n".join(rows) + "\n")

    return {"path": path, "talent": talent, "pitcher_skill": pitcher_skill,
            "park_effect": park_effect, "truth": truth}


def write_seasons(directory, seasons, seed: int = 0):
    """Write several seasons and return the ground truth for each."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out = {}
    for season in seasons:
        out[season] = write_season(directory / f"GL{season}.TXT", season, seed=seed)
    return out
