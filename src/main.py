"""Run the pipeline end to end and write RESULTS.md.

    python -m src.main                  # real Retrosheet data, downloads on first run
    python -m src.main --synthetic      # synthetic fixture, no network needed
    python -m src.main --holdout 2023   # develop on earlier seasons, score 2023 once

Every number quoted in README.md is produced here. Nothing is transcribed by
hand, which is the only way a results table stays true across edits.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA, DEFAULT, FULL_FEATURES, LADDER, Config
from .evaluation import (
    bootstrap_diff, brier, brier_decomposition, brier_diff_test,
    encompassing_test, feature_ladder, fit_isotonic_forward, log_loss,
    reliability_table,
    search_budget_note,
)
from .decision import backtest, sweep_thresholds
from .features import build_panel
from .forecast import BACKEND, permutation_check, walk_forward
from .odds import attach_odds, coverage_report, load_odds
from .retrosheet import load_games
from .shrinkage import compare_closed_form_to_mcmc, implied_talent_sd

warnings.filterwarnings("ignore", category=FutureWarning)


def load_panel(cfg: Config, synthetic: bool = False) -> pd.DataFrame:
    if synthetic:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tests.synthetic import write_seasons
        tmp = DATA / "synthetic"
        write_seasons(tmp, cfg.seasons, seed=1)
        paths = sorted(tmp.glob("GL*.TXT"))
        games = load_games(cfg.seasons, paths=paths)
    else:
        games = load_games(cfg.seasons)
    return build_panel(games, cfg)


def run(cfg: Config, synthetic: bool = False, out: Path | None = None,
        odds_path: Path | None = None) -> dict:
    lines: list[str] = []

    def say(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    say("# Results")
    say()
    say(f"Backend: `{BACKEND}`. Seasons: {', '.join(map(str, cfg.seasons))}.")
    say()

    panel = load_panel(cfg, synthetic=synthetic)
    if cfg.holdout_season is not None:
        panel = panel[panel["season"] <= cfg.holdout_season].reset_index(drop=True)

    y_all = panel["home_win"].to_numpy()
    say(f"Panel: {len(panel):,} games, {panel['season'].nunique()} seasons, "
        f"home win rate {y_all.mean():.4f}.")
    say()

    # --- shrinkage sanity ------------------------------------------------
    say("## Shrinkage")
    say()
    late = panel[panel["date"].dt.month >= 8]
    kappa_late = float(late["kappa"].median())
    say(f"Median fitted concentration in August and later: kappa = {kappa_late:.1f}, "
        f"implying a true-talent standard deviation of {implied_talent_sd(kappa_late):.4f}. "
        f"Real MLB team talent spreads near .045 to .050, so a fitted kappa around 100 "
        f"is the expected neighbourhood.")
    say()
    apr = panel[panel["date"].dt.month == 4]
    say(f"| | raw win pct | shrunk talent |")
    say(f"|---|---|---|")
    say(f"| April spread (sd) | {apr['home_raw_wpct'].std():.4f} | {apr['home_talent'].std():.4f} |")
    say(f"| April range | {apr['home_raw_wpct'].min():.3f} to {apr['home_raw_wpct'].max():.3f} "
        f"| {apr['home_talent'].min():.3f} to {apr['home_talent'].max():.3f} |")
    say()

    mcmc = None
    if not synthetic:
        snapshot = panel[panel["date"] < pd.Timestamp(f"{cfg.seasons[0]}-06-01")]
        if len(snapshot):
            tl = snapshot.groupby("home_team").agg(w=("home_win", "sum"), n=("home_win", "count"))
            mcmc = compare_closed_form_to_mcmc(tl["w"].to_numpy(), tl["n"].to_numpy())
    if mcmc is not None:
        say(f"Closed form against MCMC on the same records: max absolute difference "
            f"{mcmc['max_abs_diff']:.4f}, correlation {mcmc['correlation']:.4f}.")
    else:
        say("PyMC not installed, so the MCMC cross-check was skipped. "
            "Install pymc and rerun to populate this line.")
    say()

    # --- leakage ---------------------------------------------------------
    say("## Leakage check")
    say()
    pc = permutation_check(panel, FULL_FEATURES, cfg, model=cfg.model)
    verdict = "PASS" if pc["brier"] > pc["expected"] - 0.004 else "FAIL"
    say(f"With the target shuffled, out-of-sample Brier is {pc['brier']:.4f} against an "
        f"expected {pc['expected']:.4f}. **{verdict}.** A score meaningfully below the "
        f"expectation would mean a feature carries information about its own row.")
    say()

    # --- ladder ----------------------------------------------------------
    say("## Feature ladder")
    say()
    results = {}
    for model_name in ("gbm", "logistic"):
        say(f"### {model_name}")
        say()
        mcfg = replace(cfg, model=model_name)
        table = feature_ladder(
            panel, "home_win", LADDER,
            lambda d, f, t: walk_forward(d, f, mcfg, target=t, model=model_name),
        )
        results[model_name] = table
        say("| model | features | Brier | vs base | t | vs previous |")
        say("|---|---:|---:|---:|---:|---:|")
        for _, r in table.iterrows():
            t_txt = "" if np.isnan(r["t_vs_base"]) else f"{r['t_vs_base']:.2f}"
            prev = "" if np.isnan(r["vs_previous"]) else f"{r['vs_previous']:+.5f}"
            say(f"| {r['model']} | {int(r['n_features'])} | {r['brier']:.4f} | "
                f"{r['vs_base']:+.5f} | {t_txt} | {prev} |")
        say()

    # --- headline --------------------------------------------------------
    # Report the best rung, not the last one. Adding features monotonically is
    # an assumption, not a result, and on this data it is false: every block
    # after shrinkage costs accuracy.
    best_model = min(results, key=lambda m: results[m]["brier"].min())
    table = results[best_model]
    best_label = table.loc[int(table["brier"].idxmin()), "model"]
    best_features = dict(LADDER).get(best_label, FULL_FEATURES)
    say(f"Best rung: {best_label} ({len(best_features)} features).")
    say()
    preds = walk_forward(panel, best_features, replace(cfg, model=best_model),
                         model=best_model)
    mask = ~np.isnan(preds)
    y, p = y_all[mask], preds[mask]
    base = np.full(mask.sum(), y.mean())

    stats = brier_diff_test(y, base, p)
    boot = bootstrap_diff(y, base, p)
    decomp = brier_decomposition(y, p)

    say("## Headline")
    say()
    say(f"Best family: **{best_model}**, scored on {mask.sum():,} out-of-sample games.")
    say()
    say("| quantity | value |")
    say("|---|---:|")
    say(f"| base rate Brier | {brier(y, base):.4f} |")
    say(f"| model Brier | {brier(y, p):.4f} |")
    say(f"| improvement | {stats['improvement']:.5f} |")
    say(f"| standard error | {stats['std_error']:.5f} |")
    say(f"| t statistic | {stats['t_stat']:.2f} |")
    say(f"| 95% CI | ({stats['ci95'][0]:.5f}, {stats['ci95'][1]:.5f}) |")
    say(f"| bootstrap 95% CI | ({boot['ci95'][0]:.5f}, {boot['ci95'][1]:.5f}) |")
    say(f"| log loss | {log_loss(y, p):.4f} |")
    say(f"| reliability (lower better) | {decomp['reliability']:.5f} |")
    say(f"| resolution (higher better) | {decomp['resolution']:.5f} |")
    say()

    # --- calibration -----------------------------------------------------
    say("## Calibration")
    say()
    frame = panel.loc[mask, ["date", "home_win"]].copy()
    frame["p"] = p
    tested, _ = fit_isotonic_forward(frame, "p", "home_win", split=cfg.calibration_split)
    before = brier(tested["home_win"], tested["p"])
    after = brier(tested["home_win"], tested["calibrated"])
    say(f"Isotonic fit on the first {cfg.calibration_split:.0%} of out-of-sample rows and "
        f"applied to the remaining {1 - cfg.calibration_split:.0%}: Brier {before:.4f} "
        f"before, {after:.4f} after.")
    say()
    say(reliability_table(tested["home_win"], tested["p"]).to_markdown(index=False))
    say()

    # --- market and decision layer --------------------------------------
    if odds_path is not None and Path(odds_path).exists() and not synthetic:
        say("## Market benchmark")
        say()
        odds = load_odds(odds_path, seasons=cfg.seasons)
        merged = attach_odds(panel, odds)
        cover = coverage_report(merged)
        say(f"Matched a price for {cover['matched']:,} of {cover['games']:,} games "
            f"({cover['match_rate']:.1%}). Mean closing margin {cover['mean_closing_vig']:.4f}.")
        say()

        has = merged["has_odds"].to_numpy() & mask
        ym = y_all[has]
        bm = np.full(has.sum(), ym.mean())
        say("| forecaster | Brier | vs base | t |")
        say("|---|---:|---:|---:|")
        for label, col in (("this model", None),
                           ("market opening line", "open_p_home"),
                           ("market closing line", "close_p_home")):
            pm = preds[has] if col is None else merged.loc[has, col].to_numpy()
            s = brier_diff_test(ym, bm, pm)
            say(f"| {label} | {brier(ym, pm):.4f} | {s['improvement']:+.5f} | {s['t_stat']:.2f} |")
        say()
        room = brier(ym, bm) - brier(ym, merged.loc[has, "close_p_home"].to_numpy())
        got = brier(ym, bm) - brier(ym, preds[has])
        say(f"The closing line beats a constant by {room:.5f}. This model captures "
            f"{got / room:.0%} of that.")
        say()

        say("### Does the model know anything the market does not?")
        say()
        say("A worse Brier score is not the same as no information. Two forecasts "
            "can each be individually inferior and still combine into something "
            "better, if each sees what the other misses. Both go on the log-odds "
            "scale and the outcome is regressed on the pair; if the model adds "
            "nothing its coefficient is zero.")
        say()
        say("| benchmark | market coef | model coef | t on model |")
        say("|---|---:|---:|---:|")
        for label, col in (("opening line", "open_p_home"),
                           ("closing line", "close_p_home")):
            enc = encompassing_test(ym, preds[has], merged.loc[has, col].to_numpy())
            say(f"| {label} | {enc['benchmark_coef']:+.3f} | "
                f"{enc['model_coef']:+.3f} | {enc['model_t']:+.2f} |")
        say()
        say("Standard errors are bootstrapped, because the two regressors are "
            "collinear by construction and asymptotic errors are optimistic there.")
        say()

        say("## Decision layer")
        say()
        frame = merged.loc[has].copy()
        frame["model_p"] = preds[has]
        result = backtest(frame, "model_p")
        say(f"Filled at the best opening price across books, marked against the "
            f"consensus close. {result['n_bets']:,} bets on "
            f"{result['bet_rate']:.0%} of games, hit rate {result['hit_rate']:.4f}.")
        say()
        say("| quantity | value |")
        say("|---|---:|")
        say(f"| ROI on bankroll | {result['roi']:+.2%} |")
        say(f"| ROI per unit staked | {result['roi_per_unit_staked']:+.4f} |")
        say(f"| max drawdown | {result['max_drawdown']:.2%} |")
        say(f"| mean CLV | {result['mean_clv']:+.5f} |")
        say(f"| CLV t statistic | {result['clv_t']:.2f} |")
        say(f"| share of bets with positive CLV | {result['positive_clv_rate']:.1%} |")
        say()
        say("Threshold sweep, so the result can be read against the choice rather "
            "than at one point:")
        say()
        say(sweep_thresholds(frame, "model_p").round(4).to_markdown(index=False))
        say()
    else:
        say("## Market benchmark")
        say()
        say("No odds file supplied, so the market comparison and decision layer were "
            "skipped. They are not simulated in their absence.")
        say()

    say("## Search budget")
    say()
    say(search_budget_note(cfg.n_hyperparameter_configs, cfg.n_feature_sets))
    say()

    if out is not None:
        out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {out}")

    return {"panel": panel, "preds": preds, "results": results, "stats": stats}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="use the generated fixture instead of downloading")
    ap.add_argument("--holdout", type=int, default=None,
                    help="restrict to seasons through this year")
    ap.add_argument("--odds", type=Path, default=DATA / "mlb_odds_dataset.json",
                    help="path to the moneyline JSON; skipped if absent")
    ap.add_argument("--out", type=Path, default=Path("RESULTS.md"))
    args = ap.parse_args()

    cfg = replace(DEFAULT, holdout_season=args.holdout)
    run(cfg, synthetic=args.synthetic, out=args.out, odds_path=args.odds)
