"""Figures.

    python -m src.plots [--synthetic]
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import DEFAULT, FIGURES, FULL_FEATURES, LADDER
from .evaluation import brier, fit_isotonic_forward, reliability_table
from .forecast import walk_forward
from .main import load_panel
from .shrinkage import implied_talent_sd

INK, ACCENT, MUTED, PALE = "#1b1b1b", "#c1442f", "#4a6fa5", "#b9c6d8"


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.set_axisbelow(True)


def shrinkage_figure(panel, cfg=DEFAULT):
    """Raw record against shrunk talent, as the season accumulates.

    This is the picture the whole project is built around: a team twelve games in
    is not what its record says it is.
    """
    tl = panel[["home_team", "date", "season", "home_raw_wpct", "home_talent"]].copy()
    tl["games_in"] = tl.groupby(["season", "home_team"]).cumcount()

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for col, colour, label in [("home_raw_wpct", PALE, "raw win pct"),
                               ("home_talent", ACCENT, "shrunk talent")]:
        band = tl.groupby("games_in")[col].agg(
            lo=lambda s: s.quantile(0.05), hi=lambda s: s.quantile(0.95))
        ax.fill_between(band.index, band["lo"], band["hi"], color=colour,
                        alpha=0.55 if col == "home_raw_wpct" else 0.30, lw=0, label=label)

    ax.axhline(0.5, color=INK, ls=":", lw=1)
    ax.set_xlim(0, tl["games_in"].max() * 0.55)
    ax.set_ylim(0.15, 0.85)
    ax.set_xlabel("home games played this season")
    ax.set_ylabel("estimated win rate")
    ax.set_title("Shrinkage: the 5th to 95th percentile band across teams",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "shrinkage.png", dpi=170)
    plt.close(fig)


def kappa_figure(panel):
    """Fitted concentration over time, with the implied talent spread beside it."""
    daily = panel.groupby("date")["kappa"].median()

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(daily.index, daily.values, color=MUTED, lw=1.4)
    ax.set_ylabel("fitted $\\kappa$ (prior games)")
    ax.set_xlabel("")
    ax.set_title("Concentration refits daily; April holds the fallback until records mean something",
                 fontsize=10.5, loc="left")

    twin = ax.twinx()
    twin.plot(daily.index, [implied_talent_sd(k) for k in daily.values],
              color=ACCENT, lw=1.0, alpha=0.75)
    twin.set_ylabel("implied talent sd", color=ACCENT)
    twin.tick_params(axis="y", colors=ACCENT)
    twin.spines["top"].set_visible(False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "kappa.png", dpi=170)
    plt.close(fig)


def ladder_figure(panel, cfg=DEFAULT):
    """Brier by feature block, both model families, against the base rate."""
    y = panel["home_win"].to_numpy()
    base_rate = y.mean()

    series = {}
    for model in ("gbm", "logistic"):
        mcfg = replace(cfg, model=model)
        scores = []
        for _, feats in LADDER:
            p = walk_forward(panel, feats, mcfg, model=model)
            m = ~np.isnan(p)
            scores.append(brier(y[m], p[m]))
        series[model] = scores

    labels = [name for name, _ in LADDER]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - width / 2, series["gbm"], width, color=MUTED, label="gradient boosting")
    ax.bar(x + width / 2, series["logistic"], width, color=ACCENT, label="regularized logistic")
    ax.axhline(base_rate * (1 - base_rate), color=INK, ls="--", lw=1.2,
               label=f"constant base rate ({base_rate:.3f})")

    lo = min(min(series["gbm"]), min(series["logistic"]))
    ax.set_ylim(lo - 0.0015, base_rate * (1 - base_rate) + 0.0012)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("out-of-sample Brier (lower is better)")
    ax.set_title("Every rung on one identical walk-forward split", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "ladder.png", dpi=170)
    plt.close(fig)


def calibration_figure(panel, cfg=DEFAULT):
    """Reliability before and after isotonic, with bin counts drawn as marker size.

    Sizing the markers by count is the point. A bin holding nine games sitting far
    off the diagonal is noise, and a curve that hides the counts invites reading
    it as miscalibration.
    """
    y = panel["home_win"].to_numpy()
    scored = {}
    for model in ("gbm", "logistic"):
        pred = walk_forward(panel, FULL_FEATURES, replace(cfg, model=model), model=model)
        m = ~np.isnan(pred)
        scored[model] = (brier(y[m], pred[m]), pred)
    best = min(scored, key=lambda m: scored[m][0])
    p = scored[best][1]
    mask = ~np.isnan(p)
    frame = panel.loc[mask, ["date", "home_win"]].copy()
    frame["p"] = p[mask]
    tested, _ = fit_isotonic_forward(frame, "p", "home_win", split=cfg.calibration_split)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0.35, 0.70], [0.35, 0.70], ls="--", color=INK, lw=1, label="perfect calibration")

    for col, colour, label in [("p", MUTED, f"{best}, raw"),
                               ("calibrated", ACCENT, "after isotonic")]:
        tab = reliability_table(tested["home_win"], tested[col], bins=8)
        ax.plot(tab["mean_forecast"], tab["observed"], color=colour, lw=1.2, alpha=0.8)
        ax.scatter(tab["mean_forecast"], tab["observed"], s=tab["n"] / 3.0,
                   color=colour, alpha=0.75, label=label, zorder=4)

    ax.set_xlabel("mean forecast")
    ax.set_ylabel("observed frequency")
    ax.set_title("Reliability, marker area proportional to games in bin",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(FIGURES / "calibration.png", dpi=170)
    plt.close(fig)


def main(synthetic: bool = False):
    FIGURES.mkdir(exist_ok=True)
    panel = load_panel(DEFAULT, synthetic=synthetic)
    shrinkage_figure(panel)
    kappa_figure(panel)
    ladder_figure(panel)
    calibration_figure(panel)
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(**vars(ap.parse_args()))
