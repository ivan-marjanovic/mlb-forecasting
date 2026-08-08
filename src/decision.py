"""The decision layer: edge, sizing, and closing line value.

This exists only because real prices exist. An earlier version of this project
generated synthetic odds by perturbing the model's own forecast, which makes the
edge a function of the noise draw and the resulting return a measurement of the
random seed. That version returned a 41.5% simulated profit and meant nothing. If
the odds file is ever unavailable, the correct move is to skip this module, not
to manufacture a market.

Two quantities are reported and they answer different questions. Profit answers
"what happened", and over a few hundred bets it is almost entirely variance.
Closing line value answers "was the price good when it was taken", and because it
scores every bet against a benchmark rather than against a coin flip, it converges
far faster. A strategy with positive CLV and negative profit is usually unlucky.
A strategy with negative CLV and positive profit is usually lucky, and saying so
is the difference between a backtest and a story.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(prob, decimal_odds):
    """Full Kelly stake as a fraction of bankroll, floored at zero.

    f = (p*b - q) / b with b the net decimal payout. Full Kelly maximises long-run
    log growth and is far too volatile to run: it assumes the probability estimate
    is exactly right, and a modest overestimate produces severe drawdowns. The
    fraction is scaled down by the caller.
    """
    p = np.asarray(prob, dtype=float)
    b = np.asarray(decimal_odds, dtype=float) - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        f = (p * b - (1.0 - p)) / b
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


def backtest(df: pd.DataFrame, prob_col: str, *, kelly_scale: float = 0.25,
             edge_threshold: float = 0.02, max_stake: float = 0.02,
             bankroll: float = 10_000.0, compound: bool = True) -> dict:
    """Simulate the strategy at real prices, filling at the opening line.

    Bets are placed on whichever side the model prefers, at the best opening
    price across books, and marked against the consensus closing probability.
    Filling at the open rather than the close is the honest choice: a model that
    runs in the morning cannot take the closing number, and grading yourself at a
    price you could not have got is the most common way a sports backtest lies.
    """
    df = df.sort_values("date").reset_index(drop=True)

    p_home = df[prob_col].to_numpy(dtype=float)
    p_away = 1.0 - p_home

    edge_home = p_home - df["open_p_home"].to_numpy(dtype=float)
    edge_away = p_away - df["open_p_away"].to_numpy(dtype=float)

    take_home = edge_home >= edge_away
    edge = np.where(take_home, edge_home, edge_away)
    prob = np.where(take_home, p_home, p_away)
    price = np.where(take_home,
                     df["open_best_home_dec"].to_numpy(dtype=float),
                     df["open_best_away_dec"].to_numpy(dtype=float))
    close_p = np.where(take_home,
                       df["close_p_home"].to_numpy(dtype=float),
                       df["close_p_away"].to_numpy(dtype=float))
    won = np.where(take_home,
                   df["home_win"].to_numpy(dtype=int),
                   1 - df["home_win"].to_numpy(dtype=int))

    stake_fraction = np.where(edge > edge_threshold,
                              np.minimum(kelly_scale * kelly_fraction(prob, price), max_stake),
                              0.0)
    bet = stake_fraction > 0

    equity = np.empty(len(df))
    balance = bankroll
    for i in range(len(df)):
        if bet[i]:
            stake = (balance if compound else bankroll) * stake_fraction[i]
            balance += stake * (price[i] - 1.0) if won[i] else -stake
        equity[i] = balance

    # closing line value: the price taken against the price at the close
    implied_at_fill = 1.0 / price
    clv = close_p[bet] - implied_at_fill[bet]

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak

    n_bets = int(bet.sum())
    roi = (equity[-1] - bankroll) / bankroll if len(equity) else 0.0
    turnover = float(stake_fraction[bet].sum()) if n_bets else 0.0

    return {
        "n_games": len(df),
        "n_bets": n_bets,
        "bet_rate": n_bets / len(df) if len(df) else 0.0,
        "hit_rate": float(won[bet].mean()) if n_bets else np.nan,
        "final_bankroll": float(equity[-1]) if len(equity) else bankroll,
        "roi": float(roi),
        "roi_per_unit_staked": float((equity[-1] - bankroll) / (bankroll * turnover))
        if turnover > 0 else np.nan,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "mean_clv": float(clv.mean()) if n_bets else np.nan,
        "clv_std_error": float(clv.std(ddof=1) / np.sqrt(n_bets)) if n_bets > 1 else np.nan,
        "clv_t": float(clv.mean() / (clv.std(ddof=1) / np.sqrt(n_bets)))
        if n_bets > 1 and clv.std(ddof=1) > 0 else np.nan,
        "positive_clv_rate": float((clv > 0).mean()) if n_bets else np.nan,
        "mean_edge_taken": float(edge[bet].mean()) if n_bets else np.nan,
        "equity": equity,
        "bet_mask": bet,
    }


def sweep_thresholds(df: pd.DataFrame, prob_col: str, thresholds=None, **kwargs):
    """Run the backtest across edge thresholds.

    A strategy that only works at one threshold is fitted to the sample. The
    shape of this curve is more informative than any single row of it, and a
    reader should be able to see whether the result survives the choice.
    """
    thresholds = np.arange(0.00, 0.11, 0.01) if thresholds is None else thresholds
    out = []
    for t in thresholds:
        r = backtest(df, prob_col, edge_threshold=float(t), **kwargs)
        out.append({
            "threshold": float(t), "n_bets": r["n_bets"], "hit_rate": r["hit_rate"],
            "roi": r["roi"], "roi_per_unit": r["roi_per_unit_staked"],
            "mean_clv": r["mean_clv"], "clv_t": r["clv_t"],
            "max_drawdown": r["max_drawdown"],
        })
    return pd.DataFrame(out)


def risk_metrics(equity, bet_mask, bankroll: float = 10_000.0, periods_per_year: int = 2430):
    """Risk-adjusted summary of the simulated bankroll.

    Reported because trading desks speak this language, and with a caveat because
    the language does not fit perfectly. Quarter-Kelly staking on binary outcomes
    produces returns that are lumpy, skewed and fat-tailed, so a Sharpe ratio
    computed on them leans on a normality assumption the data does not satisfy.
    Sortino, which penalises only downside deviation, is the better of the two
    here. Maximum drawdown needs no distributional assumption at all and is the
    number to trust if they disagree.

    `periods_per_year` is set to a full MLB season of games so the annualisation
    is interpretable as per-season rather than per-trading-day.
    """
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return {}

    curve = np.concatenate([[bankroll], equity])
    returns = np.diff(curve) / curve[:-1]
    active = returns[np.asarray(bet_mask, dtype=bool)]
    if len(active) < 2:
        return {}

    mean, sd = active.mean(), active.std(ddof=1)
    downside = active[active < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else np.nan

    peak = np.maximum.accumulate(curve)
    drawdown = (curve - peak) / peak

    return {
        "sharpe_annualised": float(mean / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan,
        "sortino_annualised": float(mean / dsd * np.sqrt(periods_per_year))
        if dsd and dsd > 0 else np.nan,
        "mean_return_per_bet": float(mean),
        "volatility_per_bet": float(sd),
        "skew": float(((active - mean) ** 3).mean() / sd ** 3) if sd > 0 else np.nan,
        "excess_kurtosis": float(((active - mean) ** 4).mean() / sd ** 4 - 3.0)
        if sd > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "longest_losing_streak": int(_longest_run(active < 0)),
    }


def _longest_run(flags) -> int:
    best = run = 0
    for flag in np.asarray(flags, dtype=bool):
        run = run + 1 if flag else 0
        best = max(best, run)
    return best
