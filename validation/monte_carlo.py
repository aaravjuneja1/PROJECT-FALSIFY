"""
Monte Carlo validation for FALSIFY.

Tests whether backtest performance is explained by lucky trade sequencing.
Two simulation methods run in parallel:
  Method A (Reshuffle): permutes trade order without replacement
  Method B (Resample):  bootstraps trade returns with replacement

Entry point: run_monte_carlo(trade_log, config) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # non-interactive, must be before pyplot
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.stats import percentileofscore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.metrics_config import TRADING_DAYS_PER_YEAR
from metrics.calculator import avg_holding_trading_days, per_trade_sharpe


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(trade_log: pd.DataFrame, config: dict) -> dict:
    """
    Run Monte Carlo simulation on OOS trade returns.

    Parameters
    ----------
    trade_log : pd.DataFrame
        Stitched OOS trade log from engine/wfa.py.
        Required columns: entry_date, exit_date, return_pct.
        return_pct is decimal (0.05 = 5% gain, -0.02 = 2% loss).

    config : dict
        Keys used:
          monte_carlo.n_simulations   (default 5000)
          monte_carlo.ruin_threshold  (default 0.30 = 30% drawdown)
          monte_carlo.random_seed     (default 42)
        Per-trade Sharpe annualisation uses TRADING_DAYS_PER_YEAR from
        configs/metrics_config.py (the single source of truth).

    Returns
    -------
    dict with keys: reshuffle, resample, shared  (see bottom of function)
    """
    # ── Extract config ─────────────────────────────────────────────────────────
    mc_cfg         = config.get("monte_carlo", {})
    n_sims         = int(mc_cfg.get("n_simulations", 5000))
    ruin_threshold = float(mc_cfg.get("ruin_threshold", 0.30))
    seed           = int(mc_cfg.get("random_seed", 42))

    # ── Guard checks ──────────────────────────────────────────────────────────
    n_trades    = len(trade_log)
    low_sample  = False

    if n_trades < 10:
        raise ValueError(
            f"Insufficient trades for Monte Carlo ({n_trades} < 10). Aborting."
        )

    if n_trades < 30:
        low_sample = True
        print(
            f"\n⚠ WARNING: Monte Carlo unreliable: only {n_trades} trades in OOS "
            "log. Minimum 30 required. Results are indicative only."
        )

    # ── Set global random seed for reproducibility ────────────────────────────
    np.random.seed(seed)

    # ── Extract returns (decimal) and average holding duration ────────────────
    # Holding period and per-trade Sharpe come from the shared calculator.py
    # helpers (trading days via np.busday_count, ddof=1) so MC and stats agree.
    returns          = trade_log["return_pct"].values.astype(float)
    avg_holding_days = avg_holding_trading_days(trade_log)

    # ── Backtest (original sequence) baseline stats ───────────────────────────
    original_equity  = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    original_dd      = _max_dd_single(original_equity) * 100.0   # → positive %
    original_ret     = (original_equity[-1] - 1.0) * 100.0        # → %
    original_sharpe  = per_trade_sharpe(returns, avg_holding_days)

    # ── Method A: Reshuffle (permutation, without replacement) ────────────────
    # np.argsort on random keys is the fastest vectorised permutation approach
    rand_keys_A      = np.random.rand(n_sims, n_trades)
    shuffled_idx     = np.argsort(rand_keys_A, axis=1)            # (n_sims, n_trades)
    shuffled_returns = returns[shuffled_idx]                       # (n_sims, n_trades)
    equity_A         = _build_equity_curves(shuffled_returns)      # (n_sims, n_trades+1)

    metrics_A = _compute_method_metrics(
        equity_A, shuffled_returns, n_sims,
        avg_holding_days, ruin_threshold,
        original_dd, original_ret, original_sharpe,
    )

    # ── Method B: Resample (bootstrap, with replacement) ──────────────────────
    resampled_returns = np.random.choice(returns, size=(n_sims, n_trades), replace=True)
    equity_B          = _build_equity_curves(resampled_returns)    # (n_sims, n_trades+1)

    metrics_B = _compute_method_metrics(
        equity_B, resampled_returns, n_sims,
        avg_holding_days, ruin_threshold,
        original_dd, original_ret, original_sharpe,
    )

    # ── Verdicts ──────────────────────────────────────────────────────────────
    verdict_A, detail_A = _get_reshuffle_verdict(metrics_A["ror_pct"])
    verdict_B, detail_B = _get_resample_verdict(
        metrics_B["sharpe_p5"], metrics_B["ror_pct"]
    )

    dd_warning = (
        metrics_A["dd_inflation_factor"] > 2.0
        or metrics_B["dd_inflation_factor"] > 2.0
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_dir = Path(__file__).resolve().parents[1] / config["reporting"]["charts_dir"]
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = _generate_all_charts(
        equity_A, equity_B, original_equity,
        metrics_A, metrics_B,
        original_dd, original_sharpe,
        verdict_A, verdict_B,
        charts_dir,
    )

    # ── Console summary ───────────────────────────────────────────────────────
    _print_summary(
        metrics_A, metrics_B,
        n_trades, n_sims, ruin_threshold,
        verdict_A, verdict_B,
        dd_warning, low_sample,
    )

    # ── Build and return result dict ──────────────────────────────────────────
    def _pack(m: dict, verdict: str, detail: str) -> dict:
        return {
            "final_returns":          m["final_returns"],
            "drawdowns":              m["drawdowns"],
            "sharpes":                m["sharpes"],
            "ror_pct":                m["ror_pct"],
            "dd_p5":                  m["dd_p5"],
            "dd_p50":                 m["dd_p50"],
            "dd_p95":                 m["dd_p95"],
            "dd_backtest":            m["dd_backtest"],
            "dd_inflation_factor":    m["dd_inflation_factor"],
            "ret_p5":                 m["ret_p5"],
            "ret_p50":                m["ret_p50"],
            "ret_p95":                m["ret_p95"],
            "ret_backtest":           m["ret_backtest"],
            "ret_ci_90":              m["ret_ci_90"],
            "sharpe_p5":              m["sharpe_p5"],
            "sharpe_p50":             m["sharpe_p50"],
            "sharpe_p95":             m["sharpe_p95"],
            "sharpe_backtest":        m["sharpe_backtest"],
            "sharpe_percentile_rank": m["sharpe_percentile_rank"],
            "verdict":                verdict,
            "verdict_detail":         detail,
        }

    return {
        "reshuffle": _pack(metrics_A, verdict_A, detail_A),
        "resample":  _pack(metrics_B, verdict_B, detail_B),
        "shared": {
            "n_trades":           n_trades,
            "n_simulations":      n_sims,
            "ruin_threshold":     ruin_threshold,
            "low_sample_warning": low_sample,
            "dd_warning":         dd_warning,
            "chart_paths":        chart_paths,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _max_dd_single(equity: np.ndarray) -> float:
    """
    Max peak-to-trough drawdown for a single equity curve.
    Returns positive fraction (e.g. 0.15 for 15% drawdown).
    """
    if len(equity) < 2:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    # Guard against zero-valued running_max (shouldn't happen but be safe)
    safe_max    = np.where(running_max > 0, running_max, 1.0)
    drawdowns   = (running_max - equity) / safe_max
    return float(np.max(drawdowns))


def _build_equity_curves(returns_matrix: np.ndarray) -> np.ndarray:
    """
    Build equity curves starting at 1.0 for all simulations simultaneously.

    returns_matrix : (n_sims, n_trades) — decimal returns, already shuffled/resampled
    Returns        : (n_sims, n_trades+1) — each row starts at 1.0
    """
    n_sims, _ = returns_matrix.shape
    cum_prod   = np.cumprod(1.0 + returns_matrix, axis=1)     # (n_sims, n_trades)
    return np.hstack([np.ones((n_sims, 1)), cum_prod])         # (n_sims, n_trades+1)


def _compute_method_metrics(
    equity_curves:    np.ndarray,
    returns_matrix:   np.ndarray,
    n_sims:           int,
    avg_holding_days: float,
    ruin_threshold:   float,
    original_dd:      float,
    original_ret:     float,
    original_sharpe:  float,
) -> dict:
    """
    Compute all output metrics for one simulation method (reshuffle or resample).
    All drawdown and return values are in positive percentages (e.g. 15.3 for 15.3%).
    """
    # ── Drawdowns (vectorised) ─────────────────────────────────────────────────
    running_max   = np.maximum.accumulate(equity_curves, axis=1)
    safe_max      = np.where(running_max > 0, running_max, 1.0)
    dd_matrix     = (running_max - equity_curves) / safe_max
    max_drawdowns = np.max(dd_matrix, axis=1) * 100.0          # (n_sims,) positive %

    # ── Final returns (percentage) ─────────────────────────────────────────────
    final_returns = (equity_curves[:, -1] - 1.0) * 100.0       # (n_sims,)

    # ── Per-trade Sharpe (vectorised) ─────────────────────────────────────────
    means  = np.mean(returns_matrix, axis=1)
    stds   = np.std(returns_matrix, axis=1, ddof=1)               # ddof=1 (matches per_trade_sharpe)
    factor = np.sqrt(TRADING_DAYS_PER_YEAR / max(avg_holding_days, 1.0))
    sharpes = np.where(stds == 0.0, 0.0, (means / stds) * factor)  # (n_sims,)

    # ── Ruin detection ─────────────────────────────────────────────────────────
    ruin_flags = np.max(dd_matrix, axis=1) > ruin_threshold    # ruin_threshold is 0.30 etc.
    ror_pct    = float(ruin_flags.sum() / n_sims * 100.0)

    # ── Percentile metrics ─────────────────────────────────────────────────────
    dd_p5,  dd_p50,  dd_p95  = (float(np.percentile(max_drawdowns, p)) for p in (5, 50, 95))
    ret_p5, ret_p50, ret_p95 = (float(np.percentile(final_returns,  p)) for p in (5, 50, 95))
    sh_p5,  sh_p50,  sh_p95  = (float(np.percentile(sharpes,         p)) for p in (5, 50, 95))

    # dd_p95 / dd_backtest tells you how much worse the tail can be vs what you saw
    dd_inflation = (dd_p95 / original_dd) if original_dd > 0.0 else 0.0

    # percentileofscore with kind='rank' returns ~50 when all simulated values
    # are equal to the test value (Method A reshuffle case) — correct behaviour.
    sharpe_rank = float(
        percentileofscore(sharpes, original_sharpe, kind="rank")
    )

    return {
        "final_returns":          final_returns,
        "drawdowns":              max_drawdowns,
        "sharpes":                sharpes,
        "ror_pct":                ror_pct,
        "dd_p5":                  dd_p5,
        "dd_p50":                 dd_p50,
        "dd_p95":                 dd_p95,
        "dd_backtest":            original_dd,
        "dd_inflation_factor":    float(dd_inflation),
        "ret_p5":                 ret_p5,
        "ret_p50":                ret_p50,
        "ret_p95":                ret_p95,
        "ret_backtest":           original_ret,
        "ret_ci_90":              (ret_p5, ret_p95),
        "sharpe_p5":              sh_p5,
        "sharpe_p50":             sh_p50,
        "sharpe_p95":             sh_p95,
        "sharpe_backtest":        original_sharpe,
        "sharpe_percentile_rank": sharpe_rank,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verdict logic
# ──────────────────────────────────────────────────────────────────────────────

def _get_reshuffle_verdict(ror_pct: float) -> tuple[str, str]:
    """
    Reshuffle verdict based on Risk of Ruin only.
    Sharpe rank is not meaningful for reshuffle: permutation preserves
    mean/std so rank is always ~50 regardless of strategy quality.
    """
    if ror_pct <= 5.0:
        return (
            "ROBUST",
            "Risk of ruin is very low across all trade orderings. "
            "Strategy survives adverse sequences well.",
        )
    elif ror_pct <= 15.0:
        return (
            "MARGINAL",
            "Moderate ruin risk. Strategy is sensitive to trade sequencing.",
        )
    else:
        return (
            "FRAGILE",
            "High ruin risk. Strategy cannot survive adverse trade sequences.",
        )


def _get_resample_verdict(sharpe_p5: float, ror_pct: float) -> tuple[str, str]:
    """
    Resample verdict based on whether edge persists in the worst 5% of
    bootstrap samples.
    """
    if sharpe_p5 > 0 and ror_pct <= 10.0:
        return (
            "ROBUST",
            "Edge persists even in worst 5% of bootstrap samples. Low ruin risk.",
        )
    elif sharpe_p5 > 0 or ror_pct <= 15.0:
        return (
            "MARGINAL",
            "Edge is inconsistent across bootstrap samples or ruin risk is moderate.",
        )
    else:
        return (
            "LUCKY",
            "Bootstrap shows edge disappears in adverse samples. "
            "Result may be driven by a few outlier trades.",
        )


# ──────────────────────────────────────────────────────────────────────────────
# Console print
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(
    mA:            dict,
    mB:            dict,
    n_trades:      int,
    n_sims:        int,
    ruin_threshold: float,
    verdict_A:     str,
    verdict_B:     str,
    dd_warning:    bool,
    low_sample:    bool,
) -> None:
    SEP  = "═" * 50
    THIN = "─" * 50
    W    = 30   # left label width
    C    = 12   # column width

    def pct(v: float) -> str:
        return f"{v:.1f}%"

    def dec2(v: float) -> str:
        return f"{v:.2f}"

    def factor(v: float) -> str:
        return f"{v:.1f}x"

    print(f"\n{SEP}")
    print("MONTE CARLO RESULTS — FALSIFY")
    print(SEP)
    print(f"Simulations:  {n_sims:,} x 2 methods  |  Trades: {n_trades}")
    print(f"Ruin Threshold: {ruin_threshold * 100:.0f}%")
    print(THIN)
    print(f"{'':>{W}}{'RESHUFFLE':>{C}}{'RESAMPLE':>{C}}")
    print(f"{'Max DD  — Backtest:':<{W}}{pct(mA['dd_backtest']):>{C}}{pct(mB['dd_backtest']):>{C}}")
    print(f"{'Max DD  — P50:':<{W}}{pct(mA['dd_p50']):>{C}}{pct(mB['dd_p50']):>{C}}")
    print(f"{'Max DD  — P95:':<{W}}{pct(mA['dd_p95']):>{C}}{pct(mB['dd_p95']):>{C}}")
    print(f"{'DD Inflation Factor:':<{W}}{factor(mA['dd_inflation_factor']):>{C}}{factor(mB['dd_inflation_factor']):>{C}}")
    print()
    print(f"{'Return  — Backtest:':<{W}}{pct(mA['ret_backtest']):>{C}}{pct(mB['ret_backtest']):>{C}}")
    print(f"{'Return  — P50:':<{W}}{pct(mA['ret_p50']):>{C}}{pct(mB['ret_p50']):>{C}}")
    ci_A = f"[{mA['ret_ci_90'][0]:.1f}%, {mA['ret_ci_90'][1]:.1f}%]"
    ci_B = f"[{mB['ret_ci_90'][0]:.1f}%, {mB['ret_ci_90'][1]:.1f}%]"
    print(f"{'90% CI:':<{W}}{ci_A:>{C}}  {ci_B}")
    print()
    print(f"{'Sharpe  — Backtest:':<{W}}{dec2(mA['sharpe_backtest']):>{C}}{dec2(mB['sharpe_backtest']):>{C}}")
    print(f"{'Sharpe  — P50:':<{W}}{dec2(mA['sharpe_p50']):>{C}}{dec2(mB['sharpe_p50']):>{C}}")
    rank_A = f"{mA['sharpe_percentile_rank']:.0f}th"
    rank_B = f"{mB['sharpe_percentile_rank']:.0f}th"
    print(f"{'Sharpe Percentile Rank:':<{W}}{rank_A:>{C}}{rank_B:>{C}}")
    print()
    print(f"{'Risk of Ruin:':<{W}}{pct(mA['ror_pct']):>{C}}{pct(mB['ror_pct']):>{C}}")
    print(THIN)
    print(f"VERDICT (Reshuffle):  {verdict_A}")
    print(f"VERDICT (Resample):   {verdict_B}")
    print(THIN)
    if dd_warning:
        print("⚠ DD WARNING: 95th pct drawdown > 2x backtest")
    if low_sample:
        print("⚠ LOW SAMPLE WARNING: N < 30 trades")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Chart generation — all 5 charts
# ──────────────────────────────────────────────────────────────────────────────

_COLOR_A  = "#4A90D9"   # steel blue  — Method A Reshuffle
_COLOR_B  = "#E8735A"   # coral       — Method B Resample
_COLOR_OG = "#FFFFFF"   # white       — original backtest

def _generate_all_charts(
    equity_A:       np.ndarray,
    equity_B:       np.ndarray,
    original_equity: np.ndarray,
    metrics_A:      dict,
    metrics_B:      dict,
    original_dd:    float,
    original_sharpe: float,
    verdict_A:      str,
    verdict_B:      str,
    charts_dir:     Path,
) -> dict:
    x_trades = np.arange(equity_A.shape[1])
    paths: dict = {}

    # Chart 1: Equity fan — Reshuffle
    p1 = charts_dir / "mc_equity_fan_reshuffle.png"
    _chart_equity_fan(
        equity_A, original_equity, x_trades,
        _COLOR_A,
        "Monte Carlo — Reshuffle: Equity Curve Distribution",
        p1,
    )
    paths["equity_fan_reshuffle"] = str(p1)

    # Chart 2: Equity fan — Resample
    p2 = charts_dir / "mc_equity_fan_resample.png"
    _chart_equity_fan(
        equity_B, original_equity, x_trades,
        _COLOR_B,
        "Monte Carlo — Resample: Equity Curve Distribution",
        p2,
    )
    paths["equity_fan_resample"] = str(p2)

    # Chart 3: Max drawdown distribution
    p3 = charts_dir / "mc_drawdown_distribution.png"
    _chart_dd_distribution(
        metrics_A["drawdowns"], metrics_B["drawdowns"],
        original_dd,
        metrics_A["dd_p95"], metrics_B["dd_p95"],
        p3,
    )
    paths["drawdown_distribution"] = str(p3)

    # Chart 4: Sharpe distribution
    p4 = charts_dir / "mc_sharpe_distribution.png"
    _chart_sharpe_distribution(
        metrics_A["sharpes"], metrics_B["sharpes"],
        original_sharpe,
        metrics_A["sharpe_percentile_rank"],
        metrics_B["sharpe_percentile_rank"],
        p4,
    )
    paths["sharpe_distribution"] = str(p4)

    # Chart 5: Summary dashboard
    p5 = charts_dir / "mc_summary_dashboard.png"
    _chart_summary_dashboard(metrics_A, metrics_B, verdict_A, verdict_B, p5)
    paths["summary_dashboard"] = str(p5)

    return paths


def _chart_equity_fan(
    equity_curves:   np.ndarray,
    original_equity: np.ndarray,
    x_trades:        np.ndarray,
    color:           str,
    title:           str,
    save_path:       Path,
) -> None:
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(12, 7))

        # Fan using LineCollection — far faster than N individual plot() calls.
        # segments shape: (n_sims, n_points, 2)
        n_sims   = equity_curves.shape[0]
        x_broad  = np.broadcast_to(x_trades, (n_sims, len(x_trades)))  # (n_sims, n_points)
        segments = np.stack([x_broad, equity_curves], axis=2)            # (n_sims, n_points, 2)
        lc = LineCollection(segments, color=color, alpha=0.02, linewidth=0.5)
        ax.add_collection(lc)

        # Percentile curves
        p5  = np.percentile(equity_curves,  5, axis=0)
        p50 = np.percentile(equity_curves, 50, axis=0)
        p95 = np.percentile(equity_curves, 95, axis=0)

        ax.fill_between(x_trades, p5, p95, color=color, alpha=0.15)
        ax.plot(x_trades, p95, color=color, alpha=0.5, linewidth=1.5, linestyle="--", label="P95")
        ax.plot(x_trades, p50, color=color, alpha=0.5, linewidth=2.0, label="P50")
        ax.plot(x_trades, p5,  color=color, alpha=0.5, linewidth=1.5, linestyle="--", label="P5")

        # Original equity curve
        ax.plot(x_trades, original_equity, color=_COLOR_OG, linewidth=2.0,
                label="Original", zorder=5)

        # Set axis limits explicitly (LineCollection doesn't trigger autoscale)
        y_min = float(equity_curves.min())
        y_max = float(equity_curves.max())
        margin = (y_max - y_min) * 0.05 if y_max != y_min else 0.1
        ax.set_xlim(0, len(x_trades) - 1)
        ax.set_ylim(y_min - margin, y_max + margin)

        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Equity (start = 1.0)")
        ax.set_title(title, fontsize=13, pad=12)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_dd_distribution(
    dd_A:         np.ndarray,
    dd_B:         np.ndarray,
    dd_backtest:  float,
    dd_p95_A:     float,
    dd_p95_B:     float,
    save_path:    Path,
) -> None:
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 6))

        all_dd   = np.concatenate([dd_A, dd_B])
        bin_max  = max(float(np.percentile(all_dd, 99)) * 1.05, 1.0)   # guard zero-drawdown edge case
        bins     = np.linspace(0.0, bin_max, 50)

        counts_A, _ = np.histogram(dd_A, bins=bins)
        counts_B, _ = np.histogram(dd_B, bins=bins)
        y_max       = max(int(max(counts_A.max(), counts_B.max())) * 1.15, 1.0)

        ax.hist(dd_A, bins=bins, color=_COLOR_A, alpha=0.6, label="Reshuffle")
        ax.hist(dd_B, bins=bins, color=_COLOR_B, alpha=0.6, label="Resample")

        ax.axvline(dd_backtest, color=_COLOR_OG, linestyle="--", linewidth=1.5,
                   label=f"Backtest: {dd_backtest:.1f}%")

        ax.axvline(dd_p95_A, color=_COLOR_A, linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"Reshuffle P95: {dd_p95_A:.1f}%")
        ax.axvline(dd_p95_B, color=_COLOR_B, linestyle=":", linewidth=1.5, alpha=0.8,
                   label=f"Resample P95: {dd_p95_B:.1f}%")

        # Annotate P95 values on the chart
        ax.text(dd_p95_A + 0.3, y_max * 0.88, f"{dd_p95_A:.1f}%",
                color=_COLOR_A, fontsize=9, va="top")
        ax.text(dd_p95_B + 0.3, y_max * 0.72, f"{dd_p95_B:.1f}%",
                color=_COLOR_B, fontsize=9, va="top")

        ax.set_xlim(left=0.0)
        ax.set_ylim(0, y_max)
        ax.set_xlabel("Max Drawdown (%)")
        ax.set_ylabel("Frequency")
        ax.set_title("Max Drawdown Distribution — Reshuffle vs Resample", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_sharpe_distribution(
    sharpes_A:   np.ndarray,
    sharpes_B:   np.ndarray,
    sh_backtest: float,
    rank_A:      float,
    rank_B:      float,
    save_path:   Path,
) -> None:
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 6))

        all_sh  = np.concatenate([sharpes_A, sharpes_B])
        sh_min  = float(np.percentile(all_sh, 1))
        sh_max  = float(np.percentile(all_sh, 99))
        # Method A reshuffle produces all-identical Sharpes (permutation-invariant).
        # Guard against degenerate range so linspace and histograms don't crash.
        if sh_min >= sh_max:
            padding = max(abs(sh_backtest) * 0.5, 1.0)
            sh_min  = sh_backtest - padding
            sh_max  = sh_backtest + padding
        bins    = np.linspace(sh_min, sh_max, 50)

        counts_A, _ = np.histogram(sharpes_A, bins=bins)
        counts_B, _ = np.histogram(sharpes_B, bins=bins)
        y_max       = max(int(max(counts_A.max(), counts_B.max())) * 1.20, 1.0)

        ax.hist(sharpes_A, bins=bins, color=_COLOR_A, alpha=0.6, label="Reshuffle")
        ax.hist(sharpes_B, bins=bins, color=_COLOR_B, alpha=0.6, label="Resample")

        ax.axvline(sh_backtest, color=_COLOR_OG, linestyle="--", linewidth=2.0,
                   label=f"Backtest: {sh_backtest:.2f}")

        # Annotate percentile ranks near the vertical line
        x_offset = (sh_max - sh_min) * 0.02
        ax.text(sh_backtest + x_offset, y_max * 0.90,
                f"Reshuffle: {rank_A:.0f}th pct",
                color=_COLOR_A, fontsize=9, va="top")
        ax.text(sh_backtest + x_offset, y_max * 0.74,
                f"Resample: {rank_B:.0f}th pct",
                color=_COLOR_B, fontsize=9, va="top")

        ax.set_ylim(0, y_max)
        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Frequency")
        ax.set_title("Sharpe Ratio Distribution — Where Does Your Result Sit?", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_summary_dashboard(
    mA:        dict,
    mB:        dict,
    verdict_A: str,
    verdict_B: str,
    save_path: Path,
) -> None:
    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(14, 10))
        gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

        ax_tl = fig.add_subplot(gs[0, 0])
        ax_tr = fig.add_subplot(gs[0, 1])
        ax_bl = fig.add_subplot(gs[1, 0])
        ax_br = fig.add_subplot(gs[1, 1])

        # ── Top-left: Method A final returns ──────────────────────────────────
        _subplot_return_dist(
            ax_tl,
            mA["final_returns"], mA["ret_p5"], mA["ret_p50"], mA["ret_p95"],
            mA["ret_backtest"], _COLOR_A, "Reshuffle: Final Returns",
        )

        # ── Top-right: Method B final returns ─────────────────────────────────
        _subplot_return_dist(
            ax_tr,
            mB["final_returns"], mB["ret_p5"], mB["ret_p50"], mB["ret_p95"],
            mB["ret_backtest"], _COLOR_B, "Resample: Final Returns",
        )

        # ── Bottom-left: Risk of Ruin comparison (horizontal bars) ────────────
        ax_bl.barh(
            ["Reshuffle", "Resample"],
            [mA["ror_pct"], mB["ror_pct"]],
            color=[_COLOR_A, _COLOR_B],
            height=0.4,
        )
        ax_bl.axvline(5.0, color=_COLOR_OG, linestyle="--", linewidth=1.2,
                      alpha=0.7, label="5% threshold")
        ax_bl.set_xlabel("Risk of Ruin (%)")
        ax_bl.set_title("Risk of Ruin Comparison", fontsize=10)
        ax_bl.legend(fontsize=8)
        ax_bl.grid(alpha=0.2, axis="x")
        for i, v in enumerate([mA["ror_pct"], mB["ror_pct"]]):
            ax_bl.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=9, color="white")

        # ── Bottom-right: Metrics comparison table ────────────────────────────
        ax_br.axis("off")
        table_data = [
            [
                "Max DD P95",
                f"{mA['dd_p95']:.1f}%",
                f"{mB['dd_p95']:.1f}%",
                f"{mA['dd_backtest']:.1f}%",
            ],
            [
                "Sharpe P50",
                f"{mA['sharpe_p50']:.2f}",
                f"{mB['sharpe_p50']:.2f}",
                f"{mA['sharpe_backtest']:.2f}",
            ],
            [
                "Return P50",
                f"{mA['ret_p50']:.1f}%",
                f"{mB['ret_p50']:.1f}%",
                f"{mA['ret_backtest']:.1f}%",
            ],
            [
                "RoR %",
                f"{mA['ror_pct']:.1f}%",
                f"{mB['ror_pct']:.1f}%",
                "—",
            ],
            [
                "Verdict",
                verdict_A,
                verdict_B,
                "—",
            ],
        ]
        col_labels = ["Metric", "Reshuffle", "Resample", "Backtest Actual"]

        tbl = ax_br.table(
            cellText=table_data,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.1, 1.8)

        # Style header row
        for col_idx in range(len(col_labels)):
            tbl[0, col_idx].set_facecolor("#2A2A2A")
            tbl[0, col_idx].set_text_props(color="white", fontweight="bold")

        ax_br.set_title("Metrics Comparison", fontsize=10, pad=15)

        fig.suptitle("Monte Carlo — Summary Dashboard", fontsize=14, y=0.98)

        # tight_layout conflicts with tables; use manual spacing instead
        fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.93, bottom=0.06,
                            left=0.07, right=0.97)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _subplot_return_dist(
    ax:         plt.Axes,
    returns:    np.ndarray,
    p5:         float,
    p50:        float,
    p95:        float,
    backtest:   float,
    color:      str,
    title:      str,
) -> None:
    # Guard against degenerate range: Method A reshuffle always has identical
    # final_returns (product of returns is permutation-invariant).
    r_min, r_max = float(returns.min()), float(returns.max())
    if r_max - r_min < 1e-10:
        pad = max(abs(r_min) * 0.1, 1.0)
        bins_arg = np.linspace(r_min - pad, r_max + pad, 10)
    else:
        bins_arg = 40
    ax.hist(returns, bins=bins_arg, color=color, alpha=0.7)
    ax.axvline(p5,      color=color,    alpha=0.5, linestyle="--", linewidth=1.5,
               label=f"P5: {p5:.1f}%")
    ax.axvline(p50,     color=color,    alpha=0.5, linestyle="-",  linewidth=2.0,
               label=f"P50: {p50:.1f}%")
    ax.axvline(p95,     color=color,    alpha=0.5, linestyle="--", linewidth=1.5,
               label=f"P95: {p95:.1f}%")
    ax.axvline(backtest, color=_COLOR_OG, linestyle="--", linewidth=1.5,
               label=f"Actual: {backtest:.1f}%")
    ax.set_xlabel("Final Return (%)", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.2)
