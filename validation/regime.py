"""
Regime stress test for FALSIFY.

Slices the OOS trade log by market regime date ranges, computes full
performance metrics per regime, identifies profit concentration, and flags
dangerous patterns (crash losses, bear-market losses, single-regime dominance).

Entry point: run_regime_analysis(trade_log, equity_curve, config) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metrics.calculator import calculate_all
from configs.metrics_config import (
    TRADING_DAYS_PER_YEAR,
    DAILY_RISK_FREE_RATE,
)

# ── Regime type colour palette ────────────────────────────────────────────────
REGIME_COLORS: dict[str, str] = {
    "bull":     "#4CAF50",
    "bear":     "#E53935",
    "crash":    "#B71C1C",
    "recovery": "#42A5F5",
    "sideways": "#FFA726",
}
_DEFAULT_REGIME_COLOR = "#9E9E9E"   # grey for unknown types


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_regime_analysis(
    trade_log:    pd.DataFrame,
    equity_curve: pd.Series | pd.DataFrame,
    config:       dict,
) -> dict:
    """
    Run regime stress test on OOS trade log and equity curve.

    Parameters
    ----------
    trade_log    : OOS trade log from WFA. Required columns: entry_date,
                   exit_date, return_pct. pnl_abs is computed if missing.
    equity_curve : OOS equity curve from WFA. Accepts pd.Series (date index)
                   or pd.DataFrame with 'date' and 'equity' columns.
    config       : settings dict. Required key: 'regimes' (list of regime dicts).
                   Each regime dict: name, start, end, type.
    """
    # ── Guard: empty trade log ─────────────────────────────────────────────────
    if trade_log.empty:
        raise ValueError("Trade log is empty. Cannot run regime analysis.")

    # ── Extract config ─────────────────────────────────────────────────────────
    regime_cfg     = config.get("regime", {})
    min_trades     = int(regime_cfg.get("min_trades", 5))
    initial_capital = float(config["capital"]["starting_capital"])
    rfr_daily       = DAILY_RISK_FREE_RATE   # single source of truth: metrics_config.py

    # ── Parse equity curve → pd.Series indexed by date ────────────────────────
    equity_series = _parse_equity(equity_curve)

    # ── Ensure trade_log has pnl_abs ───────────────────────────────────────────
    trade_log = _ensure_pnl_abs(trade_log.copy(), equity_series, initial_capital)

    # ── Parse and validate regimes from config ─────────────────────────────────
    regimes = _parse_regimes(config)

    # ── Assign trades to regimes ───────────────────────────────────────────────
    regime_trade_map, unclassified_df = _assign_trades_to_regimes(
        trade_log, regimes
    )

    # ── Overall total P&L (used for contribution_pct) ─────────────────────────
    overall_pnl = float(trade_log["pnl_abs"].sum())
    if overall_pnl is None or overall_pnl <= 1e-6:
        print(
            "WARNING: Overall total_pnl_abs is zero or negative. "
            "Contribution % not shown: total P&L is zero or negative."
        )
        overall_pnl = None

    # ── Compute metrics per regime ─────────────────────────────────────────────
    regime_results: dict[str, dict] = {}

    for regime in regimes:
        r_name    = regime["name"]
        r_type    = regime["type"]
        r_start   = regime["start"]
        r_end     = regime["end"]
        r_trades  = regime_trade_map.get(r_name, pd.DataFrame())

        is_crash     = r_type == "crash"
        n_trades_reg = len(r_trades)

        # Equity slice for this regime
        equity_slice = _slice_equity(equity_series, r_start, r_end)

        if n_trades_reg == 0:
            print(f"  Regime '{r_name}': no OOS trades fall in this window.")
            regime_results[r_name] = {
                "regime":         regime,
                "status":         "no_trades",
                "always_reported": is_crash,
                **{k: None for k in _METRIC_KEYS},
            }
            continue

        # Insufficient trades (skip unless crash)
        if n_trades_reg < min_trades and not is_crash:
            regime_results[r_name] = {
                "regime":         regime,
                "status":         "insufficient_data",
                "always_reported": False,
                "num_trades":     n_trades_reg,
                **{k: None for k in _METRIC_KEYS if k != "num_trades"},
            }
            continue

        # Compute full metrics
        metrics = _compute_regime_metrics(
            r_trades, equity_slice, r_start, r_end,
            initial_capital, rfr_daily, is_crash
        )
        metrics["regime"] = {"name": r_name, "start": r_start, "end": r_end, "type": r_type}
        regime_results[r_name] = metrics

    # ── Fill contribution_pct ──────────────────────────────────────────────────
    if overall_pnl is not None:
        for r_name, r_res in regime_results.items():
            pnl = r_res.get("total_pnl_abs")
            if pnl is not None:
                r_res["contribution_pct"] = float(pnl / overall_pnl * 100.0)

    # ── Aggregate across all reported regimes ──────────────────────────────────
    aggregate = _compute_aggregate(regime_results)

    # ── Verdict ────────────────────────────────────────────────────────────────
    verdict, verdict_detail = _get_verdict(
        aggregate["profitable_regimes"],
        aggregate["total_reported"],
        aggregate.get("profit_concentration"),
        aggregate.get("bear_sharpe"),
    )

    # ── Warning flags ──────────────────────────────────────────────────────────
    crash_flag, concentration_flag, bear_flag = _compute_flags(
        regime_results, aggregate
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_dir = Path(__file__).resolve().parents[1] / config["reporting"]["charts_dir"]
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = _generate_all_charts(
        regime_results, regimes, equity_series,
        overall_pnl, charts_dir
    )

    # ── Console summary ───────────────────────────────────────────────────────
    _print_summary(
        trade_log, regime_results, unclassified_df,
        aggregate, verdict, verdict_detail,
        crash_flag, concentration_flag, bear_flag,
    )

    return {
        "regime_results":      regime_results,
        "aggregate":           aggregate,
        "unclassified_trades": unclassified_df,
        "verdict":             verdict,
        "verdict_detail":      verdict_detail,
        "crash_flag":          crash_flag,
        "concentration_flag":  concentration_flag,
        "bear_flag":           bear_flag,
        "chart_paths":         chart_paths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_METRIC_KEYS = [
    "sharpe", "cagr", "total_return", "use_total_return", "display_return",
    "return_metric", "max_drawdown", "win_rate", "profit_factor",
    "num_trades", "total_pnl_abs", "avg_trade_pnl",
    "best_trade_pnl", "worst_trade_pnl", "contribution_pct",
]


# ──────────────────────────────────────────────────────────────────────────────
# Input parsing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _parse_equity(equity_curve: pd.Series | pd.DataFrame) -> pd.Series:
    """
    Accept equity curve as pd.Series (date index) or DataFrame with
    'date' / 'equity' columns. Always returns pd.Series indexed by date.
    """
    if isinstance(equity_curve, pd.Series):
        s = equity_curve.copy()
        s.index = pd.to_datetime(s.index)
        return s

    if isinstance(equity_curve, pd.DataFrame):
        df = equity_curve.copy()
        if "equity" in df.columns and "date" in df.columns:
            s = df.set_index("date")["equity"]
        elif "equity" in df.columns:
            s = df["equity"]
            s.index = pd.to_datetime(s.index)
        else:
            # Assume first column is equity, index is dates
            s = df.iloc[:, 0]
            s.index = pd.to_datetime(s.index)
        s.index = pd.to_datetime(s.index)
        return s

    raise TypeError(f"equity_curve must be pd.Series or pd.DataFrame, got {type(equity_curve)}")


def _ensure_pnl_abs(
    trade_log:      pd.DataFrame,
    equity_series:  pd.Series,
    initial_capital: float,
) -> pd.DataFrame:
    """
    Compute pnl_abs if not present: return_pct × portfolio_value_at_entry.
    Portfolio value at entry is looked up from equity_series; falls back to
    initial_capital if not found.
    """
    if "pnl_abs" in trade_log.columns:
        return trade_log

    eq_dict = equity_series.to_dict() if not equity_series.empty else {}

    def _lookup_equity(entry_date: pd.Timestamp) -> float:
        if not eq_dict:
            return initial_capital
        valid = [d for d in eq_dict if d <= entry_date]
        return eq_dict[max(valid)] if valid else initial_capital

    entries = pd.to_datetime(trade_log["entry_date"])
    pnl_abs = [
        float(ret) * _lookup_equity(e_date)
        for ret, e_date in zip(trade_log["return_pct"], entries)
    ]
    trade_log["pnl_abs"] = pnl_abs
    return trade_log


def _parse_regimes(config: dict) -> list[dict]:
    """
    Parse regimes from config['regimes']. Validate start < end.
    Returns list of dicts with start/end as pd.Timestamp.
    """
    raw_regimes = config.get("regimes", [])
    if not raw_regimes:
        raise ValueError("No regimes defined in config['regimes'].")

    regimes = []
    for r in raw_regimes:
        start = pd.Timestamp(r["start"])
        end   = pd.Timestamp(r["end"])
        if start >= end:
            raise ValueError(
                f"Regime '{r['name']}': start ({r['start']}) must be "
                f"before end ({r['end']})."
            )
        regimes.append({
            "name":  r["name"],
            "start": start,
            "end":   end,
            "type":  r.get("type", "unknown"),
        })
    return regimes


def _slice_equity(
    equity_series: pd.Series,
    start:         pd.Timestamp,
    end:           pd.Timestamp,
) -> pd.Series:
    """Return equity rows whose date index falls within [start, end]."""
    if equity_series.empty:
        return pd.Series(dtype=float)
    mask = (equity_series.index >= start) & (equity_series.index <= end)
    return equity_series[mask]


# ──────────────────────────────────────────────────────────────────────────────
# Trade assignment
# ──────────────────────────────────────────────────────────────────────────────

def _assign_trades_to_regimes(
    trade_log: pd.DataFrame,
    regimes:   list[dict],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Assign each trade to a regime based on entry_date.
    Returns (regime_trade_map, unclassified_df).
    """
    regime_rows: dict[str, list] = {r["name"]: [] for r in regimes}
    unclassified: list = []

    entries = pd.to_datetime(trade_log["entry_date"])

    for row_idx in range(len(trade_log)):
        row        = trade_log.iloc[row_idx]
        entry_date = entries.iloc[row_idx]

        matching = [
            r for r in regimes
            if r["start"] <= entry_date <= r["end"]
        ]

        if len(matching) == 0:
            unclassified.append(row)

        elif len(matching) == 1:
            regime_rows[matching[0]["name"]].append(row)

        else:
            # Overlap: assign to regime whose midpoint is closest to entry_date
            def _midpoint(r: dict) -> pd.Timestamp:
                return r["start"] + (r["end"] - r["start"]) / 2

            closest = min(matching, key=lambda r: abs(_midpoint(r) - entry_date))
            print(
                f"  WARNING: Trade on {entry_date.date()} falls in overlapping "
                f"regimes: {[r['name'] for r in matching]}. "
                f"Assigned to '{closest['name']}' (closest midpoint)."
            )
            regime_rows[closest["name"]].append(row)

    regime_map = {
        name: (
            pd.DataFrame(rows).reset_index(drop=True)
            if rows
            else pd.DataFrame(columns=trade_log.columns)
        )
        for name, rows in regime_rows.items()
    }
    unclass_df = (
        pd.DataFrame(unclassified).reset_index(drop=True)
        if unclassified
        else pd.DataFrame(columns=trade_log.columns)
    )
    return regime_map, unclass_df


# ──────────────────────────────────────────────────────────────────────────────
# Per-regime metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_regime_metrics(
    r_trades:       pd.DataFrame,
    equity_slice:   pd.Series,
    r_start:        pd.Timestamp,
    r_end:          pd.Timestamp,
    initial_capital: float,
    rfr_daily:      float,
    is_crash:       bool,
) -> dict:
    """
    Compute all metrics for a single regime.
    Uses calculate_all for Sharpe/DD/win_rate/profit_factor.
    Overrides CAGR with regime-specific computation (start→end of regime).
    """
    # Build equity to use: equity_slice if available, else reconstruct from trades
    if len(equity_slice) >= 2:
        eq = equity_slice
    else:
        # Reconstruct a rough equity curve from trade returns
        rets = r_trades["return_pct"].values.astype(float)
        eq_vals = initial_capital * np.cumprod(1.0 + rets)
        eq = pd.Series(eq_vals)

    # calculate_all handles most metrics
    m = calculate_all(eq, r_trades, risk_free_rate_daily=rfr_daily)

    # Regime-specific return: both annualised CAGR and total (cumulative) return.
    if len(eq) >= 2 and float(eq.iloc[0]) > 0:
        eq_start  = float(eq.iloc[0])
        eq_end    = float(eq.iloc[-1])
        hold_days = max(len(eq) - 1, 1)
        regime_cagr  = float((eq_end / eq_start) ** (TRADING_DAYS_PER_YEAR / hold_days) - 1.0)
        total_return = float(eq_end / eq_start - 1.0)
    else:
        regime_cagr  = 0.0
        total_return = 0.0

    # Short windows (<6 months) and the COVID/crash window annualise to absurd
    # CAGRs (e.g. a -30% drop over 2 months → a meaningless annualised figure),
    # so report TOTAL return for them. Longer regimes keep annualised CAGR.
    short_window     = (r_start + pd.DateOffset(months=6)) > r_end
    use_total_return = bool(is_crash or short_window)
    display_return   = total_return if use_total_return else regime_cagr

    total_pnl = float(r_trades["pnl_abs"].sum())
    avg_pnl   = float(r_trades["pnl_abs"].mean())
    best_pnl  = float(r_trades["pnl_abs"].max())
    worst_pnl = float(r_trades["pnl_abs"].min())

    return {
        "status":          "computed",
        "always_reported": is_crash,
        "num_trades":      int(m["num_trades"]),
        "sharpe":          float(m["sharpe"]),
        "cagr":            regime_cagr,
        "total_return":    total_return,
        "use_total_return": use_total_return,
        "display_return":  display_return,
        "return_metric":   "total_return" if use_total_return else "cagr",
        "max_drawdown":    float(m["max_drawdown"]),
        "win_rate":        float(m["win_rate"]),
        "profit_factor":   float(m["profit_factor"]),
        "total_pnl_abs":   total_pnl,
        "avg_trade_pnl":   avg_pnl,
        "best_trade_pnl":  best_pnl,
        "worst_trade_pnl": worst_pnl,
        "contribution_pct": None,   # filled in caller
    }


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate metrics
# ──────────────────────────────────────────────────────────────────────────────

def _compute_aggregate(regime_results: dict) -> dict:
    """Aggregate metrics across all fully computed regimes."""
    computed = {
        name: r for name, r in regime_results.items()
        if r.get("status") == "computed"
    }

    if not computed:
        return {
            "profitable_regimes":  0,
            "losing_regimes":      0,
            "total_reported":      0,
            "best_regime":         None,
            "worst_regime":        None,
            "profit_concentration": None,
            "bear_sharpe":         None,
            "bear_return":         None,
            "bull_sharpe":         None,
        }

    profitable = sum(1 for r in computed.values() if (r.get("total_pnl_abs") or 0) > 0)
    losing     = sum(1 for r in computed.values() if (r.get("total_pnl_abs") or 0) < 0)

    sharpes = {name: r["sharpe"] for name, r in computed.items() if r["sharpe"] == r["sharpe"]}
    best    = max(sharpes, key=sharpes.get) if sharpes else None
    worst   = min(sharpes, key=sharpes.get) if sharpes else None

    # Profit concentration: max |contribution_pct| across all computed regimes
    contribs = [
        r["contribution_pct"] for r in computed.values()
        if r.get("contribution_pct") is not None
    ]
    profit_concentration = float(max(contribs)) if contribs else None

    # Bear/crash average Sharpe
    bear_sharpes = [
        r["sharpe"] for name, r in computed.items()
        if regime_results[name]["regime"]["type"] in ("bear", "crash")
        and r["sharpe"] == r["sharpe"]
    ]
    bear_sharpe = float(np.mean(bear_sharpes)) if bear_sharpes else None

    # Average TOTAL return across bear/crash regimes — used by the verdict
    # bear-survival gate (config verdict.bear_return_min).
    bear_returns = [
        r["total_return"] for name, r in computed.items()
        if regime_results[name]["regime"]["type"] in ("bear", "crash")
        and r.get("total_return") is not None
    ]
    bear_return = float(np.mean(bear_returns)) if bear_returns else None

    # Bull average Sharpe
    bull_sharpes = [
        r["sharpe"] for name, r in computed.items()
        if regime_results[name]["regime"]["type"] == "bull"
        and r["sharpe"] == r["sharpe"]
    ]
    bull_sharpe = float(np.mean(bull_sharpes)) if bull_sharpes else None

    return {
        "profitable_regimes":   profitable,
        "losing_regimes":       losing,
        "total_reported":       len(computed),
        "best_regime":          {"name": best, "sharpe": sharpes.get(best)} if best else None,
        "worst_regime":         {"name": worst, "sharpe": sharpes.get(worst)} if worst else None,
        "profit_concentration": profit_concentration,
        "bear_sharpe":          bear_sharpe,
        "bear_return":          bear_return,
        "bull_sharpe":          bull_sharpe,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Verdict and flags
# ──────────────────────────────────────────────────────────────────────────────

def _get_verdict(
    profitable:           int,
    total_reported:       int,
    profit_concentration: float | None,
    bear_sharpe:          float | None,
) -> tuple[str, str]:
    if total_reported == 0:
        return ("INSUFFICIENT DATA", "Not enough regimes with data to judge.")

    ratio        = profitable / total_reported
    concentration = profit_concentration if profit_concentration is not None else 100.0
    bear_ok       = (bear_sharpe is not None and bear_sharpe > 0)

    if ratio >= 0.7 and concentration <= 50 and bear_ok:
        return (
            "ROBUST",
            "Strategy profitable across most regimes including bear conditions. "
            "No dangerous profit concentration.",
        )
    elif ratio >= 0.5 and concentration <= 70:
        return (
            "REGIME-DEPENDENT",
            "Strategy profitable in some regimes but not others. "
            "Understand which conditions it requires before trading.",
        )
    else:
        return (
            "FRAGILE",
            "Strategy performance concentrated in specific regimes "
            "or consistently loses in adverse conditions.",
        )


def _compute_flags(
    regime_results: dict,
    aggregate:      dict,
) -> tuple[bool, bool, bool]:
    """Return (crash_flag, concentration_flag, bear_flag)."""
    # crash_flag: any crash regime with Sharpe < -0.5
    crash_flag = any(
        r.get("sharpe", 0) is not None
        and r.get("sharpe", 0) < -0.5
        and r.get("regime", {}).get("type") == "crash"
        for r in regime_results.values()
        if r.get("status") == "computed"
    )

    concentration_flag = (
        aggregate.get("profit_concentration") is not None
        and aggregate["profit_concentration"] > 60.0
    )

    bear_flag = (
        aggregate.get("bear_sharpe") is not None
        and aggregate["bear_sharpe"] < 0
    )

    return crash_flag, concentration_flag, bear_flag


# ──────────────────────────────────────────────────────────────────────────────
# Console print
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(
    trade_log:      pd.DataFrame,
    regime_results: dict,
    unclassified:   pd.DataFrame,
    aggregate:      dict,
    verdict:        str,
    verdict_detail: str,
    crash_flag:     bool,
    conc_flag:      bool,
    bear_flag:      bool,
) -> None:
    SEP  = "═" * 50
    THIN = "─" * 50

    n_regimes_with_trades = sum(
        1 for r in regime_results.values()
        if r.get("status") in ("computed", "insufficient_data")
        and r.get("num_trades", 0) > 0
    )

    print(f"\n{SEP}")
    print("REGIME STRESS TEST RESULTS — FALSIFY")
    print(SEP)
    print(f"Total OOS trades analysed:  {len(trade_log)}")
    print(f"Regimes defined in config:  {len(regime_results)}")
    print(f"Regimes with OOS trades:    {n_regimes_with_trades}")
    print(f"Unclassified trades:        {len(unclassified)}")
    print(THIN)
    print("REGIME BREAKDOWN")
    print(THIN)

    # Header
    print(
        f"{'Regime':<22} {'Type':<10} {'Trades':>6} "
        f"{'Sharpe':>7} {'Return':>8} {'MaxDD':>7} {'Contrib%':>9}"
    )

    for r_name, r_res in regime_results.items():
        regime  = r_res.get("regime", {})
        r_type  = regime.get("type", "?")
        always  = r_res.get("always_reported", False)
        status  = r_res.get("status", "?")
        label   = f"{r_name} ⚠" if always else r_name

        if status == "no_trades":
            print(f"  {label:<20} {r_type:<10} {'0':>6}  {'—':>7} {'—':>7} {'—':>7} {'—':>9}")
            continue
        if status == "insufficient_data":
            n = r_res.get("num_trades", 0)
            print(f"  {label:<20} {r_type:<10} {n:>6}  {'<min':>7} {'—':>7} {'—':>7} {'—':>9}")
            continue

        n     = r_res.get("num_trades", 0)
        sh    = r_res.get("sharpe")
        disp  = r_res.get("display_return")
        rmet  = r_res.get("return_metric")
        dd    = r_res.get("max_drawdown")
        cont  = r_res.get("contribution_pct")

        trades_str = f"{n}*" if always else str(n)
        sh_str    = f"{sh:.2f}"   if sh   is not None else "—"
        if disp is not None:
            mark    = "t" if rmet == "total_return" else ""
            ret_str = f"{disp*100:+.1f}%{mark}"
        else:
            ret_str = "—"
        dd_str    = f"{dd*100:.1f}%"   if dd   is not None else "—"
        cont_str  = (f"+{cont:.1f}%" if cont >= 0 else f"{cont:.1f}%") if cont is not None else "—"

        print(
            f"  {label:<20} {r_type:<10} {trades_str:>6}  "
            f"{sh_str:>7} {ret_str:>8} {dd_str:>7} {cont_str:>9}"
        )

    if any(r.get("always_reported") for r in regime_results.values()):
        print("\n  * COVID crash reported regardless of trade count.")
    if any(r.get("use_total_return") for r in regime_results.values()):
        print("  t = total return (window <6 months or crash); others annualised CAGR.")

    print(THIN)
    print("AGGREGATE")
    print(THIN)
    total = aggregate.get("total_reported", 0)
    profitable = aggregate.get("profitable_regimes", 0)
    print(f"Profitable regimes:     {profitable} / {total}")

    bull_sh = aggregate.get("bull_sharpe")
    bear_sh = aggregate.get("bear_sharpe")
    conc    = aggregate.get("profit_concentration")
    best    = aggregate.get("best_regime")
    worst   = aggregate.get("worst_regime")

    if bull_sh is not None:
        print(f"Bull Sharpe (avg):      {bull_sh:.2f}")
    if bear_sh is not None:
        print(f"Bear/Crash Sharpe(avg): {bear_sh:.2f}")
    if conc is not None:
        best_name = best["name"] if best else "?"
        print(f"Profit concentration:   {conc:.1f}% ({best_name})")

    print(f"\nVERDICT: {verdict}")
    print(f"Detail: {verdict_detail}")

    print(THIN)
    if conc_flag:
        print("⚠ CONCENTRATION WARNING: >60% profit from one regime")
    if crash_flag:
        print("⚠ CRASH WARNING: severe loss in crash period (Sharpe < -0.5)")
    if bear_flag:
        print("⚠ BEAR WARNING: negative Sharpe in bear/crash regimes on average")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Chart generation
# ──────────────────────────────────────────────────────────────────────────────

def _generate_all_charts(
    regime_results: dict,
    regimes:        list[dict],
    equity_series:  pd.Series,
    overall_pnl:    float | None,
    charts_dir:     Path,
) -> list[str]:
    paths: list[str] = []

    p1 = charts_dir / "regime_sharpe_bars.png"
    _chart_sharpe_bars(regime_results, p1)
    paths.append(str(p1))

    p2 = charts_dir / "regime_profit_donut.png"
    _chart_profit_donut(regime_results, overall_pnl, p2)
    paths.append(str(p2))

    p3 = charts_dir / "regime_equity_overlay.png"
    _chart_equity_overlay(equity_series, regimes, regime_results, p3)
    paths.append(str(p3))

    p4 = charts_dir / "regime_metrics_heatmap.png"
    _chart_metrics_heatmap(regime_results, p4)
    paths.append(str(p4))

    return paths


def _chart_sharpe_bars(regime_results: dict, save_path: Path) -> None:
    """Horizontal bar chart: Sharpe per regime, coloured by type."""
    computed = [
        (name, r) for name, r in regime_results.items()
        if r.get("status") == "computed"
    ]
    if not computed:
        _save_empty_chart("No computed regimes", save_path)
        return

    names  = [n for n, _ in computed]
    sharpes = [r.get("sharpe", 0.0) or 0.0 for _, r in computed]
    colors  = [
        REGIME_COLORS.get(r.get("regime", {}).get("type", ""), _DEFAULT_REGIME_COLOR)
        for _, r in computed
    ]
    n_trades = [r.get("num_trades", 0) for _, r in computed]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))

        y_pos = np.arange(len(names))
        ax.barh(y_pos, sharpes, color=colors, height=0.5)
        ax.axvline(0.0, color="white", linewidth=1.2, alpha=0.8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.set_xlabel("Sharpe Ratio")
        ax.set_title("Sharpe Ratio by Market Regime (OOS)", fontsize=12)
        ax.grid(alpha=0.2, axis="x")

        # Annotate with trade count
        for i, (sh, n) in enumerate(zip(sharpes, n_trades)):
            x_ann = sh + (0.03 if sh >= 0 else -0.03)
            ha    = "left" if sh >= 0 else "right"
            ax.text(x_ann, i, f"n={n}", va="center", fontsize=8, color="white", ha=ha)

        # Legend
        legend_handles = [
            mpatches.Patch(color=c, label=t)
            for t, c in REGIME_COLORS.items()
        ]
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_profit_donut(
    regime_results: dict,
    overall_pnl:    float | None,
    save_path:      Path,
) -> None:
    """Donut chart: profit contribution per regime."""
    computed = [
        (name, r) for name, r in regime_results.items()
        if r.get("status") == "computed" and r.get("contribution_pct") is not None
    ]
    if not computed or overall_pnl is None:
        _save_empty_chart("No contribution data", save_path)
        return

    labels: list[str] = []
    sizes:  list[float] = []
    colors: list[str]   = []

    for name, r in computed:
        contrib = r["contribution_pct"]
        r_type  = r.get("regime", {}).get("type", "")
        if contrib >= 0:
            labels.append(name)
            sizes.append(contrib)
            colors.append(REGIME_COLORS.get(r_type, _DEFAULT_REGIME_COLOR))
        else:
            labels.append(f"Loss: {name}")
            sizes.append(abs(contrib))
            colors.append("#B71C1C")   # dark red for losses

    if not sizes or sum(sizes) < 1e-6:
        _save_empty_chart("All contributions zero", save_path)
        return

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 7))

        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            pctdistance=0.75,
            startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "#111111", "linewidth": 1},
        )
        for t in texts:
            t.set_color("white")
            t.set_fontsize(8)
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(8)

        # Total P&L in centre
        pnl_str = f"₹{overall_pnl:,.0f}" if overall_pnl >= 0 else f"-₹{abs(overall_pnl):,.0f}"
        ax.text(0, 0, f"Total\n{pnl_str}", ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

        ax.set_title("Profit Contribution by Regime", fontsize=12)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_equity_overlay(
    equity_series:  pd.Series,
    regimes:        list[dict],
    regime_results: dict,
    save_path:      Path,
) -> None:
    """OOS equity curve with regime period shading."""
    if equity_series.empty:
        _save_empty_chart("No equity curve data", save_path)
        return

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(14, 6))

        # Equity curve
        ax.plot(
            equity_series.index, equity_series.values,
            color="white", linewidth=1.5, zorder=5, label="OOS Equity"
        )

        y_min = float(equity_series.min())
        y_max = float(equity_series.max())
        y_range = max(y_max - y_min, 1.0)

        # Regime bands
        for i, regime in enumerate(regimes):
            r_type  = regime["type"]
            r_color = REGIME_COLORS.get(r_type, _DEFAULT_REGIME_COLOR)
            band_start = regime["start"]
            band_end   = regime["end"]

            # If adjacent to next regime, offset end by 1 day to avoid overlap
            if i < len(regimes) - 1:
                next_start = regimes[i + 1]["start"]
                if band_end >= next_start:
                    band_end = next_start - pd.Timedelta(days=1)

            ax.axvspan(band_start, band_end, alpha=0.15, color=r_color, zorder=1)

            # Regime name label at top of band
            mid_date = band_start + (band_end - band_start) / 2
            short_name = regime["name"][:10]
            ax.text(
                mid_date, y_max + y_range * 0.02, short_name,
                ha="center", va="bottom", fontsize=7,
                color=r_color, rotation=30, clip_on=True
            )

        ax.set_xlim(equity_series.index[0], equity_series.index[-1])
        ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio Value (INR)")
        ax.set_title("OOS Equity Curve — Regime Overlay", fontsize=12)
        ax.grid(alpha=0.15)

        # Legend for regime types
        present_types = list({r["type"] for r in regimes})
        handles = [
            mpatches.Patch(color=REGIME_COLORS.get(t, _DEFAULT_REGIME_COLOR),
                           alpha=0.4, label=t)
            for t in present_types
        ]
        handles.append(
            plt.Line2D([0], [0], color="white", linewidth=1.5, label="Equity")
        )
        ax.legend(handles=handles, fontsize=8, loc="upper left")

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_metrics_heatmap(regime_results: dict, save_path: Path) -> None:
    """
    Regime × metric heatmap. Cells coloured by RdYlGn, normalised per column.
    Max drawdown column has inverted scale (higher DD = redder).
    """
    computed = {
        name: r for name, r in regime_results.items()
        if r.get("status") == "computed"
    }
    if not computed:
        _save_empty_chart("No computed regimes", save_path)
        return

    _COLS        = ["sharpe", "cagr", "max_drawdown", "win_rate",
                    "profit_factor", "num_trades", "contribution_pct"]
    _COL_LABELS  = ["Sharpe", "CAGR", "Max DD", "Win Rate",
                    "PF", "Trades", "Contrib%"]
    _INVERT_COLS = {"max_drawdown"}   # higher = worse = redder

    regime_names = list(computed.keys())
    n_rows = len(regime_names)
    n_cols = len(_COLS)

    raw_matrix  = np.full((n_rows, n_cols), np.nan)
    disp_matrix = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for i, r_name in enumerate(regime_names):
        r = computed[r_name]
        for j, col in enumerate(_COLS):
            val = r.get(col)
            if val is None or (isinstance(val, float) and val != val):
                continue
            fval = float(val)
            # Skip inf in raw matrix (normalisation breaks); display string is set below
            if not np.isfinite(fval):
                continue
            raw_matrix[i, j] = fval
            # Format for display
            if col == "cagr":
                disp_matrix[i][j] = f"{val * 100:.1f}%"
            elif col in ("win_rate", "contribution_pct"):
                disp_matrix[i][j] = f"{val:.1f}%"
            elif col == "max_drawdown":
                disp_matrix[i][j] = f"{val * 100:.1f}%"
            elif col == "num_trades":
                disp_matrix[i][j] = str(int(val))
            elif col == "profit_factor" and val == float("inf"):
                disp_matrix[i][j] = "∞"
            else:
                disp_matrix[i][j] = f"{val:.2f}"

    # Normalise each column 0→1 (or 1→0 for inverted cols)
    norm_matrix = np.full_like(raw_matrix, 0.5)
    for j, col in enumerate(_COLS):
        col_vals = raw_matrix[:, j]
        valid    = col_vals[~np.isnan(col_vals)]
        if len(valid) < 2:
            norm_matrix[:, j] = np.where(np.isnan(col_vals), np.nan, 0.5)
            continue
        c_min, c_max = valid.min(), valid.max()
        if c_max == c_min:
            norm_matrix[:, j] = np.where(np.isnan(col_vals), np.nan, 0.5)
        else:
            normed = (col_vals - c_min) / (c_max - c_min)
            if col in _INVERT_COLS:
                normed = 1.0 - normed
            norm_matrix[:, j] = np.where(np.isnan(col_vals), np.nan, normed)

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("gray", alpha=0.3)

    with plt.style.context("dark_background"):
        cell_w = 1.4
        fig, ax = plt.subplots(
            figsize=(cell_w * n_cols + 2.5, max(3, n_rows * 0.7 + 1.5))
        )

        masked = np.ma.masked_invalid(norm_matrix)
        im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(_COL_LABELS, fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(regime_names, fontsize=9)

        # Annotate cells
        for i in range(n_rows):
            for j in range(n_cols):
                txt = disp_matrix[i][j]
                if txt:
                    bg = norm_matrix[i, j]
                    text_color = "black" if (bg == bg and 0.3 < bg < 0.8) else "white"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=8, color=text_color)

        plt.colorbar(im, ax=ax, label="Relative performance (per column)",
                     shrink=0.6, pad=0.02)
        ax.set_title("Regime Performance Metrics Heatmap", fontsize=12, pad=12)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _save_empty_chart(message: str, save_path: Path) -> None:
    """Save a placeholder chart when there is no data to plot."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, message, transform=ax.transAxes,
                ha="center", va="center", fontsize=13, color="gray")
        ax.set_axis_off()
        fig.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
