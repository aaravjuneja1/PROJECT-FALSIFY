"""
Compute all performance metrics for FALSIFY.

Inputs
------
equity : pd.Series
    Daily portfolio value, index = pd.DatetimeIndex (OOS period only).
trades : pd.DataFrame
    One row per closed round-trip trade with columns:
    entry_date, exit_date, entry_price, exit_price, return_pct, direction.
    return_pct is a decimal (0.05 = 5% gain, -0.02 = 2% loss).

Output
------
dict — all metrics. Keys listed in calculate_all docstring.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.metrics_config import (
    DAILY_RISK_FREE_RATE,
    PROFIT_FACTOR_SUSPICIOUS_THRESHOLD,
    TRADING_DAYS_PER_YEAR,
)


def calculate_all(
    equity: pd.Series,
    trades: pd.DataFrame,
    risk_free_rate_daily: float = DAILY_RISK_FREE_RATE,
    start_value: float | None = None,
) -> dict:
    """
    Compute all metrics and return as a single dict.

    Parameters
    ----------
    equity : pd.Series
        Daily portfolio value, index = pd.DatetimeIndex.
    trades : pd.DataFrame
        Closed trade log with columns: entry_date, exit_date,
        entry_price, exit_price, return_pct, direction.
    risk_free_rate_daily : float
        Daily risk-free rate. Defaults to value in metrics_config.py.

    Returns
    -------
    dict with keys:
        sharpe                  — annualised Sharpe Ratio
        cagr                    — Compound Annual Growth Rate (decimal)
        max_drawdown            — maximum drawdown (decimal, e.g. 0.25 = 25%)
        max_drawdown_pct        — same, as percentage (e.g. 25.0)
        drawdown_start_date     — date of the peak before max drawdown
        trough_date             — date of the max drawdown low
        recovery_date           — date equity recovered to pre-drawdown peak (None if not yet)
        drawdown_duration_days  — trading days from peak to trough
        calmar                  — Calmar Ratio (CAGR / max drawdown)
        win_rate                — percentage of trades with return_pct > 0
        avg_win                 — mean return_pct of winning trades (decimal)
        avg_loss                — mean return_pct of losing trades (decimal, negative)
        risk_reward             — abs(avg_win) / abs(avg_loss)
        num_trades              — total number of closed trades
        avg_trades_per_week     — trades / (trading_days / 5)
        profit_factor           — gross wins / abs(gross losses)
        warnings                — list of strings flagging suspicious results
    """
    warnings: list[str] = []
    total_trading_days = len(equity)

    # Empty equity curve → return a fully-formed zeroed dict rather than crash
    # on equity.iloc[0]. Trade-based metrics are still computed if trades exist.
    if total_trading_days == 0:
        return {
            "sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "drawdown_start_date": None, "trough_date": None, "recovery_date": None,
            "drawdown_duration_days": 0, "calmar": 0.0,
            "win_rate": _win_rate(trades), "avg_win": _avg_win(trades),
            "avg_loss": _avg_loss(trades), "risk_reward": _risk_reward(trades),
            "num_trades": len(trades), "avg_trades_per_week": 0.0,
            "profit_factor": _profit_factor(trades), "warnings": warnings,
        }

    dd = _max_drawdown_details(equity)
    cagr_val = _cagr(equity, total_trading_days, start_value=float(equity.iloc[0]) if start_value is None else start_value)
    calmar_val = cagr_val / dd["max_drawdown"] if dd["max_drawdown"] != 0 else 0.0
    pf = _profit_factor(trades)

    if pf == float("inf") or pf > PROFIT_FACTOR_SUSPICIOUS_THRESHOLD:
        warnings.append(
            f"Profit factor is {pf:.2f} — exceeds {PROFIT_FACTOR_SUSPICIOUS_THRESHOLD}. "
            "Check for look-ahead bias or overfitting."
        )

    return {
        "sharpe": _sharpe(equity, risk_free_rate_daily),
        "cagr": cagr_val,
        "max_drawdown": dd["max_drawdown"],
        "max_drawdown_pct": dd["max_drawdown_pct"],
        "drawdown_start_date": dd["drawdown_start_date"],
        "trough_date": dd["trough_date"],
        "recovery_date": dd["recovery_date"],
        "drawdown_duration_days": dd["drawdown_duration_days"],
        "calmar": calmar_val,
        "win_rate": _win_rate(trades),
        "avg_win": _avg_win(trades),
        "avg_loss": _avg_loss(trades),
        "risk_reward": _risk_reward(trades),
        "num_trades": len(trades),
        "avg_trades_per_week": _avg_trades_per_week(trades, total_trading_days),
        "profit_factor": pf,
        "warnings": warnings,
    }


def _sharpe(equity: pd.Series, risk_free_rate_daily: float) -> float:
    """
    Annualised Sharpe Ratio (Lo 2002, Benhamou 2019).

    Formula : mean(excess) / std(excess) × √252
    excess   = daily_return - daily_risk_free_rate
    daily_risk_free_rate = (1.06)^(1/252) - 1

    Computed from equity curve daily returns, not trade returns.
    Returns 0.0 if fewer than 2 points or std of excess is zero (flat curve).
    """
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if len(returns) == 0:
        return 0.0
    excess = returns - risk_free_rate_daily
    std = float(excess.std())
    if std == 0.0:
        return 0.0
    return float((excess.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR))


# ──────────────────────────────────────────────────────────────────────────────
# Per-trade (holding-period) Sharpe — SINGLE SOURCE OF TRUTH.
# Imported by validation/monte_carlo.py and validation/stats.py so all three
# modules annualise per-trade Sharpe identically. Holding period is measured in
# TRADING days (np.busday_count) to be dimensionally consistent with the
# 252-trading-day year, and std uses ddof=1 (sample) everywhere.
# ──────────────────────────────────────────────────────────────────────────────

def avg_holding_trading_days(trades: pd.DataFrame) -> float:
    """
    Mean number of TRADING days (np.busday_count) between entry_date and exit_date.
    Returns at least 1.0. Falls back to 1.0 if trades is empty or the date
    columns are missing.
    """
    if trades is None or trades.empty:
        return 1.0
    if "entry_date" not in trades.columns or "exit_date" not in trades.columns:
        return 1.0
    entry = pd.to_datetime(trades["entry_date"]).values.astype("datetime64[D]")
    exit_ = pd.to_datetime(trades["exit_date"]).values.astype("datetime64[D]")
    days = np.busday_count(entry, exit_).astype(float)
    days = np.where(days < 1.0, 1.0, days)
    return float(np.mean(days)) if len(days) else 1.0


def per_trade_sharpe(returns, holding_days: float) -> float:
    """
    Annualised Sharpe from per-trade returns (decimal).

    Formula: (mean / std_ddof1) * sqrt(TRADING_DAYS_PER_YEAR / holding_days)
    Uses sample std (ddof=1). Returns 0.0 if fewer than 2 returns or std is zero.
    """
    arr = np.asarray(returns, dtype=float)
    if arr.size < 2:
        return 0.0
    std = float(np.std(arr, ddof=1))
    if std == 0.0:
        return 0.0
    return float((arr.mean() / std) * np.sqrt(TRADING_DAYS_PER_YEAR / max(holding_days, 1.0)))


def _cagr(equity: pd.Series, total_trading_days: int, start_value: float | None = None) -> float:
    """
    Compound Annual Growth Rate.

    Formula : (Final / start)^(252 / holding_days) - 1
    holding_days = len(equity) - 1  (5 data points = 4 return periods)
    start = start_value if provided, else equity.iloc[0].

    Returns 0.0 if fewer than 2 data points.
    Returns -1.0 if final value is zero or negative (total loss).
    """
    holding_days = total_trading_days - 1
    start = start_value if start_value is not None else float(equity.iloc[0])
    if holding_days < 1 or start <= 0:
        return 0.0
    final = float(equity.iloc[-1])
    ratio = final / start
    if ratio <= 0:
        return -1.0
    return float(ratio ** (TRADING_DAYS_PER_YEAR / holding_days) - 1)


def _max_drawdown_details(equity: pd.Series) -> dict:
    """
    Maximum Drawdown and associated dates.

    Formula : MDD = (Peak - Trough) / Peak, walked day by day.

    Also computes:
    - drawdown_start_date : the last date equity was at a running peak before the trough
    - trough_date         : date of deepest point of the worst drawdown
    - recovery_date       : first date after trough where equity >= peak at drawdown_start_date
                            (None if equity has not recovered within the data)
    - drawdown_duration_days : trading days from drawdown_start_date to trough_date

    Returns 0.0 MDD dict if fewer than 2 data points or no drawdown occurred.
    """
    empty: dict = {
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "drawdown_start_date": equity.index[0] if len(equity) > 0 else None,
        "trough_date": equity.index[0] if len(equity) > 0 else None,
        "recovery_date": None,
        "drawdown_duration_days": 0,
    }

    if len(equity) < 2:
        return empty

    running_peak = equity.cummax()
    drawdown = (running_peak - equity) / running_peak
    max_dd = float(drawdown.max())

    if max_dd == 0.0:
        return empty

    trough_date = drawdown.idxmax()

    # Drawdown start: last date strictly before trough where equity was at its running peak.
    # np.isclose handles floating point so we don't miss the peak due to rounding.
    before_trough_idx = equity.index[equity.index < trough_date]
    if len(before_trough_idx) == 0:
        dd_start = equity.index[0]
    else:
        eq_before = equity.loc[before_trough_idx].values
        pk_before = running_peak.loc[before_trough_idx].values
        at_peak_mask = np.isclose(eq_before, pk_before)
        at_peak_dates = before_trough_idx[at_peak_mask]
        dd_start = at_peak_dates[-1] if len(at_peak_dates) > 0 else before_trough_idx[0]

    # Recovery date: first date strictly after trough where equity >= peak value.
    peak_value = float(equity.loc[dd_start])
    after_trough_idx = equity.index[equity.index > trough_date]
    recovery_date = None
    if len(after_trough_idx) > 0:
        eq_after = equity.loc[after_trough_idx].values
        recovered_mask = eq_after >= peak_value
        if recovered_mask.any():
            recovery_date = after_trough_idx[recovered_mask][0]

    # Duration in trading days (using searchsorted — safe for all pandas versions).
    start_pos = int(equity.index.searchsorted(dd_start))
    trough_pos = int(equity.index.searchsorted(trough_date))
    duration = trough_pos - start_pos

    return {
        "max_drawdown": max_dd,
        "max_drawdown_pct": round(max_dd * 100, 4),
        "drawdown_start_date": dd_start,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
        "drawdown_duration_days": duration,
    }


def _win_rate(trades: pd.DataFrame) -> float:
    """
    Win Rate = profitable trades / total trades × 100.
    A trade is profitable if return_pct > 0.
    Returns 0.0 if trade log is empty.
    """
    if trades.empty:
        return 0.0
    wins = (trades["return_pct"] > 0).sum()
    return float(wins / len(trades) * 100)


def _avg_win(trades: pd.DataFrame) -> float:
    """
    Mean return_pct of all winning trades (return_pct > 0). Decimal.
    Returns 0.0 if no winning trades.
    """
    if trades.empty:
        return 0.0
    wins = trades[trades["return_pct"] > 0]["return_pct"]
    return float(wins.mean()) if len(wins) > 0 else 0.0


def _avg_loss(trades: pd.DataFrame) -> float:
    """
    Mean return_pct of all losing trades (return_pct < 0). Will be negative. Decimal.
    Returns 0.0 if no losing trades.
    """
    if trades.empty:
        return 0.0
    losses = trades[trades["return_pct"] < 0]["return_pct"]
    return float(losses.mean()) if len(losses) > 0 else 0.0


def _risk_reward(trades: pd.DataFrame) -> float:
    """
    Risk/Reward Ratio = abs(avg_win) / abs(avg_loss).
    Returns 0.0 if no trades or no losing trades to divide by.
    """
    avg_w = _avg_win(trades)
    avg_l = _avg_loss(trades)
    if avg_l == 0.0:
        return 0.0
    return float(abs(avg_w) / abs(avg_l))


def _avg_trades_per_week(trades: pd.DataFrame, total_trading_days: int) -> float:
    """
    Average trades per week = total_trades / (total_trading_days / 5).
    Assumes 5 trading days per week.
    Returns 0.0 if total_trading_days is zero.
    """
    if total_trading_days == 0:
        return 0.0
    weeks = total_trading_days / 5
    return float(len(trades) / weeks)


def _profit_factor(trades: pd.DataFrame) -> float:
    """
    Profit Factor = sum of winning return_pct / abs(sum of losing return_pct).

    Returns 0.0 if no trades.
    Returns float('inf') if there are wins but zero losses.
    A score above 1.5 is minimum viable. Above 4.0 is suspicious (flagged in calculate_all).
    """
    if trades.empty:
        return 0.0
    gross_profit = trades[trades["return_pct"] > 0]["return_pct"].sum()
    gross_loss = trades[trades["return_pct"] < 0]["return_pct"].sum()
    if gross_loss == 0.0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / abs(gross_loss))
