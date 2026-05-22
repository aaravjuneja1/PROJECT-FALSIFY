"""
Walk-Forward Analysis engine with integrated grid search.

Inputs
------
price_data : pd.DataFrame
    Single-ticker OHLCV DataFrame with columns
    [open, high, low, close, adj_close] and a pd.DatetimeIndex.
    Extract from data/fetcher.py PricePanel with:
        df = pd.DataFrame({
            "open":      panel.open["TICKER"],
            "close":     panel.close["TICKER"],
            "adj_close": panel.adj_close["TICKER"],
        })
strategy : object
    Must implement:
        generate_signals(data: pd.DataFrame, params: dict) -> pd.Series
    Signal values: 1 = long, 0 = flat/exit. -1 reserved for future short.
    The returned Series must be indexed by trading date.
    T+1 entry is enforced by the backtester: a signal at close of bar T is
    executed at the OPEN of bar T+1 (pending-order mechanism). Do NOT shift
    signals here or inside strategies — that would delay entry to T+2.
param_grid : dict[str, list]
    Maps parameter name → list of values to search.
    e.g. {'short_window': [10, 15, 20], 'long_window': [40, 50, 60]}
    Pass {} for strategies with no parameters (e.g. buy_hold).
train_years : int
    Length of each training window in years.
oos_years : int
    Length of each OOS window in years.
objective : str
    Metric to optimise in grid search. Only 'sharpe' is supported.

Output
------
dict — see run_wfa() docstring for full schema.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtester import backtest_signals
from engine.portfolio_backtester import backtest_portfolio
from metrics.calculator import calculate_all

PARAM_INSTABILITY_RATIO = 0.50   # flag if optimal range > 50% of grid range


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_wfa(
    price_data: pd.DataFrame,
    strategy: Any,
    param_grid: dict[str, list],
    train_years: int,
    oos_years: int,
    config: dict,
    objective: str = "sharpe",
    results_dir: str | Path | None = None,
) -> dict:
    """
    Run Walk-Forward Analysis.

    Returns
    -------
    dict with keys:
        per_window_results : list[dict]
            One dict per OOS window with keys:
              window_number, train_start, train_end, oos_start, oos_end,
              best_params, is_sharpe, is_cagr,
              oos_sharpe, oos_cagr, oos_max_drawdown, oos_win_rate,
              oos_profit_factor, oos_num_trades, oos_trades, oos_equity
        aggregate_results : dict
            equity_curve, full_trade_log,
            avg_oos_sharpe, avg_oos_cagr, worst_oos_drawdown,
            total_oos_trades,
            wfe, wfe_label,
            consistency_score, consistency_label,
            parameter_stability,
            warnings
    """
    wfa_cfg = config.get("wfa", {})
    max_params_warning   = int(wfa_cfg.get("max_params_warning", 3))
    is_sharpe_warning    = float(wfa_cfg.get("is_sharpe_warning", 3.0))

    wfa_warnings: list[str] = []

    # ── param count warning ───────────────────────────────────────────────────
    if len(param_grid) > max_params_warning:
        msg = (
            f"WARNING: High parameter count ({len(param_grid)} params) increases "
            "overfitting risk. Consider simplifying the strategy."
        )
        print(msg)
        wfa_warnings.append(msg)

    # ── results directory for grid search audit logs ──────────────────────────
    log_dir = Path(results_dir) if results_dir else (
        Path(__file__).resolve().parents[1] / "results" / "wfa_grid_search_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── detect mode and build rolling windows ─────────────────────────────────
    # price_data is a DataFrame (single mode) or {ticker: DataFrame} (portfolio).
    portfolio_mode = isinstance(price_data, dict)
    if portfolio_mode:
        full_index = None
        for _df in price_data.values():
            full_index = _df.index if full_index is None else full_index.union(_df.index)
        full_index = full_index.sort_values()
    else:
        full_index = price_data.index

    windows = _build_windows(full_index, train_years, oos_years)
    if not windows:
        raise ValueError(
            f"Insufficient data. Need at least {train_years + oos_years} years."
        )
    print(f"\nWFA: {len(windows)} windows  |  "
          f"train={train_years}yr  OOS={oos_years}yr  "
          f"param_combos={_count_combos(param_grid)}")

    # ── run each window ───────────────────────────────────────────────────────
    per_window_results: list[dict] = []
    running_capital = float(config["capital"]["starting_capital"])

    for i, (train_start, train_end, oos_start, oos_end) in enumerate(windows):
        win_n = i + 1

        # ── hard anti-lookahead assertion ─────────────────────────────────────
        assert train_end < oos_start, (
            f"CRITICAL: train_end ({train_end.date()}) is not strictly before "
            f"oos_start ({oos_start.date()}). Look-ahead bias would occur. Halting."
        )

        print(f"\n[Window {win_n}/{len(windows)}]  "
              f"Train {train_start.date()} → {train_end.date()}  |  "
              f"OOS  {oos_start.date()} → {oos_end.date()}")

        # Build data slices (pass full history up to each boundary for warmup)
        # and run grid search on the training window only.
        if portfolio_mode:
            train_slice_index = full_index[(full_index >= train_start) & (full_index <= train_end)]
            oos_slice_index   = full_index[(full_index >= oos_start)   & (full_index <= oos_end)]

            train_hist  = {t: df.loc[:train_end]            for t, df in price_data.items()}
            train_slice = {t: df.loc[train_start:train_end] for t, df in price_data.items()}
            oos_hist    = {t: df.loc[:oos_end]              for t, df in price_data.items()}
            oos_slice   = {t: df.loc[oos_start:oos_end]     for t, df in price_data.items()}

            best_params, is_sharpe, is_cagr, grid_log = _grid_search_portfolio(
                strategy, train_hist, train_slice, train_slice_index,
                param_grid, objective, config
            )
        else:
            train_history = price_data.loc[:train_end]
            oos_history   = price_data.loc[:oos_end]
            train_slice   = price_data.loc[train_start:train_end]
            oos_slice     = price_data.loc[oos_start:oos_end]

            best_params, is_sharpe, is_cagr, grid_log = _grid_search(
                strategy, train_history, train_slice, param_grid, objective, config
            )

        if is_sharpe > is_sharpe_warning:
            msg = (
                f"WARNING: [Window {win_n}] IS Sharpe={is_sharpe:.2f} — "
                "exceptionally high IS Sharpe may indicate overfitting."
            )
            print(f"  {msg}")
            wfa_warnings.append(msg)

        # Save grid search audit log (all param combos + IS Sharpe)
        pd.DataFrame(grid_log).to_csv(
            log_dir / f"window_{win_n:02d}_grid_search.csv", index=False
        )
        print(f"  Grid: {len(grid_log)} combos  "
              f"best_params={best_params}  IS_sharpe={is_sharpe:.4f}")

        # ── OOS backtest with best params ─────────────────────────────────────
        # T+1 entry is enforced by the backtester (pending order -> next open).
        # Do NOT shift here — that would delay entry to T+2.
        if portfolio_mode:
            max_pos   = int(config["capital"].get("max_positions", 6))
            slot_size = running_capital / max_pos
            print(f"  Slot size: ₹{slot_size:,.0f}  "
                  f"(window start capital ₹{running_capital:,.0f} / {max_pos} slots)")
            oos_sigs = strategy.generate_signals_universe(oos_hist, best_params)
            oos_sigs = {t: s.reindex(oos_slice_index).fillna(0) for t, s in oos_sigs.items()}
            oos_equity, oos_trades = backtest_portfolio(
                oos_slice, oos_sigs, config, strategy, starting_capital=running_capital
            )
        else:
            oos_signals_full = strategy.generate_signals(oos_history, best_params)
            assert oos_signals_full.index.equals(oos_history.index), \
                "Signal index must match data index"
            assert not oos_signals_full.isna().any(), \
                "Signals contain NaN values. Fill warmup NaN with 0 in your strategy."

            oos_signals = oos_signals_full.reindex(oos_slice.index).fillna(0)
            oos_equity, oos_trades = backtest_signals(
                oos_slice, oos_signals, config, initial_capital=running_capital
            )

        oos_metrics = calculate_all(oos_equity, oos_trades)

        per_window_results.append({
            "window_number":    win_n,
            "train_start":      train_start,
            "train_end":        train_end,
            "oos_start":        oos_start,
            "oos_end":          oos_end,
            "best_params":      best_params,
            "is_sharpe":        is_sharpe,
            "is_cagr":          is_cagr,
            "oos_sharpe":       oos_metrics["sharpe"],
            "oos_cagr":         oos_metrics["cagr"],
            "oos_max_drawdown": oos_metrics["max_drawdown"],
            "oos_win_rate":     oos_metrics["win_rate"],
            "oos_profit_factor":oos_metrics["profit_factor"],
            "oos_num_trades":   oos_metrics["num_trades"],
            "oos_trades":       oos_trades,
            "oos_equity":       oos_equity,
        })

        # Portfolio value carries forward into next window
        if not oos_equity.empty:
            running_capital = float(oos_equity.iloc[-1])

        print(f"  OOS  sharpe={oos_metrics['sharpe']:.4f}  "
              f"CAGR={oos_metrics['cagr']*100:.2f}%  "
              f"maxDD={oos_metrics['max_drawdown_pct']:.2f}%  "
              f"trades={oos_metrics['num_trades']}")

    # ── build aggregate results ───────────────────────────────────────────────
    aggregate = _build_aggregate(per_window_results, param_grid, config)
    aggregate["warnings"] = wfa_warnings

    return {
        "per_window_results": per_window_results,
        "aggregate_results":  aggregate,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Window generation
# ──────────────────────────────────────────────────────────────────────────────

def _build_windows(
    index: pd.DatetimeIndex,
    train_years: int,
    oos_years: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Generate (train_start, train_end, oos_start, oos_end) tuples.

    Uses calendar-year DateOffset to define boundaries, then maps to actual
    trading days in the index. OOS end clips to last available date rather
    than requiring a full calendar year (avoids discarding the final partial
    window when data ends mid-year).

    Advances by oos_years each iteration (rolling, not expanding).
    """
    windows = []
    train_start_date = index[0]

    while True:
        train_end_date = train_start_date + pd.DateOffset(years=train_years)
        oos_end_date   = train_end_date   + pd.DateOffset(years=oos_years)

        # Map calendar dates → positions in trading calendar
        ts_pos = int(index.searchsorted(train_start_date, side="left"))
        te_pos = int(index.searchsorted(train_end_date,   side="right")) - 1

        if te_pos < ts_pos:
            break  # Not enough data for training

        os_pos = te_pos + 1
        if os_pos >= len(index):
            break  # No OOS data exists at all

        # OOS end clips to last available data point
        oe_pos = min(
            int(index.searchsorted(oos_end_date, side="right")) - 1,
            len(index) - 1,
        )

        windows.append((
            index[ts_pos],
            index[te_pos],
            index[os_pos],
            index[oe_pos],
        ))

        train_start_date += pd.DateOffset(years=oos_years)

    return windows


# ──────────────────────────────────────────────────────────────────────────────
# Grid search
# ──────────────────────────────────────────────────────────────────────────────

def _grid_search(
    strategy: Any,
    train_history: pd.DataFrame,
    train_slice: pd.DataFrame,
    param_grid: dict[str, list],
    objective: str,
    config: dict,
) -> tuple[dict, float, float, list[dict]]:
    """
    Sweep all parameter combinations on the training slice.

    train_history : full price history up to train_end (for indicator warmup)
    train_slice   : the training window only (train_start → train_end)

    Returns
    -------
    (best_params, best_is_sharpe, best_is_cagr, grid_log)
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys])) if keys else [()]

    best_score   = float("-inf")
    best_params: dict = {}
    best_is_cagr = 0.0
    grid_log: list[dict] = []

    for combo in combos:
        params = dict(zip(keys, combo)) if keys else {}

        try:
            signals_full = strategy.generate_signals(train_history, params)
        except Exception as exc:
            grid_log.append({**params, "is_sharpe": float("nan"), "error": str(exc)})
            continue

        # T+1 entry is enforced by the backtester (pending order executes at the
        # next bar's open). Do NOT shift here — that would delay entry to T+2.
        assert signals_full.index.equals(train_history.index), \
            "Signal index must match data index"
        assert not signals_full.isna().any(), \
            "Signals contain NaN values. Fill warmup NaN with 0 in your strategy."

        signals = signals_full.reindex(train_slice.index).fillna(0)
        equity, trades = backtest_signals(train_slice, signals, config)
        metrics = calculate_all(equity, trades)

        score = float(metrics.get(objective, 0.0))
        if score != score:  # NaN → treat as zero
            score = 0.0

        grid_log.append({**params, "is_sharpe": score})

        if score > best_score:
            best_score   = score
            best_params  = params
            best_is_cagr = float(metrics.get("cagr", 0.0))

    # Edge case: every combo errored or param_grid is empty
    if not grid_log or (keys and not best_params):
        best_params  = dict(zip(keys, [param_grid[k][0] for k in keys])) if keys else {}
        best_score   = 0.0
        best_is_cagr = 0.0

    return best_params, float(best_score), best_is_cagr, grid_log


def _grid_search_portfolio(
    strategy: Any,
    train_hist:  dict[str, pd.DataFrame],
    train_slice: dict[str, pd.DataFrame],
    train_slice_index: pd.DatetimeIndex,
    param_grid: dict[str, list],
    objective: str,
    config: dict,
) -> tuple[dict, float, float, list[dict]]:
    """
    Portfolio-mode grid search. For each parameter combination, generate signals
    across ALL tickers (warmup on full history), run the portfolio backtester on
    the training window, and score the combined result. Mirrors _grid_search.
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys])) if keys else [()]

    best_score   = float("-inf")
    best_params: dict = {}
    best_is_cagr = 0.0
    grid_log: list[dict] = []

    for combo in combos:
        params = dict(zip(keys, combo)) if keys else {}

        try:
            sigs = strategy.generate_signals_universe(train_hist, params)
        except Exception as exc:
            grid_log.append({**params, "is_sharpe": float("nan"), "error": str(exc)})
            continue

        sigs = {t: s.reindex(train_slice_index).fillna(0) for t, s in sigs.items()}
        equity, trades = backtest_portfolio(train_slice, sigs, config, strategy)
        metrics = calculate_all(equity, trades)

        score = float(metrics.get(objective, 0.0))
        if score != score:  # NaN → zero
            score = 0.0

        grid_log.append({**params, "is_sharpe": score})

        if score > best_score:
            best_score   = score
            best_params  = params
            best_is_cagr = float(metrics.get("cagr", 0.0))

    if not grid_log or (keys and not best_params):
        best_params  = dict(zip(keys, [param_grid[k][0] for k in keys])) if keys else {}
        best_score   = 0.0
        best_is_cagr = 0.0

    return best_params, float(best_score), best_is_cagr, grid_log


# ──────────────────────────────────────────────────────────────────────────────
# Aggregate computation
# ──────────────────────────────────────────────────────────────────────────────

def _build_aggregate(
    per_window: list[dict],
    param_grid: dict[str, list],
    config: dict,
) -> dict:
    """Stitch OOS equity curves and compute aggregate metrics."""
    if not per_window:
        return {}

    # Stitch equity curve and trade log across all OOS windows
    equity_parts = [w["oos_equity"] for w in per_window if not w["oos_equity"].empty]
    trade_parts  = [w["oos_trades"] for w in per_window if not w["oos_trades"].empty]

    equity_curve = pd.concat(equity_parts) if equity_parts else pd.Series(dtype=float)
    full_trade_log = (
        pd.concat(trade_parts, ignore_index=True)
        if trade_parts
        else pd.DataFrame(
            columns=["entry_date", "exit_date", "entry_price",
                     "exit_price", "return_pct", "direction"]
        )
    )

    oos_sharpes   = [w["oos_sharpe"]       for w in per_window]
    oos_cagrs     = [w["oos_cagr"]         for w in per_window]
    oos_drawdowns = [w["oos_max_drawdown"] for w in per_window]
    oos_trades_n  = [w["oos_num_trades"]   for w in per_window]

    avg_oos_sharpe   = float(np.mean(oos_sharpes))
    avg_oos_cagr     = float(np.mean(oos_cagrs))
    worst_oos_dd     = float(np.max(oos_drawdowns))
    total_oos_trades = int(np.sum(oos_trades_n))

    wfe_val, wfe_label = _compute_wfe(per_window, config)
    cons_score, cons_label = _compute_consistency(per_window, config)
    param_stability = _compute_param_stability(per_window, param_grid)

    print("\n── Aggregate Results ────────────────────────────────────────────────")
    print(f"  Avg OOS Sharpe      : {avg_oos_sharpe:.4f}")
    print(f"  Avg OOS CAGR        : {avg_oos_cagr * 100:.2f}%")
    print(f"  Worst OOS Drawdown  : {worst_oos_dd * 100:.2f}%")
    print(f"  Total OOS Trades    : {total_oos_trades}")
    wfe_str = f"{wfe_val:.1f}%" if wfe_val is not None else "N/A"
    print(f"  WFE                 : {wfe_str}  ({wfe_label})")
    print(f"  Consistency         : {cons_score:.1f}%  ({cons_label})")
    print("─────────────────────────────────────────────────────────────────────")

    aggregate = {
        "equity_curve":        equity_curve,
        "full_trade_log":      full_trade_log,
        "avg_oos_sharpe":      avg_oos_sharpe,
        "avg_oos_cagr":        avg_oos_cagr,
        "worst_oos_drawdown":  worst_oos_dd,
        "total_oos_trades":    total_oos_trades,
        "wfe":                 wfe_val,
        "wfe_label":           wfe_label,
        "consistency_score":   cons_score,
        "consistency_label":   cons_label,
        "parameter_stability": param_stability,
    }
    # Buy-and-hold benchmark (^NSEI) over the same OOS span — report-only.
    aggregate.update(_compute_benchmark(equity_curve, config))
    return aggregate


def _compute_benchmark(equity_curve: pd.Series, config: dict) -> dict:
    """
    Buy-and-hold benchmark (^NSEI Nifty 50) over the same OOS span as the
    stitched strategy equity curve. Applies the same round-trip cost model.

    Returns a dict of benchmark keys; values are None if the curve is empty.
    On download failure, strategy figures are still returned and benchmark
    figures stay None (WFA never crashes on a benchmark error).
    """
    keys = {
        "benchmark_symbol":       None,
        "benchmark_label":        None,
        "benchmark_cagr":         None,
        "benchmark_total_return": None,
        "benchmark_sharpe":       None,
        "strategy_cagr":          None,
        "strategy_total_return":  None,
        "alpha_cagr":             None,
        "outperformed_benchmark": None,
    }
    if equity_curve is None or equity_curve.empty or len(equity_curve) < 2:
        return keys

    oos_start = equity_curve.index[0]
    oos_end   = equity_curve.index[-1]
    years = max((oos_end - oos_start).days / 365.25, 1e-9)

    s0 = float(equity_curve.iloc[0])
    s1 = float(equity_curve.iloc[-1])
    strat_total = s1 / s0 - 1.0
    strat_cagr  = (s1 / s0) ** (1.0 / years) - 1.0 if (s0 > 0 and s1 > 0) else float("nan")
    keys["strategy_total_return"] = float(strat_total)
    keys["strategy_cagr"]         = float(strat_cagr)

    data_cfg = config.get("data", {})
    primary  = str(data_cfg.get("benchmark_symbol", "NIFTYBEES.NS"))
    fallback = str(data_cfg.get("benchmark_symbol_fallback", "N100.NS") or "")
    costs    = config.get("costs", {})
    cost     = float(costs.get("brokerage", 0.001)) + float(costs.get("slippage", 0.0005))

    try:
        from data.fetcher import load_prices
        from configs.metrics_config import DAILY_RISK_FREE_RATE, TRADING_DAYS_PER_YEAR

        def _fetch_bench(sym_raw: str):
            """Fetch a benchmark OHLC frame over the OOS span. Returns None if no data."""
            try:
                panel, _ = load_prices(
                    tickers=[sym_raw],
                    start=str(oos_start.date()),
                    end=str((oos_end + pd.Timedelta(days=1)).date()),
                )
                sym = sym_raw.upper()
                bdf = pd.DataFrame(
                    {"open": panel.open[sym], "close": panel.close[sym]}, index=panel.dates
                ).dropna().loc[oos_start:oos_end]
                return bdf if len(bdf) >= 2 else None
            except Exception as fe:
                print(f"  [benchmark] {sym_raw} fetch failed: {fe}")
                return None

        # NIFTYBEES (Nifty 50 ETF, dividend-reinvesting) is a total-return proxy —
        # fair vs the strategy's dividend-adjusted stock prices. Fall back only if
        # the primary returns no data; if both fail, benchmark stays N/A (no crash).
        used_symbol = primary
        bdf = _fetch_bench(primary)
        if bdf is None and fallback and fallback != primary:
            print(f"  [benchmark] {primary} unavailable — trying fallback {fallback}.")
            bdf = _fetch_bench(fallback)
            used_symbol = fallback
        if bdf is None:
            print(f"  [benchmark] no data for {primary} or {fallback} — skipping benchmark.")
            return keys

        buy  = float(bdf["open"].iloc[0])  * (1.0 + cost)
        sell = float(bdf["close"].iloc[-1]) * (1.0 - cost)
        bench_total = sell / buy - 1.0
        bench_cagr  = (sell / buy) ** (1.0 / years) - 1.0 if (buy > 0 and sell > 0) else float("nan")

        daily = bdf["close"].pct_change().dropna()
        bench_sharpe = (
            float((daily.mean() - DAILY_RISK_FREE_RATE) / daily.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
            if daily.std() > 0 else 0.0
        )
        alpha = strat_cagr - bench_cagr

        keys.update({
            "benchmark_symbol":       used_symbol,
            "benchmark_label":        _benchmark_label(used_symbol),
            "benchmark_cagr":         float(bench_cagr),
            "benchmark_total_return": float(bench_total),
            "benchmark_sharpe":       bench_sharpe,
            "alpha_cagr":             float(alpha),
            "outperformed_benchmark": bool(alpha > 0),
        })
        print(
            f"  Benchmark {_benchmark_label(used_symbol)}: CAGR {bench_cagr*100:.2f}%  |  "
            f"Strategy CAGR {strat_cagr*100:.2f}%  |  Alpha {alpha*100:+.2f}%"
        )
    except Exception as e:
        print(f"  [benchmark] could not compute benchmark: {e}")

    return keys


def _benchmark_label(symbol: str) -> str:
    """Human-readable benchmark label for the report/dashboard."""
    s = (symbol or "").upper()
    return {
        "NIFTYBEES.NS": "Nifty 50 TRI (NIFTYBEES)",
        "N100.NS":      "Nifty 100 (N100)",
        "^NSEI":        "Nifty 50 (price)",
    }.get(s, symbol)


def _compute_wfe(per_window: list[dict], config: dict) -> tuple[float | None, str]:
    """
    Walk Forward Efficiency = avg_oos_cagr / avg_is_cagr × 100.

    Returns (wfe_pct, label). Returns (None, 'N/A') if IS CAGR is zero.
    """
    wfa_cfg    = config.get("wfa", {})
    robust_thr = float(wfa_cfg.get("wfe_robust_threshold",     0.60)) * 100.0
    accept_thr = float(wfa_cfg.get("wfe_acceptable_threshold", 0.40)) * 100.0

    is_cagrs  = [w["is_cagr"]  for w in per_window]
    oos_cagrs = [w["oos_cagr"] for w in per_window]

    avg_is  = float(np.mean(is_cagrs))
    avg_oos = float(np.mean(oos_cagrs))

    # WFE = OOS efficiency relative to in-sample. Only meaningful when the
    # in-sample CAGR is positive. If avg_is <= 0 the ratio is undefined or
    # sign-flips (e.g. avg_oos=-10% / avg_is=-5% = +200%, a FALSE "robust"),
    # so report N/A rather than a misleading number.
    if avg_is <= 0.0:
        return None, "N/A (IS CAGR ≤ 0 — no in-sample edge to measure against)"

    wfe = (avg_oos / avg_is) * 100.0

    if wfe >= robust_thr:
        label = "Robust"
    elif wfe >= accept_thr:
        label = "Acceptable"
    else:
        label = "Likely overfitted — review strategy"

    return float(wfe), label


def _compute_consistency(per_window: list[dict], config: dict) -> tuple[float, str]:
    """
    Consistency = profitable_windows / total_windows × 100.
    A window is profitable if oos_cagr > 0.
    """
    wfa_cfg    = config.get("wfa", {})
    high_thr   = float(wfa_cfg.get("consistency_threshold", 0.70)) * 100.0
    mid_thr    = 50.0   # fixed mid-point threshold

    total       = len(per_window)
    profitable  = sum(1 for w in per_window if w["oos_cagr"] > 0)
    score       = (profitable / total * 100.0) if total > 0 else 0.0

    if score >= high_thr:
        label = "Consistent"
    elif score >= mid_thr:
        label = "Moderate"
    else:
        label = "Inconsistent — strategy may not generalise"

    return float(score), label


def _compute_param_stability(
    per_window: list[dict],
    param_grid: dict[str, list],
) -> dict[str, dict]:
    """
    For each parameter, collect optimal values across windows and flag instability.

    Instability criterion: range of optima > 50% of the grid range for that parameter.

    Returns dict keyed by param name:
        optimal_values : list of values chosen per window
        grid_min, grid_max : bounds of the grid
        optima_min, optima_max : observed range of chosen values
        stable : bool
    """
    if not param_grid:
        return {}

    stability: dict[str, dict] = {}

    for param, grid_values in param_grid.items():
        grid_min = min(grid_values)
        grid_max = max(grid_values)
        grid_range = grid_max - grid_min

        optima = []
        for w in per_window:
            v = w["best_params"].get(param)
            if v is not None:
                optima.append(v)

        if not optima:
            stability[param] = {
                "optimal_values": [],
                "grid_min": grid_min,
                "grid_max": grid_max,
                "optima_min": None,
                "optima_max": None,
                "stable": True,
            }
            continue

        optima_min = min(optima)
        optima_max = max(optima)
        optima_range = optima_max - optima_min

        is_stable = (grid_range == 0) or (optima_range / grid_range <= PARAM_INSTABILITY_RATIO)

        if not is_stable:
            print(
                f"WARNING: Parameter '{param}' is unstable across windows "
                f"(range={optima_range} on grid_range={grid_range}). "
                "This suggests curve fitting."
            )

        stability[param] = {
            "optimal_values": optima,
            "grid_min":       grid_min,
            "grid_max":       grid_max,
            "optima_min":     optima_min,
            "optima_max":     optima_max,
            "stable":         is_stable,
        }

    return stability


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _count_combos(param_grid: dict[str, list]) -> int:
    if not param_grid:
        return 1
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n
