"""
Parameter sensitivity analysis for FALSIFY.

Sweeps the full parameter grid on complete historical data, measures how
sensitive Sharpe is to parameter choice, and distinguishes robust plateaus
from fragile spikes. Does NOT replace WFA — this is an in-sample landscape
characterisation, not an OOS test.

Entry point: run_sensitivity(strategy_class, data, param_grid,
                              optimal_params, config) -> dict
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtester import backtest_signals
from engine.portfolio_backtester import backtest_portfolio
from metrics.calculator import calculate_all
from configs.metrics_config import TRADING_DAYS_PER_YEAR


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_sensitivity(
    strategy_class,
    data:           pd.DataFrame,
    param_grid:     dict,
    optimal_params: dict,
    config:         dict,
) -> dict:
    """
    Run full parameter sensitivity analysis on complete historical data.

    Parameters
    ----------
    strategy_class  : strategy class (not instance). Must implement
                      generate_signals(data, params) -> pd.Series.
    data            : full OHLCV DataFrame from fetcher.py.
    param_grid      : {param_name: [list of values to test]}
    optimal_params  : best params from WFA (used as baseline for 1D analysis)
    config          : settings dict — see module docstring for expected keys.

    Returns
    -------
    dict — see return section at bottom of function.
    """
    # ── Extract config ─────────────────────────────────────────────────────────
    sens_cfg           = config.get("sensitivity", {})
    robustness_thresh  = float(sens_cfg.get("robustness_threshold", 0.20))
    min_trades         = int(sens_cfg.get("min_trades", 10))

    # ── Guard checks ──────────────────────────────────────────────────────────
    if len(param_grid) == 0:
        raise ValueError("param_grid is empty.")

    if len(param_grid) > 3:
        print(
            f"WARNING: More than 3 parameters in grid ({len(param_grid)}). "
            "Visualisation will only cover pairwise combinations. "
            "Results may be hard to interpret. Consider simplifying."
        )

    total_combos = math.prod(len(v) for v in param_grid.values())
    if total_combos > 10_000:
        print(
            f"WARNING: Grid has {total_combos:,} combinations. "
            "This may be slow. Consider reducing grid resolution."
        )

    missing_keys = [k for k in optimal_params if k not in param_grid]
    if missing_keys:
        raise ValueError(
            f"optimal_params keys not found in param_grid: {missing_keys}"
        )

    # ── Build combination list ─────────────────────────────────────────────────
    param_names = list(param_grid.keys())
    value_lists = [param_grid[k] for k in param_names]
    all_combos  = [
        dict(zip(param_names, combo))
        for combo in itertools.product(*value_lists)
    ]

    portfolio_mode = isinstance(data, dict)
    if portfolio_mode:
        sample_tickers = list(data.keys())[:20]
        sample_prices  = {t: data[t] for t in sample_tickers}
        data_desc = f"{len(sample_tickers)}/{len(data)} tickers (sample)"
        print(
            f"Sensitivity (portfolio mode): testing {total_combos} combinations "
            f"across {len(sample_tickers)}-ticker sample"
        )
    else:
        data_desc = f"{len(data):,} rows"

    print(
        f"\n{'═'*50}\n"
        f"PARAMETER SENSITIVITY — FALSIFY\n"
        f"{'═'*50}\n"
        f"Running on FULL HISTORICAL DATA (in-sample).\n"
        f"This is a landscape characterisation, not an OOS test.\n"
        f"Total combinations: {total_combos:,}  |  Data: {data_desc}\n"
        f"{'─'*50}"
    )

    # ── Main sweep loop ────────────────────────────────────────────────────────
    milestone       = max(1, len(all_combos) // 10)
    records: list[dict] = []
    strategy_obj    = strategy_class()

    for idx, params in enumerate(all_combos):
        if portfolio_mode:
            row = _run_single_combo_portfolio(
                strategy_obj, sample_prices, params, config, min_trades
            )
        else:
            row = _run_single_combo(
                strategy_obj, data, params, config, min_trades
            )
        row.update(params)
        records.append(row)

        completed = idx + 1
        if completed % milestone == 0 or completed == len(all_combos):
            pct = completed / len(all_combos) * 100
            print(f"  Progress: {pct:.0f}% ({completed}/{len(all_combos)} combinations tested)")

    results_df = pd.DataFrame(records)

    # ── Peak and optimal metrics ───────────────────────────────────────────────
    valid_df = results_df.dropna(subset=["sharpe"])
    n_valid  = len(valid_df)

    if n_valid == 0:
        raise ValueError(
            "No combinations produced enough trades to compute Sharpe. "
            "Check min_trades threshold or strategy logic."
        )

    peak_sharpe        = float(valid_df["sharpe"].max())
    peak_idx           = valid_df["sharpe"].idxmax()
    peak_sharpe_params = {p: results_df.loc[peak_idx, p] for p in param_names}

    # Sharpe at WFA optimal params exactly
    optimal_mask = results_df[param_names].eq(
        pd.Series({p: optimal_params[p] for p in param_names})
    ).all(axis=1)
    optimal_rows = results_df[optimal_mask]
    optimal_sharpe = float(optimal_rows["sharpe"].iloc[0]) if not optimal_rows.empty else float("nan")

    # ── Robustness score ───────────────────────────────────────────────────────
    # Guard: a non-positive peak Sharpe has no meaningful plateau. Multiplying a
    # negative/zero peak by (1 - thresh) inverts the band (the threshold ends up
    # ABOVE the peak), which can report a misleadingly high robustness when the
    # peak is ≈ 0. Force robustness to 0 in that regime.
    if peak_sharpe <= 0.0:
        sharpe_threshold = peak_sharpe
        robustness_score = 0.0
    else:
        sharpe_threshold = peak_sharpe * (1.0 - robustness_thresh)
        above_threshold  = (valid_df["sharpe"] >= sharpe_threshold).sum()
        robustness_score = float(above_threshold / n_valid * 100.0)

    # ── Per-parameter 1D analysis ──────────────────────────────────────────────
    plateau_analysis: dict[str, dict] = {}
    for param_name in param_names:
        plateau_analysis[param_name] = _compute_1d_analysis(
            results_df, param_name, param_grid, optimal_params,
            peak_sharpe, robustness_thresh
        )

    # ── Parameter importance (Spearman) ───────────────────────────────────────
    param_importance = _compute_param_importance(valid_df, param_names)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, verdict_detail = _get_verdict(robustness_score)

    # ── Warning flags ──────────────────────────────────────────────────────────
    edge_case_warnings = _check_edge_cases(param_grid, optimal_params)
    edge_case_warning  = len(edge_case_warnings) > 0
    low_peak_warning   = peak_sharpe < 0.5

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_dir = Path(__file__).resolve().parents[1] / config["reporting"]["charts_dir"]
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths: list[str] = []

    heatmap_path = _chart_heatmap(
        results_df, param_grid, param_names,
        optimal_params, peak_sharpe_params, charts_dir
    )
    if heatmap_path:
        chart_paths.append(heatmap_path)

    lines_path = _chart_1d_lines(
        results_df, param_grid, param_names,
        optimal_params, peak_sharpe, robustness_thresh, charts_dir
    )
    chart_paths.append(lines_path)

    plateau_path = _chart_plateau_bars(
        plateau_analysis, param_grid, charts_dir
    )
    chart_paths.append(plateau_path)

    dist_path = _chart_sharpe_distribution(
        results_df, peak_sharpe, optimal_sharpe,
        sharpe_threshold, robustness_score, charts_dir
    )
    chart_paths.append(dist_path)

    # ── Console summary ───────────────────────────────────────────────────────
    _print_summary(
        total_combos, n_valid, min_trades,
        peak_sharpe, peak_sharpe_params,
        optimal_sharpe, robustness_score, verdict,
        plateau_analysis, param_grid, param_importance,
        edge_case_warnings, low_peak_warning,
    )

    return {
        "results_df":         results_df,
        "peak_sharpe":        peak_sharpe,
        "peak_sharpe_params": peak_sharpe_params,
        "optimal_sharpe":     optimal_sharpe,
        "robustness_score":   robustness_score,
        "plateau_analysis":   plateau_analysis,
        "param_importance":   param_importance,
        "verdict":            verdict,
        "verdict_detail":     verdict_detail,
        "edge_case_warning":  edge_case_warning,
        "low_peak_warning":   low_peak_warning,
        "chart_paths":        chart_paths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Simulation helper
# ──────────────────────────────────────────────────────────────────────────────

def _run_single_combo(
    strategy_obj,
    data:       pd.DataFrame,
    params:     dict,
    config:     dict,
    min_trades: int,
) -> dict:
    """
    Run one parameter combination on the full historical data.
    Returns a dict with sharpe, cagr, max_drawdown, win_rate,
    profit_factor, num_trades. Sets sharpe=NaN if num_trades < min_trades.
    """
    _EMPTY = {
        "sharpe": float("nan"), "cagr": float("nan"),
        "max_drawdown": float("nan"), "win_rate": float("nan"),
        "profit_factor": float("nan"), "num_trades": 0,
    }
    try:
        signals = strategy_obj.generate_signals(data, params)
        signals = signals.reindex(data.index).fillna(0)
        equity, trades = backtest_signals(data, signals, config)

        if equity.empty:
            return _EMPTY

        n_trades = len(trades)
        if n_trades < min_trades:
            return {**_EMPTY, "num_trades": n_trades}

        m = calculate_all(equity, trades)
        return {
            "sharpe":       float(m["sharpe"]),
            "cagr":         float(m["cagr"]),
            "max_drawdown": float(m["max_drawdown"]),
            "win_rate":     float(m["win_rate"]),
            "profit_factor": float(m["profit_factor"]),
            "num_trades":   int(m["num_trades"]),
        }
    except Exception:
        return _EMPTY


def _run_single_combo_portfolio(
    strategy_obj,
    sample_prices: dict,
    params:        dict,
    config:        dict,
    min_trades:    int,
) -> dict:
    """
    Portfolio-mode variant of _run_single_combo: run one parameter combination
    across the sampled universe via the portfolio backtester. Same return shape.
    """
    _EMPTY = {
        "sharpe": float("nan"), "cagr": float("nan"),
        "max_drawdown": float("nan"), "win_rate": float("nan"),
        "profit_factor": float("nan"), "num_trades": 0,
    }
    try:
        sigs = strategy_obj.generate_signals_universe(sample_prices, params)
        equity, trades = backtest_portfolio(sample_prices, sigs, config, strategy_obj)

        if equity.empty:
            return _EMPTY

        n_trades = len(trades)
        if n_trades < min_trades:
            return {**_EMPTY, "num_trades": n_trades}

        m = calculate_all(equity, trades)
        return {
            "sharpe":       float(m["sharpe"]),
            "cagr":         float(m["cagr"]),
            "max_drawdown": float(m["max_drawdown"]),
            "win_rate":     float(m["win_rate"]),
            "profit_factor": float(m["profit_factor"]),
            "num_trades":   int(m["num_trades"]),
        }
    except Exception:
        return _EMPTY


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compute_1d_analysis(
    results_df:       pd.DataFrame,
    param_name:       str,
    param_grid:       dict,
    optimal_params:   dict,
    peak_sharpe:      float,
    robustness_thresh: float,
) -> dict:
    """
    Hold all other parameters at their optimal values, vary this one.
    Returns plateau_width, grid_size, plateau_pct, is_edge_case,
    degradation_gradient, and the 1D series for charting.
    """
    param_names = list(param_grid.keys())
    other_params = {p: optimal_params[p] for p in param_names if p != param_name}

    # Filter to rows where all other params equal optimal
    mask = pd.Series(True, index=results_df.index)
    for p, v in other_params.items():
        mask &= (results_df[p] == v)

    subset = results_df[mask].sort_values(param_name).copy()
    grid_values  = param_grid[param_name]
    grid_size    = len(grid_values)
    threshold    = peak_sharpe * (1.0 - robustness_thresh)

    if subset.empty or subset["sharpe"].isna().all():
        return {
            "plateau_width":         0,
            "grid_size":             grid_size,
            "plateau_pct":           0.0,
            "is_edge_case":          False,
            "degradation_gradient":  float("nan"),
            "sharpe_by_value":       {},
            "optimal_value":         optimal_params.get(param_name),
        }

    # Plateau: values where sharpe >= threshold (excluding NaN)
    valid_sub    = subset.dropna(subset=["sharpe"])
    above        = (valid_sub["sharpe"] >= threshold).sum()
    plateau_width = int(above)
    plateau_pct  = float(plateau_width / grid_size * 100.0)

    # Edge case: optimal value is at the boundary of its grid range
    opt_val      = optimal_params[param_name]
    is_edge_case = (opt_val <= min(grid_values) or opt_val >= max(grid_values))

    # Degradation gradient: avg |Δsharpe| per unit Δparam away from optimal
    opt_rows = valid_sub[valid_sub[param_name] == opt_val]
    if not opt_rows.empty:
        opt_sharpe  = float(opt_rows["sharpe"].iloc[0])
        gradients   = []
        for _, row in valid_sub.iterrows():
            v = row[param_name]
            delta_param = abs(float(v) - float(opt_val))
            if delta_param == 0.0:
                continue
            delta_sharpe = abs(float(row["sharpe"]) - opt_sharpe)
            gradients.append(delta_sharpe / delta_param)
        degradation_gradient = float(np.mean(gradients)) if gradients else float("nan")
    else:
        degradation_gradient = float("nan")

    # Build sharpe_by_value dict for charting
    sharpe_by_value = dict(
        zip(
            valid_sub[param_name].astype(float).tolist(),
            valid_sub["sharpe"].tolist(),
        )
    )

    return {
        "plateau_width":        plateau_width,
        "grid_size":            grid_size,
        "plateau_pct":          plateau_pct,
        "is_edge_case":         is_edge_case,
        "degradation_gradient": degradation_gradient,
        "sharpe_by_value":      sharpe_by_value,
        "optimal_value":        optimal_params.get(param_name),
    }


def _compute_param_importance(valid_df: pd.DataFrame, param_names: list) -> dict:
    """Spearman rank correlation of each parameter with Sharpe."""
    importance: dict[str, float] = {}
    for p in param_names:
        if p not in valid_df.columns or len(valid_df) < 3:
            importance[p] = float("nan")
            continue
        x = valid_df[p].astype(float)
        y = valid_df["sharpe"].astype(float)
        # Drop rows where either is NaN
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            importance[p] = float("nan")
            continue
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")   # suppress ConstantInputWarning
            r, _ = spearmanr(x[mask], y[mask])
        # spearmanr returns NaN when input is constant — handled cleanly
        importance[p] = float(r) if r == r else float("nan")
    return importance


def _get_verdict(robustness_score: float) -> tuple[str, str]:
    if robustness_score >= 50.0:
        return (
            "ROBUST",
            "Wide plateau. Strategy does not depend on precise parameter tuning.",
        )
    elif robustness_score >= 25.0:
        return (
            "MARGINAL",
            "Narrow plateau. Performance is somewhat sensitive to parameter choice. "
            "Trade with reduced size.",
        )
    else:
        return (
            "FRAGILE",
            "Spike detected. Strategy performance depends heavily on specific "
            "parameter values. Likely overfitted.",
        )


def _check_edge_cases(param_grid: dict, optimal_params: dict) -> list[str]:
    """Return list of warning strings for params whose optimal sits at grid boundary."""
    warnings: list[str] = []
    for param, values in param_grid.items():
        if param not in optimal_params:
            continue
        opt = optimal_params[param]
        if opt <= min(values) or opt >= max(values):
            warnings.append(
                f"Optimal value for '{param}' is at grid boundary "
                f"(optimal={opt}, grid=[{min(values)}, {max(values)}]). "
                "Widen the search range — true optimum may be outside."
            )
    return warnings


# ──────────────────────────────────────────────────────────────────────────────
# Console print
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(
    total_combos:       int,
    n_valid:            int,
    min_trades:         int,
    peak_sharpe:        float,
    peak_sharpe_params: dict,
    optimal_sharpe:     float,
    robustness_score:   float,
    verdict:            str,
    plateau_analysis:   dict,
    param_grid:         dict,
    param_importance:   dict,
    edge_case_warnings: list,
    low_peak_warning:   bool,
) -> None:
    SEP  = "═" * 50
    THIN = "─" * 50

    print(f"\n{SEP}")
    print("PARAMETER SENSITIVITY RESULTS — FALSIFY")
    print(SEP)
    print(f"Total combinations tested:       {total_combos:>6,}")
    print(f"Valid combinations (≥{min_trades} trades): {n_valid:>6,}")
    print()
    print(f"Peak Sharpe (anywhere in grid):  {peak_sharpe:.2f}")
    print(f"Peak Sharpe params:              {peak_sharpe_params}")
    opt_str = f"{optimal_sharpe:.2f}" if optimal_sharpe == optimal_sharpe else "N/A"
    print(f"Sharpe at WFA optimal params:    {opt_str}")
    print()
    print(f"Robustness Score:                {robustness_score:.1f}%")
    print(f"Verdict:                         {verdict}")
    print(THIN)
    print("PER-PARAMETER PLATEAU ANALYSIS")
    print(THIN)

    # Header row
    col1_w = max(12, max(len(p) for p in plateau_analysis))
    print(f"{'Parameter':<{col1_w}}  {'Optimal':>10}  {'Plateau Width':>14}  {'Edge Case?':>10}")
    for param, pa in plateau_analysis.items():
        opt_val = pa.get("optimal_value", "?")
        pw      = pa["plateau_width"]
        gs      = pa["grid_size"]
        edge_str = "Yes ⚠" if pa["is_edge_case"] else "No"
        print(
            f"{param:<{col1_w}}  {str(opt_val):>10}  "
            f"{pw}/{gs} values    {edge_str:>10}"
        )

    print(THIN)
    print("PARAMETER IMPORTANCE (Spearman correlation w/ Sharpe)")
    print(THIN)

    for param, r in param_importance.items():
        if r != r:  # NaN
            print(f"  {param}:   r = N/A")
            continue
        abs_r = abs(r)
        if abs_r >= 0.6:
            label = "high influence"
        elif abs_r >= 0.3:
            label = "moderate influence"
        else:
            label = "low influence"
        print(f"  {param}:   r = {r:.2f}  ({label})")

    print(THIN)
    if edge_case_warnings:
        for w in edge_case_warnings:
            print(f"⚠ EDGE CASE WARNING: {w}")
    if low_peak_warning:
        print("⚠ LOW PEAK WARNING: peak Sharpe < 0.5")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────

_COLOR_LINE    = "#4A90D9"
_COLOR_THRESH  = "#E8735A"
_COLOR_OPTIMAL = "#FFFFFF"
_COLOR_PEAK    = "#FFD700"   # gold star for grid peak


def _chart_heatmap(
    results_df:         pd.DataFrame,
    param_grid:         dict,
    param_names:        list,
    optimal_params:     dict,
    peak_sharpe_params: dict,
    charts_dir:         Path,
) -> str | None:
    """
    2D heatmap for each pair of parameters.
    Returns save path or None if only 1 parameter.
    """
    n_params = len(param_names)
    if n_params < 2:
        return None

    # All unique pairs
    pairs = [
        (param_names[i], param_names[j])
        for i in range(n_params)
        for j in range(i + 1, n_params)
    ]

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(1, len(pairs), figsize=(8 * len(pairs), 7),
                                 squeeze=False)

        for col_idx, (p1, p2) in enumerate(pairs):
            ax = axes[0][col_idx]

            # For 3+ params, hold other params at their optimal value
            other_params = [p for p in param_names if p not in (p1, p2)]
            mask = pd.Series(True, index=results_df.index)
            for op in other_params:
                mask &= (results_df[op] == optimal_params.get(op))

            subset = results_df[mask]

            if subset.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="white")
                continue

            # Pivot for heatmap: p2 on y-axis (rows), p1 on x-axis (columns)
            pivot = subset.pivot_table(
                values="sharpe", index=p2, columns=p1, aggfunc="mean"
            )

            sharpe_vals = pivot.values.astype(float)
            # Mask NaN for imshow
            masked_vals = np.ma.masked_invalid(sharpe_vals)
            cmap = plt.get_cmap("RdYlGn").copy()
            cmap.set_bad("gray", alpha=0.3)

            im = ax.imshow(
                masked_vals,
                cmap=cmap,
                vmin=-0.5, vmax=2.0,
                aspect="auto",
                origin="lower",
            )

            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(
                [str(v) for v in pivot.columns], rotation=45, ha="right", fontsize=8
            )
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(v) for v in pivot.index], fontsize=8)
            ax.set_xlabel(p1)
            ax.set_ylabel(p2)
            ax.set_title(
                f"Parameter Sensitivity — {p1} vs {p2}\n(Sharpe Heatmap)",
                fontsize=11,
            )

            plt.colorbar(im, ax=ax, label="Sharpe Ratio", shrink=0.8)

            # Mark WFA optimal params (white star)
            _mark_heatmap_point(ax, pivot, p1, p2, optimal_params, "white", "*", 16, "WFA Optimal")
            # Mark grid peak params (gold star) — only if different from optimal
            if peak_sharpe_params != {p: optimal_params.get(p) for p in peak_sharpe_params}:
                _mark_heatmap_point(ax, pivot, p1, p2, peak_sharpe_params, _COLOR_PEAK, "*", 16, "Grid Peak")

            ax.legend(
                handles=[
                    plt.Line2D([0], [0], marker="*", color="w", markersize=10,
                                markerfacecolor="white", label="WFA Optimal"),
                    plt.Line2D([0], [0], marker="*", color="w", markersize=10,
                                markerfacecolor=_COLOR_PEAK, label="Grid Peak"),
                ],
                loc="upper right", fontsize=7,
            )

        fig.suptitle("Parameter Sensitivity Heatmap", fontsize=13, y=1.01)
        plt.tight_layout()
        path = charts_dir / "sensitivity_heatmap.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return str(path)


def _mark_heatmap_point(
    ax, pivot, p1: str, p2: str,
    params: dict, color: str, marker: str, size: int, label: str
) -> None:
    """Mark a (p1, p2) point on the heatmap if both values exist in pivot."""
    if p1 not in params or p2 not in params:
        return
    v1 = params[p1]
    v2 = params[p2]
    col_vals = list(pivot.columns)
    row_vals = list(pivot.index)
    if v1 in col_vals and v2 in row_vals:
        x = col_vals.index(v1)
        y = row_vals.index(v2)
        ax.plot(x, y, marker=marker, color=color, markersize=size,
                zorder=6, linestyle="none", label=label)


def _chart_1d_lines(
    results_df:       pd.DataFrame,
    param_grid:       dict,
    param_names:      list,
    optimal_params:   dict,
    peak_sharpe:      float,
    robustness_thresh: float,
    charts_dir:       Path,
) -> str:
    """
    One subplot per parameter: Sharpe vs parameter value.
    Others held at optimal. Shows threshold band and optimal marker.
    """
    n_params   = len(param_names)
    ncols      = min(n_params, 3)
    nrows      = math.ceil(n_params / ncols)
    threshold  = peak_sharpe * (1.0 - robustness_thresh)

    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(6 * ncols, 5 * nrows),
            squeeze=False,
        )

        for idx, param_name in enumerate(param_names):
            row_idx = idx // ncols
            col_idx = idx % ncols
            ax      = axes[row_idx][col_idx]

            # Get 1D data: hold other params at optimal, vary this one
            other_params = {p: optimal_params[p] for p in param_names if p != param_name}
            mask = pd.Series(True, index=results_df.index)
            for p, v in other_params.items():
                mask &= (results_df[p] == v)
            subset = results_df[mask].sort_values(param_name)
            valid  = subset.dropna(subset=["sharpe"])

            if not valid.empty:
                x = valid[param_name].astype(float).values
                y = valid["sharpe"].values

                ax.plot(x, y, color=_COLOR_LINE, linewidth=2.0, marker="o",
                        markersize=5, label="Sharpe")

                # Threshold line and shading
                ax.axhline(threshold, color=_COLOR_THRESH, linestyle="--",
                           linewidth=1.5, label=f"Threshold ({threshold:.2f})")
                ax.fill_between(x, threshold, y,
                                where=(y >= threshold),
                                color="green", alpha=0.1, label="Above threshold")

            # Optimal value marker
            opt_val = optimal_params.get(param_name)
            if opt_val is not None:
                ax.axvline(float(opt_val), color=_COLOR_OPTIMAL, linestyle="--",
                           linewidth=1.5, label=f"Optimal ({opt_val})")

            ax.set_xlabel(param_name)
            ax.set_ylabel("Sharpe Ratio")
            ax.set_title(f"Sensitivity — {param_name}", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.2)

        # Hide unused subplots
        for idx in range(n_params, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle("1D Parameter Sensitivity Analysis", fontsize=13)
        plt.tight_layout()
        path = charts_dir / "sensitivity_1d_lines.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return str(path)


def _chart_plateau_bars(
    plateau_analysis: dict,
    param_grid:       dict,
    charts_dir:       Path,
) -> str:
    """Horizontal bar chart: plateau width per parameter, coloured by robustness tier."""
    param_names = list(plateau_analysis.keys())
    pcts    = [plateau_analysis[p]["plateau_pct"]   for p in param_names]
    widths  = [plateau_analysis[p]["plateau_width"] for p in param_names]
    sizes   = [plateau_analysis[p]["grid_size"]     for p in param_names]

    def _bar_color(pct: float) -> str:
        if pct >= 50.0:
            return "#4CAF50"   # green
        elif pct >= 25.0:
            return "#FFC107"   # yellow
        else:
            return "#F44336"   # red

    colors = [_bar_color(p) for p in pcts]

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, max(3, len(param_names) * 1.2)))

        y_pos = np.arange(len(param_names))
        # Use PERCENTAGE x-axis so the 50% threshold line is meaningful for all params
        ax.barh(y_pos, pcts, color=colors, height=0.5)

        # Single 50% threshold line — valid for all params since x-axis is in %
        ax.axvline(50.0, color=_COLOR_THRESH, linestyle="--",
                   linewidth=1.5, alpha=0.8, label="50% threshold (robust)")
        ax.axvline(25.0, color="#FFC107", linestyle=":",
                   linewidth=1.2, alpha=0.6, label="25% threshold (marginal)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel("% of Parameter Values Within Robustness Threshold")
        ax.set_xlim(0, 110)
        ax.set_title("Parameter Plateau Width by Parameter", fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="x")

        # Annotate bar values as w/size
        for i, (w, s, pct) in enumerate(zip(widths, sizes, pcts)):
            ax.text(pct + 1.0, i, f"{w}/{s}", va="center", fontsize=9, color="white")

        plt.tight_layout()
        path = charts_dir / "sensitivity_plateau_bars.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return str(path)


def _chart_sharpe_distribution(
    results_df:       pd.DataFrame,
    peak_sharpe:      float,
    optimal_sharpe:   float,
    sharpe_threshold: float,
    robustness_score: float,
    charts_dir:       Path,
) -> str:
    """Histogram of all valid Sharpe values across the grid."""
    valid_sharpes = results_df["sharpe"].dropna().values

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(10, 6))

        if len(valid_sharpes) > 0:
            # Guard degenerate range (all sharpes identical)
            s_min, s_max = valid_sharpes.min(), valid_sharpes.max()
            if s_max - s_min < 1e-10:
                pad = max(abs(s_min) * 0.2, 0.5)
                bins = np.linspace(s_min - pad, s_max + pad, 20)
            else:
                bins = min(40, max(10, len(valid_sharpes) // 5))

            ax.hist(valid_sharpes, bins=bins, color=_COLOR_LINE, alpha=0.7, label="Sharpe values")

            # Threshold shading
            ax.axvline(sharpe_threshold, color="green", linestyle="--",
                       linewidth=1.5, label=f"Threshold ({sharpe_threshold:.2f})")

            # Peak sharpe
            ax.axvline(peak_sharpe, color=_COLOR_OPTIMAL, linestyle="-",
                       linewidth=2.0, label=f"Peak Sharpe ({peak_sharpe:.2f})")

            # Optimal params Sharpe
            if optimal_sharpe == optimal_sharpe:   # not NaN
                ax.axvline(optimal_sharpe, color=_COLOR_THRESH, linestyle="--",
                           linewidth=1.5, label=f"WFA Optimal ({optimal_sharpe:.2f})")

            # Shade area above threshold up to 1 unit beyond peak
            ax.axvspan(sharpe_threshold, peak_sharpe + 1.0, alpha=0.08, color="green")

            # Annotate robustness score
            ax.text(0.97, 0.95, f"Robustness: {robustness_score:.1f}%",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=11, color=_COLOR_OPTIMAL,
                    bbox=dict(boxstyle="round", facecolor="#222222", alpha=0.8))

        ax.set_xlabel("Sharpe Ratio")
        ax.set_ylabel("Number of Combinations")
        ax.set_title("Distribution of Sharpe Across Parameter Grid", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)

        plt.tight_layout()
        path = charts_dir / "sensitivity_sharpe_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return str(path)
