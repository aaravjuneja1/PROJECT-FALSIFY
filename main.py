"""
FALSIFY — main pipeline orchestrator.

Runs all validation steps in sequence, passing outputs forward.
A single module failure does not crash the pipeline (except pre-check).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    path = ROOT / "config.yml"
    if not path.exists():
        raise FileNotFoundError(f"config.yml not found at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _fetch_data(config: dict) -> pd.DataFrame:
    """
    Load price data from Yahoo Finance via data/fetcher.py.
    Returns a single-ticker OHLCV DataFrame matching the symbol in config.
    """
    from data.fetcher import load_prices

    symbol     = config["data"]["symbol"]
    start_date = config["data"]["start_date"]
    end_date   = config["data"]["end_date"]

    panel, report = load_prices(
        tickers=[symbol],
        start=str(start_date),
        end=str(end_date),
    )
    if report.warnings:
        for w in report.warnings:
            print(f"  [data warning] {w}")

    # Extract single-ticker OHLCV DataFrame from PricePanel
    df = pd.DataFrame({
        "open":      panel.open[symbol],
        "high":      panel.high[symbol],
        "low":       panel.low[symbol],
        "close":     panel.close[symbol],
        "volume":    panel.volume[symbol],
        "adj_close": panel.adj_close[symbol],
    }, index=panel.dates)

    return df


def _fetch_universe(config: dict) -> dict:
    """
    Portfolio mode: resolve the Nifty 200 universe and load OHLCV for every
    ticker. Returns {ticker: OHLCV DataFrame}, outer-aligned (NaN where a
    ticker has no data on a given day).
    """
    from data.universe import get_nifty200_universe
    from data.fetcher import load_universe_prices

    data_cfg = config["data"]
    tickers  = get_nifty200_universe(config)

    limit = data_cfg.get("universe_limit")
    if limit:
        tickers = tickers[: int(limit)]
        print(f"  (universe_limit={limit}: using first {len(tickers)} tickers)")

    panel, report = load_universe_prices(
        tickers=tickers,
        start=str(data_cfg["start_date"]),
        end=str(data_cfg["end_date"]),
        min_bars=int(data_cfg.get("min_ticker_bars", 250)),
    )
    if report.warnings:
        for w in report.warnings[:5]:
            print(f"  [data warning] {w}")

    prices: dict = {}
    for t in panel.tickers:
        prices[t] = pd.DataFrame({
            "open":      panel.open[t],
            "high":      panel.high[t],
            "low":       panel.low[t],
            "close":     panel.close[t],
            "volume":    panel.volume[t],
            "adj_close": panel.adj_close[t],
        }, index=panel.dates)
    return prices


def _optimal_params_from_wfa(wfa_results: dict) -> dict:
    """
    Derive a single optimal_params dict from WFA windows.
    Uses the last window's best_params as a representative value.
    Returns {} if no windows exist.
    """
    windows = wfa_results.get("per_window_results", [])
    if not windows:
        return {}
    return dict(windows[-1].get("best_params") or {})


def _print_banner(
    step_statuses: dict[str, str | None],
    verdict_results: dict | None,
    sizing_results:  dict | None,
) -> None:
    SEP = "══════════════════════════════════════"
    print(f"\n{SEP}")
    print("FALSIFY PIPELINE COMPLETE")
    print(SEP)
    for step, status in step_statuses.items():
        if status is None or status == "success":
            icon = "✓"
            line = f"{icon} {step}"
        elif status == "pending":
            icon = "—"
            line = f"{icon} {step} (did not run)"
        else:
            icon = "✗"
            line = f"{icon} {step} (failed: {status})"
        print(line)

    if verdict_results:
        tier  = verdict_results.get("tier", "?")
        label = verdict_results.get("label", "UNKNOWN")
        print(f"✓ Verdict: TIER {tier} — {label}")

    if sizing_results and not sizing_results.get("no_edge_flag"):
        q_pct = sizing_results.get("recommended_pct", 0)
        print(f"✓ Position sizing: Quarter Kelly {q_pct:.1f}%")

    print(SEP)
    print("Launching dashboard...")


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load config ───────────────────────────────────────────────────────────
    config = _load_config()

    wfa_cfg  = config.get("wfa", {})
    rep_cfg  = config.get("reporting", {})

    step_statuses: dict[str, str | None] = {
        "Data":                  "pending",
        "Pre-check":             "pending",
        "WFA":                   "pending",
        "Monte Carlo":           "pending",
        "Parameter Sensitivity": "pending",
        "Regime Analysis":       "pending",
        "Statistical Tests":     "pending",
    }

    data              = None
    wfa_results       = None
    mc_results        = None
    sens_results      = None
    regime_results    = None
    stats_results     = None
    verdict_results   = None
    sizing_results    = None
    pre_check_results = None

    # ── Step 1 — Data ─────────────────────────────────────────────────────────
    mode = config.get("data", {}).get("mode", "single")
    try:
        if mode == "portfolio":
            data = _fetch_universe(config)
            any_idx = next(iter(data.values())).index
            step_statuses["Data"] = "success"
            print(
                f"✓ Universe loaded: {len(data)} tickers  "
                f"{any_idx[0].date()} → {any_idx[-1].date()}"
            )
            print("⚠ WARNING: Survivorship bias present. Using current constituents only.")
        else:
            data = _fetch_data(config)
            symbol = config["data"]["symbol"]
            step_statuses["Data"] = "success"
            print(
                f"✓ Data loaded: {len(data)} bars  "
                f"{data.index[0].date()} → {data.index[-1].date()}  "
                f"({symbol})"
            )
    except Exception as e:
        step_statuses["Data"] = str(e)
        print(f"✗ Data failed: {e}")
        _print_banner(step_statuses, None, None)
        return

    # ── Step 1.5 — Pre-check ─────────────────────────────────────────────────
    from strategy.ma_crossover import MACrossoverStrategy as Strategy
    from validation.pre_check import run_pre_check, PreCheckFailedError

    try:
        pre_check_results = run_pre_check(Strategy, data, config)
        step_statuses["Pre-check"] = "success"
        print("✓ Pre-check passed")
    except PreCheckFailedError as e:
        step_statuses["Pre-check"] = str(e)
        pre_check_results = e.results
        print(f"✗ Pre-check failed: {e}")
        print("Pipeline stopped. Fix the strategy and re-run FALSIFY.")
        _print_banner(step_statuses, None, None)
        sys.exit(1)
    except Exception as e:
        step_statuses["Pre-check"] = str(e)
        print(f"✗ Pre-check error: {e}")
        _print_banner(step_statuses, None, None)
        return

    # ── Step 2 — Walk-Forward Analysis ────────────────────────────────────────
    try:
        from engine.wfa import run_wfa

        strategy   = Strategy()
        param_grid = Strategy.param_grid()

        wfa_results = run_wfa(
            price_data=data,
            strategy=strategy,
            param_grid=param_grid,
            train_years=int(wfa_cfg.get("train_years", 3)),
            oos_years=int(wfa_cfg.get("oos_years", 1)),
            config=config,
            objective=str(wfa_cfg.get("optimize_metric", "sharpe")),
        )
        n_windows = len(wfa_results.get("per_window_results", []))
        step_statuses["WFA"] = "success"
        print(f"✓ WFA complete: {n_windows} windows")
    except Exception as e:
        step_statuses["WFA"] = str(e)
        print(f"✗ WFA failed: {e}")

    # ── Step 3 — Monte Carlo ──────────────────────────────────────────────────
    try:
        if wfa_results is None:
            raise RuntimeError("WFA results required — WFA step did not complete.")

        from validation.monte_carlo import run_monte_carlo

        trade_log  = wfa_results["aggregate_results"]["full_trade_log"]
        mc_results = run_monte_carlo(trade_log, config)
        step_statuses["Monte Carlo"] = "success"
        print("✓ Monte Carlo complete")
    except Exception as e:
        step_statuses["Monte Carlo"] = str(e)
        print(f"✗ Monte Carlo failed: {e}")

    # ── Step 4 — Parameter Sensitivity ───────────────────────────────────────
    try:
        if wfa_results is None:
            raise RuntimeError("WFA results required — WFA step did not complete.")

        from validation.sensitivity import run_sensitivity

        param_grid_sens = Strategy.param_grid()
        optimal_params  = _optimal_params_from_wfa(wfa_results)

        if not param_grid_sens:
            raise RuntimeError(
                "Strategy has no tunable parameters — sensitivity analysis skipped."
            )

        sens_results = run_sensitivity(
            strategy_class=Strategy,
            data=data,
            param_grid=param_grid_sens,
            optimal_params=optimal_params,
            config=config,
        )
        step_statuses["Parameter Sensitivity"] = "success"
        print("✓ Sensitivity complete")
    except Exception as e:
        step_statuses["Parameter Sensitivity"] = str(e)
        print(f"✗ Sensitivity failed: {e}")

    # ── Step 5 — Regime Analysis ──────────────────────────────────────────────
    try:
        if wfa_results is None:
            raise RuntimeError("WFA results required — WFA step did not complete.")

        from validation.regime import run_regime_analysis

        trade_log    = wfa_results["aggregate_results"]["full_trade_log"]
        equity_curve = wfa_results["aggregate_results"]["equity_curve"]

        regime_results = run_regime_analysis(
            trade_log=trade_log,
            equity_curve=equity_curve,
            config=config,
        )
        step_statuses["Regime Analysis"] = "success"
        print("✓ Regime analysis complete")
    except Exception as e:
        step_statuses["Regime Analysis"] = str(e)
        print(f"✗ Regime analysis failed: {e}")

    # ── Step 6 — Statistical Tests ────────────────────────────────────────────
    try:
        if wfa_results is None:
            raise RuntimeError("WFA results required — WFA step did not complete.")

        from validation.stats import run_statistical_tests

        trade_log     = wfa_results["aggregate_results"]["full_trade_log"]
        stats_results = run_statistical_tests(trade_log, config)
        step_statuses["Statistical Tests"] = "success"
        print("✓ Statistical tests complete")
    except Exception as e:
        step_statuses["Statistical Tests"] = str(e)
        print(f"✗ Statistical tests failed: {e}")

    # ── Step 6.5 — Master Verdict ─────────────────────────────────────────────
    try:
        from validation.verdict import compute_verdict
        verdict_results = compute_verdict(
            wfa_results, mc_results, sens_results,
            regime_results, stats_results, config
        )
        print(f"✓ Master verdict: TIER {verdict_results['tier']} — {verdict_results['label']}")
    except Exception as e:
        print(f"✗ Verdict computation failed: {e}")

    # ── Step 6.6 — Position Sizing ────────────────────────────────────────────
    try:
        from validation.position_sizing import compute_position_sizing
        trade_log_for_sizing = (
            wfa_results["aggregate_results"]["full_trade_log"]
            if wfa_results else pd.DataFrame()
        )
        sizing_results = compute_position_sizing(trade_log_for_sizing, config)
        print("✓ Position sizing computed")
    except Exception as e:
        print(f"✗ Position sizing failed: {e}")

    # ── Pipeline summary ──────────────────────────────────────────────────────
    _print_banner(step_statuses, verdict_results, sizing_results)

    # ── Step 7 — Dashboard ────────────────────────────────────────────────────
    if rep_cfg.get("auto_launch_dashboard", True):
        try:
            from report.dashboard import launch_dashboard
            launch_dashboard(
                wfa_results=wfa_results,
                mc_results=mc_results,
                sensitivity_results=sens_results,
                regime_results=regime_results,
                stats_results=stats_results,
                config=config,
                verdict_results=verdict_results,
                sizing_results=sizing_results,
                pre_check_results=pre_check_results,
            )
        except Exception as e:
            print(f"✗ Dashboard failed: {e}")


if __name__ == "__main__":
    main()
