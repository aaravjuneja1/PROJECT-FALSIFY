"""
Minimum viable edge pre-check for FALSIFY.

Runs a single full backtest on all historical data before WFA.
Three gates must all pass before the pipeline continues.

Entry point: run_pre_check(strategy_class, data, config) -> dict
Raises PreCheckFailedError if any gate fails.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.backtester import backtest_signals
from engine.portfolio_backtester import backtest_portfolio
from metrics.calculator import _win_rate, _profit_factor


class PreCheckFailedError(Exception):
    """Raised when one or more pre-check gates fail. results dict attached as .results."""
    def __init__(self, message: str, results: dict):
        super().__init__(message)
        self.results = results


def run_pre_check(
    strategy_class,
    data: pd.DataFrame,
    config: dict,
) -> dict:
    """
    Run three minimum-viable-edge gates on the full historical dataset.

    Parameters
    ----------
    strategy_class : class (not instance) inheriting from BaseStrategy.
    data           : full OHLCV DataFrame from data/fetcher.py.
    config         : full config dict.

    Returns
    -------
    dict with keys: passed, gates, fail_reasons

    Raises
    ------
    PreCheckFailedError if any gate fails.
    """
    pc_cfg            = config.get("pre_check", {})
    min_trades        = int(pc_cfg.get("min_trades",        30))
    min_win_rate      = float(pc_cfg.get("min_win_rate",    0.35))
    min_profit_factor = float(pc_cfg.get("min_profit_factor", 1.0))

    # ── Instantiate strategy and generate signals on full history ─────────────
    strategy = strategy_class()
    pg = strategy_class.param_grid()
    # Use first value of each param as default; empty dict if no params
    params = {k: v[0] for k, v in pg.items()} if pg else {}

    if isinstance(data, dict):
        # ── Portfolio mode: run the REAL portfolio backtester on a 20-ticker
        #    sample — same engine as WFA/sensitivity (6-position cap, volume-ranked
        #    entry, intraday SL/TP, T+1). The book is capped at max_positions, so
        #    trade count does NOT scale linearly with universe size; we therefore
        #    do NOT extrapolate. The sample book's trade count is a conservative
        #    proxy for the full universe (more candidates only keep slots fuller).
        SAMPLE = 20
        all_tickers   = list(data.keys())
        sampled       = all_tickers[:SAMPLE]
        sample_prices = {t: data[t] for t in sampled}
        sigs = strategy.generate_signals_universe(sample_prices, params)
        _, trades = backtest_portfolio(sample_prices, sigs, config, strategy)
        n_trades = len(trades)
        print(
            f"Pre-check (portfolio mode): ran the portfolio backtester on "
            f"{len(sampled)}/{len(all_tickers)} tickers (6-position cap, SL/TP). "
            f"{n_trades} trades in the sample book — a conservative proxy for the "
            f"full universe (not extrapolated)."
        )
    else:
        signals = strategy.generate_signals(data, params)
        # T+1 entry is enforced by the backtester (pending order -> next open).
        # Do NOT shift here — that would delay entry to T+2.
        assert signals.index.equals(data.index), \
            "Signal index must match data index"
        assert not signals.isna().any(), \
            "Signals contain NaN. Fill warmup NaN with 0 in your strategy."
        _, trades = backtest_signals(data, signals, config)
        n_trades  = len(trades)

    # ── Gate evaluation ───────────────────────────────────────────────────────
    win_rate_val = float(_win_rate(trades))          # 0-100 percentage
    pf_val       = float(_profit_factor(trades))

    gate_trades = {
        "value":     n_trades,
        "threshold": min_trades,
        "passed":    n_trades >= min_trades,
    }
    gate_win_rate = {
        "value":     win_rate_val,
        "threshold": min_win_rate * 100,             # store threshold as pct too
        "passed":    (win_rate_val / 100.0) >= min_win_rate,
    }
    gate_pf = {
        "value":     pf_val,
        "threshold": min_profit_factor,
        "passed":    (pf_val != float("inf")) and (pf_val >= min_profit_factor),
    }

    # inf profit factor with > 0 trades and 0 losses = all wins → pass
    if pf_val == float("inf") and n_trades > 0:
        gate_pf["passed"] = True

    fail_reasons: list[str] = []

    if not gate_trades["passed"]:
        fail_reasons.append(
            f"Strategy produced only {n_trades} trades on full historical data. "
            f"Minimum is {min_trades}. The strategy trades too infrequently to validate "
            "statistically. Widen entry conditions or use a longer data period."
        )
    if not gate_win_rate["passed"]:
        fail_reasons.append(
            f"Win rate of {win_rate_val:.1f}% is below the {min_win_rate*100:.0f}% minimum. "
            f"Strategy loses more than {100 - min_win_rate*100:.0f}% of trades. Even with good "
            "reward/risk this is extremely difficult to trade psychologically and mathematically marginal."
        )
    if not gate_pf["passed"]:
        fail_reasons.append(
            f"Profit factor of {pf_val:.2f} is below 1.0. Strategy loses more money than it "
            "makes in aggregate. Do not proceed."
        )

    passed = len(fail_reasons) == 0

    # ── Print results ─────────────────────────────────────────────────────────
    SEP = "══════════════════════════════════════════════════"

    def _gate_line(label: str, value_str: str, threshold_str: str, gate_passed: bool) -> str:
        icon = "✓" if gate_passed else "✗"
        return f"{label:<16} {value_str:<10} (minimum: {threshold_str:<8}) {icon}"

    trades_icon    = "✓" if gate_trades["passed"]   else "✗"
    wr_icon        = "✓" if gate_win_rate["passed"]  else "✗"
    pf_icon        = "✓" if gate_pf["passed"]        else "✗"

    if passed:
        print(f"\n{SEP}")
        print("PRE-CHECK PASSED ✓")
        print(SEP)
        print(_gate_line("Trades:",        str(n_trades),            str(min_trades),      gate_trades["passed"]))
        print(_gate_line("Win Rate:",      f"{win_rate_val:.1f}%",   f"{min_win_rate*100:.0f}%", gate_win_rate["passed"]))
        pf_str = "∞" if pf_val == float("inf") else f"{pf_val:.2f}"
        print(_gate_line("Profit Factor:", pf_str,                   f"{min_profit_factor:.1f}", gate_pf["passed"]))
        print(SEP)
        print("Proceeding to full validation pipeline...")
    else:
        print(f"\n{SEP}")
        print("PRE-CHECK FAILED ✗ — PIPELINE STOPPED")
        print(SEP)
        print(_gate_line("Trades:",        str(n_trades),            str(min_trades),      gate_trades["passed"]))
        print(_gate_line("Win Rate:",      f"{win_rate_val:.1f}%",   f"{min_win_rate*100:.0f}%", gate_win_rate["passed"]))
        pf_str = "∞" if pf_val == float("inf") else f"{pf_val:.2f}"
        print(_gate_line("Profit Factor:", pf_str,                   f"{min_profit_factor:.1f}", gate_pf["passed"]))
        print()
        for reason in fail_reasons:
            print(f"REASON: {reason}")
        print()
        print("Fix the strategy and re-run FALSIFY.")
        print(SEP)

    results = {
        "passed": passed,
        "gates": {
            "trades":        gate_trades,
            "win_rate":      gate_win_rate,
            "profit_factor": gate_pf,
        },
        "fail_reasons": fail_reasons,
        "n_trades":     n_trades,
        "win_rate":     win_rate_val,
        "profit_factor": pf_val,
    }

    if not passed:
        raise PreCheckFailedError(
            f"Pre-check failed: {'; '.join(fail_reasons)}",
            results,
        )

    return results
