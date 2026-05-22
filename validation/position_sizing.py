"""
Kelly Criterion position sizing for FALSIFY.

Derives mathematically justified position sizes from proven OOS statistics.
Only run after validation — these numbers are meaningless on in-sample data.

Entry point: compute_position_sizing(trade_log, config) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_position_sizing(
    trade_log: pd.DataFrame,
    config:    dict,
) -> dict:
    """
    Compute Kelly Criterion position sizes from OOS trade log.

    Parameters
    ----------
    trade_log : OOS trade log from WFA. Required column: return_pct (decimal).
    config    : full config dict.

    Returns
    -------
    dict — see module docstring for full schema.
    """
    cap_cfg          = config.get("capital", {})
    starting_capital = float(cap_cfg.get("starting_capital", 1_000_000))
    max_positions    = int(cap_cfg.get("max_positions", 1))

    # ── Extract OOS trade statistics ──────────────────────────────────────────
    if trade_log.empty:
        return _no_edge_result(starting_capital, max_positions)

    returns   = trade_log["return_pct"].values
    wins      = returns[returns > 0]
    losses    = returns[returns < 0]
    n_trades  = len(returns)

    if n_trades == 0:
        return _no_edge_result(starting_capital, max_positions)

    win_rate  = float(len(wins)  / n_trades)
    avg_win   = float(wins.mean())  if len(wins)   > 0 else 0.0
    avg_loss  = float(abs(losses.mean())) if len(losses) > 0 else 0.0

    if avg_loss == 0.0:
        # All wins — edge exists but Kelly is infinite; cap at 1.0
        reward_risk = float("inf")
        kelly_full  = 1.0
        no_edge     = False
    else:
        reward_risk = avg_win / avg_loss
        kelly_full  = win_rate - ((1.0 - win_rate) / reward_risk)
        no_edge     = kelly_full <= 0.0

    if no_edge:
        kelly_full    = 0.0
        kelly_half    = 0.0
        kelly_quarter = 0.0
    else:
        kelly_full    = min(kelly_full, 1.0)   # cap at 100%
        kelly_half    = kelly_full / 2.0
        kelly_quarter = kelly_full / 4.0

    # ── Position sizes in INR ─────────────────────────────────────────────────
    def _sizes(f: float) -> dict:
        size_inr    = f * starting_capital
        total       = size_inr * max_positions
        reserve     = starting_capital - total
        over_cap    = total > starting_capital
        return {
            "fraction_pct": round(f * 100, 2),
            "size_inr":     round(size_inr, 0),
            "total_inr":    round(total, 0),
            "reserve_inr":  round(reserve, 0),
            "over_capital": over_cap,
        }

    full_sz    = _sizes(kelly_full)
    half_sz    = _sizes(kelly_half)
    quarter_sz = _sizes(kelly_quarter)

    capital_warning = full_sz["over_capital"]

    explanation = (
        "Full Kelly maximises long-term geometric growth in theory but assumes your edge "
        "estimate is perfectly accurate — it never is. Full Kelly also produces extreme "
        "drawdowns (up to 50%+ between peaks). Half Kelly halves the drawdown for ~75% of "
        "the growth. Quarter Kelly is the practical standard for systematic traders — "
        "conservative enough to survive edge uncertainty, aggressive enough to compound meaningfully."
    )

    # ── Print output ──────────────────────────────────────────────────────────
    SEP  = "══════════════════════════════════════════════════"
    THIN = "──────────────────────────────────────────────────"

    print(f"\n{SEP}")
    print("KELLY CRITERION POSITION SIZING")
    print(SEP)
    print(f"OOS Win Rate:      {win_rate*100:.1f}%")
    print(f"Avg Win:           {avg_win*100:.2f}%")
    print(f"Avg Loss:          {avg_loss*100:.2f}%")
    rr_str = f"{reward_risk:.2f}" if reward_risk != float("inf") else "∞"
    print(f"Reward/Risk:       {rr_str}")
    print()

    if no_edge:
        print("Kelly Fraction (f*): 0%  — NO EDGE (Kelly ≤ 0)")
    else:
        print(f"Kelly Fraction (f*): {kelly_full*100:.1f}%")

    print()
    print(f"{'':18} {'FRACTION':>10}   {'SIZE (INR)':>14}   {'% OF CAPITAL':>13}")

    def _fmt_row(label, sz, suffix=""):
        frac_s = f"{sz['fraction_pct']:.1f}%"
        size_s = f"₹{sz['size_inr']:,.0f}"
        pct_s  = f"{sz['fraction_pct']:.1f}%"
        return f"{label:<18} {frac_s:>10}   {size_s:>14}   {pct_s:>13}{suffix}"

    print(_fmt_row("Full Kelly:",    full_sz))
    print(_fmt_row("Half Kelly:",    half_sz))
    print(_fmt_row("Quarter Kelly:", quarter_sz, "  ← RECOMMENDED"))
    print()
    print(f"With max {max_positions} simultaneous position(s):")
    print(f"Quarter Kelly total exposure: ₹{quarter_sz['total_inr']:,.0f} ({quarter_sz['fraction_pct']*max_positions:.1f}% of capital)")
    print(f"Cash reserve:                 ₹{quarter_sz['reserve_inr']:,.0f} ({100 - quarter_sz['fraction_pct']*max_positions:.1f}% of capital)")

    if capital_warning:
        print(f"{THIN}")
        print("⚠ WARNING: Full Kelly exceeds capital with max positions. Use Half Kelly or Quarter Kelly.")
    if no_edge:
        print(f"{THIN}")
        print("⚠ WARNING: Kelly formula returns 0 — strategy has no mathematical edge.")
        print("   Do not size positions using this output.")

    print(SEP)

    return {
        "win_rate":             round(win_rate, 4),
        "avg_win":              round(avg_win,  4),
        "avg_loss":             round(avg_loss, 4),
        "reward_risk":          round(reward_risk, 4) if reward_risk != float("inf") else None,
        "kelly_full":           round(kelly_full,    4),
        "kelly_half":           round(kelly_half,    4),
        "kelly_quarter":        round(kelly_quarter, 4),
        "recommended_fraction": "quarter",
        "recommended_pct":      round(kelly_quarter * 100, 2),
        "recommended_inr":      round(kelly_quarter * starting_capital, 0),
        "total_exposure_inr":   round(kelly_quarter * starting_capital * max_positions, 0),
        "cash_reserve_inr":     round(starting_capital - kelly_quarter * starting_capital * max_positions, 0),
        "capital_warning":      capital_warning,
        "no_edge_flag":         no_edge,
        "explanation":          explanation,
        "full_sizing":          full_sz,
        "half_sizing":          half_sz,
        "quarter_sizing":       quarter_sz,
        "max_positions":        max_positions,
        "starting_capital":     starting_capital,
    }


def _no_edge_result(starting_capital: float, max_positions: int) -> dict:
    """Return a zeroed result when trade log is empty."""
    zero_sz = {"fraction_pct": 0.0, "size_inr": 0.0, "total_inr": 0.0,
               "reserve_inr": starting_capital, "over_capital": False}
    print("\n⚠ Position sizing skipped: trade log is empty.")
    return {
        "win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "reward_risk": None,
        "kelly_full": 0.0, "kelly_half": 0.0, "kelly_quarter": 0.0,
        "recommended_fraction": "quarter", "recommended_pct": 0.0,
        "recommended_inr": 0.0, "total_exposure_inr": 0.0,
        "cash_reserve_inr": starting_capital,
        "capital_warning": False, "no_edge_flag": True,
        "explanation": "No trades — cannot compute Kelly criterion.",
        "full_sizing": zero_sz, "half_sizing": zero_sz, "quarter_sizing": zero_sz,
        "max_positions": max_positions, "starting_capital": starting_capital,
    }
