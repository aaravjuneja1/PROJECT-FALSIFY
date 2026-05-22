"""
Master verdict system for FALSIFY.

Aggregates results from all five validation modules into a single
tiered verdict with diagnosis and actionable next steps.

Entry point: compute_verdict(wfa, mc, sensitivity, regime, stats, config) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_verdict(
    wfa_results:         dict | None,
    mc_results:          dict | None,
    sensitivity_results: dict | None,
    regime_results:      dict | None,
    stats_results:       dict | None,
    config:              dict,
) -> dict:
    """
    Evaluate all five validation module results and produce a master verdict.

    Returns
    -------
    dict with keys:
        tier, label, colour, summary,
        gates (wfa/monte_carlo/sensitivity/regime/stats/trade_count),
        gates_passed, diagnoses, next_steps,
        total_oos_trades, sufficient_trades
    """
    wfa_cfg = config.get("wfa", {})
    wfe_accept_thr  = float(wfa_cfg.get("wfe_acceptable_threshold", 0.40)) * 100.0

    # Gate thresholds — single source of truth: config.yml verdict: section.
    v_cfg           = config.get("verdict", {})
    consistency_min = float(v_cfg.get("consistency_min", 0.60))
    ror_max         = float(v_cfg.get("ror_max", 0.05))
    bear_return_min = float(v_cfg.get("bear_return_min", -0.05))
    min_trades      = int(v_cfg.get("min_trades", 30))
    cons_thr        = consistency_min * 100.0   # consistency_score is a percentage

    # ── Trade count (hard gate) ───────────────────────────────────────────────
    agg = (wfa_results or {}).get("aggregate_results", {})
    total_oos_trades = int(agg.get("total_oos_trades", 0))
    sufficient_trades = total_oos_trades >= min_trades

    # ── Gate 1 — WFA ─────────────────────────────────────────────────────────
    if not wfa_results or not agg:
        gate_wfa = {"passed": False, "detail": "WFA not run"}
    else:
        wfe          = agg.get("wfe")
        cons_score   = agg.get("consistency_score", 0.0)
        avg_sharpe   = agg.get("avg_oos_sharpe", 0.0)
        wfe_ok       = (wfe is not None) and (wfe >= wfe_accept_thr)
        cons_ok      = (cons_score or 0.0) >= cons_thr
        sharpe_ok    = (avg_sharpe or 0.0) > 0
        wfa_pass     = wfe_ok and cons_ok and sharpe_ok
        wfe_str      = f"{wfe:.1f}%" if wfe is not None else "N/A"
        gate_wfa = {
            "passed": wfa_pass,
            "detail": f"WFE {wfe_str} | Consistency {cons_score:.1f}% | Avg Sharpe {avg_sharpe:.3f}",
        }

    # ── Gate 2 — Monte Carlo ──────────────────────────────────────────────────
    if not mc_results:
        gate_mc = {"passed": False, "detail": "Monte Carlo not run"}
    else:
        mc_verdict = mc_results.get("reshuffle", {}).get("verdict", "")
        ror_pct    = float(mc_results.get("reshuffle", {}).get("ror_pct", 100.0))
        ror_thr    = ror_max * 100.0   # config ror_max is a fraction; ror_pct is a percentage
        mc_pass    = (mc_verdict in ("ROBUST", "MARGINAL")) and (ror_pct <= ror_thr)
        gate_mc = {
            "passed":    mc_pass,
            "detail":    f"Verdict: {mc_verdict} | RoR: {ror_pct:.1f}% (max {ror_thr:.0f}%)",
            "high_ror":  ror_pct > ror_thr,
            "lucky":     mc_verdict not in ("ROBUST", "MARGINAL"),
        }

    # ── Gate 3 — Sensitivity ──────────────────────────────────────────────────
    if not sensitivity_results:
        gate_sens = {"passed": False, "detail": "Sensitivity not run"}
    else:
        sens_v     = sensitivity_results.get("verdict", "")
        sens_pass  = sens_v in ("ROBUST", "MARGINAL")
        rob_score  = float(sensitivity_results.get("robustness_score", 0.0))
        gate_sens = {
            "passed": sens_pass,
            "detail": f"Verdict: {sens_v} | Robustness: {rob_score:.1f}%",
        }

    # ── Gate 4 — Regime ───────────────────────────────────────────────────────
    if not regime_results:
        gate_reg = {"passed": False, "detail": "Regime analysis not run"}
    else:
        reg_v      = regime_results.get("verdict", "")
        bear_sh    = regime_results.get("aggregate", {}).get("bear_sharpe")
        # bear_sharpe None = no bear/crash data = don't fail on it
        bear_ok    = (bear_sh is None) or (bear_sh > -1.0)
        reg_pass   = (reg_v in ("ROBUST", "REGIME-DEPENDENT")) and bear_ok
        bear_str   = f"{bear_sh:.3f}" if bear_sh is not None else "N/A"
        gate_reg = {
            "passed": reg_pass,
            "detail": f"Verdict: {reg_v} | Bear Sharpe: {bear_str}",
        }

    # ── Gate 5 — Stats ────────────────────────────────────────────────────────
    if not stats_results:
        gate_stats = {"passed": False, "detail": "Statistical tests not run"}
    else:
        sig_t    = bool(stats_results.get("significant_ttest",       False))
        sig_perm = bool(stats_results.get("significant_permutation", False))
        st_pass  = sig_t and sig_perm
        p_t      = float(stats_results.get("p_value_ttest",       1.0))
        p_p      = float(stats_results.get("p_value_permutation", 1.0))
        gate_stats = {
            "passed": st_pass,
            "detail": f"t-test p={p_t:.4f} | Permutation p={p_p:.4f}",
        }

    # ── Trade count gate info ─────────────────────────────────────────────────
    gate_trade_count = {
        "passed": sufficient_trades,
        "detail": f"{total_oos_trades} OOS trades (minimum 30)",
    }

    gates_passed = sum([
        gate_wfa["passed"],
        gate_mc["passed"],
        gate_sens["passed"],
        gate_reg["passed"],
        gate_stats["passed"],
    ])

    # ── Tier assignment ───────────────────────────────────────────────────────
    if not sufficient_trades:
        tier    = 0
        label   = "INCONCLUSIVE"
        colour  = "grey"
        summary = (
            f"Only {total_oos_trades} OOS trades produced. Minimum 30 required for "
            "statistically valid results. Results cannot be trusted regardless of other metrics."
        )
    elif gates_passed == 5:
        tier    = 1
        label   = "TRADEABLE"
        colour  = "green"
        summary = (
            "All validation gates passed. Strategy shows robust, statistically significant "
            "edge across time, parameter space, and market regimes."
        )
    elif gates_passed == 4:
        tier    = 2
        label   = "PAPER TRADE"
        colour  = "amber"
        summary = (
            "Strong results with one area of concern. Paper trade for 60–90 days before "
            "committing real capital. Monitor the flagged area closely."
        )
    elif gates_passed == 3:
        tier    = 3
        label   = "REFINE"
        colour  = "orange"
        summary = (
            "Strategy shows partial edge but has significant weaknesses. Identify and fix "
            "the failing areas before considering deployment."
        )
    else:
        tier    = 4
        label   = "REJECT"
        colour  = "red"
        summary = (
            "Insufficient evidence of robust edge. Do not trade with real capital. "
            "Major rework required."
        )

    # ── Diagnoses for failed gates ────────────────────────────────────────────
    diagnoses: list[str] = []
    if not sufficient_trades:
        diagnoses.append("Insufficient OOS trades for reliable statistics.")
    if not gate_wfa["passed"] and sufficient_trades:
        diagnoses.append(
            "Walk-forward efficiency is low — strategy parameters do not generalise across time periods."
        )
    if not gate_mc["passed"] and sufficient_trades:
        diagnoses.append(
            "Monte Carlo shows high ruin risk or result is explained by lucky trade sequencing."
        )
    if not gate_sens["passed"] and sufficient_trades:
        diagnoses.append(
            "Strategy depends on precise parameter values — performance collapses with small parameter changes."
        )
    if not gate_reg["passed"] and sufficient_trades:
        diagnoses.append(
            "Strategy is regime-dependent — significant losses in bear or crash conditions."
        )
    if not gate_stats["passed"] and sufficient_trades:
        diagnoses.append(
            "Returns are not statistically distinguishable from random chance at the 5% significance level."
        )

    # Benchmark underperformance — report-only, not a pass/fail gate.
    alpha_cagr = agg.get("alpha_cagr")
    if alpha_cagr is not None and alpha_cagr < 0:
        bench_cagr = agg.get("benchmark_cagr")
        bench_str = (
            f" (Nifty 50 buy-and-hold returned {bench_cagr*100:.1f}% CAGR)"
            if bench_cagr is not None else ""
        )
        diagnoses.append(
            f"Strategy underperforms Nifty 50 buy-and-hold by "
            f"{abs(alpha_cagr)*100:.1f}% CAGR{bench_str}. Consider whether the "
            "added complexity is justified over simply holding the index."
        )

    # ── Next steps ────────────────────────────────────────────────────────────
    next_steps: list[str] = []

    if tier == 1:
        next_steps = [
            "1. Start with quarter Kelly position sizing.",
            "2. Paper trade for 30 days minimum.",
            "3. If live performance tracks OOS expectations, scale to half Kelly.",
            "4. Re-run FALSIFY every 6 months or after any major market regime change.",
        ]
    else:
        if not gate_wfa["passed"]:
            next_steps.append(
                "Simplify the strategy. Fewer parameters, broader entry conditions. "
                "Re-run with a wider OOS window."
            )
        if not gate_mc["passed"]:
            mc_info = gate_mc
            if mc_info.get("high_ror"):
                next_steps.append(
                    "Reduce position size. Current sizing produces unacceptable ruin probability."
                )
            if mc_info.get("lucky"):
                next_steps.append(
                    "Strategy edge may not be real. Test on additional instruments or time periods before proceeding."
                )
        if not gate_sens["passed"]:
            next_steps.append(
                "Move optimal parameters toward the centre of the grid. Avoid edge values. "
                "Widen entry/exit conditions."
            )
        if not gate_reg["passed"]:
            next_steps.append(
                "Add a market regime filter — only trade when broad market conditions are favourable. "
                "Consider adding a long-term trend filter (e.g. 200-day MA on index) to avoid trading in bear regimes."
            )
        if not gate_stats["passed"]:
            next_steps.append(
                "Collect more trades. Strategy needs longer testing period or more frequent signals "
                "to establish statistical significance."
            )
        if not sufficient_trades:
            next_steps.append(
                "The strategy needs more trades. Use a longer data period or a more active strategy."
            )

    return {
        "tier":             tier,
        "label":            label,
        "colour":           colour,
        "summary":          summary,
        "gates": {
            "wfa":         gate_wfa,
            "monte_carlo": gate_mc,
            "sensitivity": gate_sens,
            "regime":      gate_reg,
            "stats":       gate_stats,
            "trade_count": gate_trade_count,
        },
        "gates_passed":     gates_passed,
        "diagnoses":        diagnoses,
        "next_steps":       next_steps,
        "total_oos_trades": total_oos_trades,
        "sufficient_trades": sufficient_trades,
    }
