"""
Statistical significance testing for FALSIFY.

Tests whether OOS strategy returns are distinguishable from zero (random chance).
Runs a one-sample t-test (parametric) and a permutation test (non-parametric) and
reports effect size, statistical power, and Sharpe significance per Lo (2002).

Entry point: run_statistical_tests(trade_log, config) -> dict
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats_mod

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.metrics_config import TRADING_DAYS_PER_YEAR
from metrics.calculator import avg_holding_trading_days, per_trade_sharpe


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_statistical_tests(
    trade_log: pd.DataFrame,
    config:    dict,
) -> dict:
    """
    Run a one-sample t-test, permutation test, effect size, power, and Sharpe
    significance on the OOS trade log.

    Parameters
    ----------
    trade_log : OOS trade log from WFA.
                Required column: return_pct (decimal, e.g. 0.05 = 5%).
                Optional columns: entry_date, exit_date (for holding-day calc).
    config    : settings dict. Optional nested key 'stats' with sub-keys:
                significance_level, permutation_iterations, random_seed,
                min_trades_warning.
    """
    # ── Extract config ─────────────────────────────────────────────────────────
    st_cfg         = config.get("stats", {})
    sig_level      = float(st_cfg.get("significance_level",     0.05))
    n_iterations   = int(  st_cfg.get("permutation_iterations", 10_000))
    rand_seed      = int(  st_cfg.get("random_seed",            42))
    min_trades_warn= int(  st_cfg.get("min_trades_warning",     30))

    np.random.seed(rand_seed)

    returns   = trade_log["return_pct"].values.astype(float)
    n_trades  = len(returns)

    # ── Guard checks ──────────────────────────────────────────────────────────
    if n_trades < 10:
        raise ValueError(
            f"Insufficient trades for statistical testing (N={n_trades} < 10). "
            "Results would be meaningless."
        )

    low_power_warning = n_trades < min_trades_warn
    if low_power_warning:
        print(
            f"\nWARNING: Only {n_trades} trades available. Statistical tests will "
            "have low power. Results may not be reliable. Interpret with caution."
        )
        if sig_level > 0.05:
            print(
                "  Note: Consider using significance_level = 0.10 for low trade counts."
            )

    # ── Return statistics ─────────────────────────────────────────────────────
    mean_return = float(np.mean(returns))
    std_return  = float(np.std(returns, ddof=1))
    stderr      = std_return / np.sqrt(n_trades)
    ci_lower    = mean_return - 1.96 * stderr
    ci_upper    = mean_return + 1.96 * stderr

    # ── Normality check ────────────────────────────────────────────────────────
    if n_trades <= 50:
        norm_stat, norm_p = stats_mod.shapiro(returns)
        norm_test_name = "Shapiro-Wilk"
    else:
        norm_stat, norm_p = stats_mod.normaltest(returns)
        norm_test_name = "D'Agostino-Pearson"

    is_normal   = bool(norm_p > 0.05)
    non_normal_note = (
        "Returns are non-normal. One-sample t-test assumptions may be violated. "
        "Weight permutation test result more heavily."
        if not is_normal else ""
    )

    # ── Test 1 — One-sample t-test ────────────────────────────────────────────
    t_stat, p_ttest = stats_mod.ttest_1samp(returns, popmean=0.0)
    t_stat   = float(t_stat)
    p_ttest  = float(p_ttest)
    sig_ttest = p_ttest < sig_level

    # ── Test 2 — Permutation test ─────────────────────────────────────────────
    # Memory guard: the signs matrix is (n_iterations × n_trades) of 8-byte ints.
    matrix_mb = (n_iterations * n_trades * 8) / 1024 / 1024
    if matrix_mb > 500:
        safe_iterations = int(500 * 1024 * 1024 / 8 / n_trades)
        print(
            f"⚠ Permutation matrix would be {matrix_mb:.0f}MB. Reducing "
            f"iterations from {n_iterations} to {safe_iterations} to stay under 500MB."
        )
        n_iterations = safe_iterations
        matrix_mb = (n_iterations * n_trades * 8) / 1024 / 1024
    if matrix_mb > 100:
        print(f"Note: permutation matrix is {matrix_mb:.0f}MB — this may take a moment.")

    # Signs matrix shape (n_iterations, n_trades); multiply by |returns|
    abs_returns = np.abs(returns)
    signs = np.random.choice([-1, 1], size=(n_iterations, n_trades))
    null_distribution = (signs * abs_returns).mean(axis=1)

    observed_mean = mean_return
    # Two-tailed: proportion where |null mean| >= |observed mean|
    p_perm = float(np.mean(np.abs(null_distribution) >= abs(observed_mean)))
    sig_perm = p_perm < sig_level

    perm_percentile = float(
        stats_mod.percentileofscore(null_distribution, observed_mean, kind="rank")
    )

    # ── Effect size — Cohen's d ───────────────────────────────────────────────
    cohens_d = float(mean_return / std_return) if std_return > 0.0 else 0.0
    cohens_d_size = _effect_size_label(abs(cohens_d))

    # ── Statistical power ─────────────────────────────────────────────────────
    power, min_n_for_power = _compute_power(
        abs(cohens_d), n_trades, sig_level
    )
    power_interp = _power_label(power)

    # ── Sharpe significance (Lo 2002) ─────────────────────────────────────────
    # SE formula requires consistent time units: use per-trade SR throughout.
    # Annualise only for display. Using annualised SR in the SE formula inflates
    # t-stat by sqrt(252/avg_hold_days) ≈ 3-5x. Holding days (TRADING days) and
    # the annual Sharpe come from the shared calculator.py helpers (ddof=1).
    avg_hold_days = avg_holding_trading_days(trade_log)
    sr_per_trade  = (mean_return / std_return) if std_return > 0.0 else 0.0
    sr_stderr     = np.sqrt((1.0 + 0.5 * sr_per_trade ** 2) / n_trades)
    sr_t_stat     = sr_per_trade / sr_stderr if sr_stderr > 0.0 else 0.0
    sr_p_value    = float(2.0 * (1.0 - stats_mod.norm.cdf(abs(sr_t_stat))))
    sharpe_sig    = sr_p_value < sig_level
    sharpe_annual = per_trade_sharpe(returns, avg_hold_days)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict, verdict_detail = _get_verdict(
        sig_ttest, sig_perm, cohens_d_size, power, low_power_warning
    )

    # ── Charts ────────────────────────────────────────────────────────────────
    charts_dir = Path(__file__).resolve().parents[1] / config["reporting"]["charts_dir"]
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_paths = _generate_all_charts(
        returns, null_distribution, observed_mean, ci_lower, ci_upper,
        sig_level, perm_percentile, p_perm, norm_test_name, norm_stat, norm_p,
        is_normal, t_stat, p_ttest, sig_ttest, sig_perm, cohens_d, cohens_d_size,
        power, power_interp, min_n_for_power, sharpe_annual, sr_p_value, sharpe_sig,
        verdict, n_trades, mean_return, std_return, charts_dir,
    )

    # ── Console print ──────────────────────────────────────────────────────────
    _print_summary(
        n_trades, sig_level, n_iterations,
        mean_return, std_return, ci_lower, ci_upper,
        t_stat, p_ttest, sig_ttest, norm_test_name, norm_p, is_normal, non_normal_note,
        perm_percentile, p_perm, sig_perm,
        cohens_d, cohens_d_size, power, power_interp, min_n_for_power,
        sharpe_annual, sr_stderr, sr_p_value, sharpe_sig,
        verdict, verdict_detail,
        low_power_warning, not is_normal,
    )

    return {
        "n_trades":               n_trades,
        "mean_return":            mean_return,
        "std_return":             std_return,
        "ci_95":                  (ci_lower, ci_upper),
        "is_normal":              is_normal,
        "t_statistic":            t_stat,
        "p_value_ttest":          p_ttest,
        "significant_ttest":      sig_ttest,
        "null_distribution":      null_distribution,
        "p_value_permutation":    p_perm,
        "permutation_percentile": perm_percentile,
        "significant_permutation": sig_perm,
        "cohens_d":               cohens_d,
        "cohens_d_size":          cohens_d_size,
        "power":                  power,
        "power_interpretation":   power_interp,
        "min_n_for_power":        min_n_for_power,
        "sharpe_annual":          float(sharpe_annual),
        "sharpe_stderr":          float(sr_stderr),
        "sharpe_p_value":         sr_p_value,
        "sharpe_significant":     sharpe_sig,
        "sharpe_sr_note":         "Significance tested on per-trade SR per Lo (2002). Annual SR shown for display only.",
        "verdict":                verdict,
        "verdict_detail":         verdict_detail,
        "low_power_warning":      low_power_warning,
        "chart_paths":            chart_paths,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _effect_size_label(d: float) -> str:
    if d >= 0.8:
        return "large"
    if d >= 0.5:
        return "medium"
    if d >= 0.2:
        return "small"
    return "negligible"


def _power_label(power: float) -> str:
    if power >= 0.80:
        return "adequate"
    if power >= 0.50:
        return "moderate"
    return "low — insufficient trades to reliably detect this effect size"


def _compute_power(
    effect_size: float,
    n:           int,
    alpha:       float,
) -> tuple[float, int | None]:
    """
    Returns (power, min_n_for_80pct_power).
    Uses statsmodels if available, else normal approximation.
    """
    try:
        from statsmodels.stats.power import TTestPower
        analysis = TTestPower()

        if effect_size < 1e-9:
            return 0.0, None

        power = float(analysis.power(
            effect_size=effect_size,
            nobs=n,
            alpha=alpha,
            alternative="two-sided",
        ))

        if power < 0.80:
            try:
                min_n = int(np.ceil(analysis.solve_power(
                    effect_size=effect_size,
                    power=0.80,
                    alpha=alpha,
                    alternative="two-sided",
                )))
            except Exception:
                min_n = None
        else:
            min_n = None

        return power, min_n

    except ImportError:
        print(
            "  statsmodels not found. Using normal approximation for power calculation."
        )
        return _power_normal_approx(effect_size, n, alpha)


def _power_normal_approx(
    effect_size: float,
    n:           int,
    alpha:       float,
) -> tuple[float, int | None]:
    """Normal approximation fallback for power when statsmodels is absent."""
    from scipy.stats import norm

    if effect_size < 1e-9:
        return 0.0, None

    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    ncp     = effect_size * np.sqrt(n)
    power   = float(1.0 - norm.cdf(z_alpha - ncp) + norm.cdf(-z_alpha - ncp))

    min_n: int | None = None
    if power < 0.80:
        z_beta  = norm.ppf(0.80)
        min_n   = int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2))

    return power, min_n


# ──────────────────────────────────────────────────────────────────────────────
# Verdict
# ──────────────────────────────────────────────────────────────────────────────

def _get_verdict(
    sig_ttest:        bool,
    sig_perm:         bool,
    cohens_d_size:    str,
    power:            float,
    low_power_warning: bool,
) -> tuple[str, str]:
    both_sig = sig_ttest and sig_perm

    if both_sig and cohens_d_size in ("large", "medium") and power >= 0.80:
        return (
            "STATISTICALLY ROBUST",
            "Both tests significant. Effect size meaningful. Adequate statistical "
            "power. Strong evidence of edge.",
        )

    if both_sig and power < 0.80:
        return (
            "SIGNIFICANT BUT UNDERPOWERED",
            "Both tests significant but low power due to trade count. Edge likely "
            "real but needs more trades to confirm reliably.",
        )

    if sig_perm and not sig_ttest:
        return (
            "MARGINAL",
            "Non-parametric test significant but parametric fails. Likely due to "
            "non-normal return distribution. Edge may be real but distribution is skewed.",
        )

    if not both_sig and low_power_warning:
        return (
            "INCONCLUSIVE",
            "Cannot reject null hypothesis, but test is underpowered. Collect more "
            "trades before drawing conclusions.",
        )

    return (
        "NOT SIGNIFICANT",
        "No statistical evidence of edge above random chance. Strategy returns are "
        "not distinguishable from zero at the chosen significance level.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Console print
# ──────────────────────────────────────────────────────────────────────────────

def _print_summary(
    n_trades:         int,
    sig_level:        float,
    n_iterations:     int,
    mean_return:      float,
    std_return:       float,
    ci_lower:         float,
    ci_upper:         float,
    t_stat:           float,
    p_ttest:          float,
    sig_ttest:        bool,
    norm_test_name:   str,
    norm_p:           float,
    is_normal:        bool,
    non_normal_note:  str,
    perm_percentile:  float,
    p_perm:           float,
    sig_perm:         bool,
    cohens_d:         float,
    cohens_d_size:    str,
    power:            float,
    power_interp:     str,
    min_n_for_power:  int | None,
    sharpe_annual:    float,
    sr_stderr:        float,
    sr_p_value:       float,
    sharpe_sig:       bool,
    verdict:          str,
    verdict_detail:   str,
    low_power_warning: bool,
    non_normal_flag:  bool,
) -> None:
    SEP  = "═" * 50
    THIN = "─" * 50

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    yesno = lambda b: "YES" if b else "NO"

    print(f"\n{SEP}")
    print("STATISTICAL SIGNIFICANCE RESULTS — FALSIFY")
    print(SEP)
    print(f"OOS Trades analysed:    {n_trades}")
    print(f"Significance level:     {sig_level}")
    print(f"Permutation iterations: {n_iterations:,}")

    print(THIN)
    print("RETURN STATISTICS")
    print(THIN)
    print(f"Mean trade return:      {pct(mean_return)}")
    print(f"Std dev:                {pct(std_return)}")
    print(f"95% CI on mean:         [{pct(ci_lower)}, {pct(ci_upper)}]")

    print(THIN)
    print("TEST 1 — ONE-SAMPLE T-TEST")
    print(THIN)
    print(f"t-statistic:            {t_stat:.2f}")
    print(f"p-value:                {p_ttest:.4f}")
    print(f"Significant (p<{sig_level}):   {yesno(sig_ttest)}")
    norm_result = "Normal" if is_normal else f"Non-normal (p={norm_p:.4f})"
    print(f"Normality ({norm_test_name}): {norm_result}")

    print(THIN)
    print("TEST 2 — PERMUTATION TEST")
    print(THIN)
    print(f"Observed mean percentile: {perm_percentile:.1f}th (of {n_iterations:,} random shuffles)")
    print(f"p-value:                {p_perm:.4f}")
    print(f"Significant (p<{sig_level}):   {yesno(sig_perm)}")

    print(THIN)
    print("EFFECT SIZE & POWER")
    print(THIN)
    print(f"Cohen's d:              {cohens_d:.2f} ({cohens_d_size})")
    print(f"Statistical power:      {power:.2f} ({power_interp})")
    if min_n_for_power is not None:
        print(f"Min trades for 80% power: {min_n_for_power}")

    print(THIN)
    print("SHARPE SIGNIFICANCE (Lo 2002)")
    print(THIN)
    print(f"Annualised Sharpe:      {sharpe_annual:.2f}")
    print(f"Sharpe std error:       {sr_stderr:.4f}")
    print(f"Sharpe p-value:         {sr_p_value:.4f}")
    print(f"Sharpe significant:     {yesno(sharpe_sig)}")

    print(THIN)
    print(f"VERDICT: {verdict}")
    print(f"Detail: {verdict_detail}")
    print(THIN)
    if low_power_warning:
        print(f"⚠ LOW POWER WARNING: only {n_trades} trades")
    if non_normal_flag:
        print("⚠ NON-NORMAL RETURNS: weight permutation test more heavily")
    print(SEP)


# ──────────────────────────────────────────────────────────────────────────────
# Charts
# ──────────────────────────────────────────────────────────────────────────────

def _generate_all_charts(
    returns:          np.ndarray,
    null_dist:        np.ndarray,
    observed_mean:    float,
    ci_lower:         float,
    ci_upper:         float,
    sig_level:        float,
    perm_percentile:  float,
    p_perm:           float,
    norm_test_name:   str,
    norm_stat:        float,
    norm_p:           float,
    is_normal:        bool,
    t_stat:           float,
    p_ttest:          float,
    sig_ttest:        bool,
    sig_perm:         bool,
    cohens_d:         float,
    cohens_d_size:    str,
    power:            float,
    power_interp:     str,
    min_n_for_power:  int | None,
    sharpe_annual:    float,
    sr_p_value:       float,
    sharpe_sig:       bool,
    verdict:          str,
    n_trades:         int,
    mean_return:      float,
    std_return:       float,
    charts_dir:       Path,
) -> list[str]:
    paths: list[str] = []

    p1 = charts_dir / "stats_permutation_null.png"
    _chart_permutation_null(
        null_dist, observed_mean, sig_level, perm_percentile, p_perm, p1
    )
    paths.append(str(p1))

    p2 = charts_dir / "stats_return_distribution.png"
    _chart_return_distribution(
        returns, mean_return, std_return, ci_lower, ci_upper,
        norm_test_name, norm_p, is_normal, p2
    )
    paths.append(str(p2))

    p3 = charts_dir / "stats_summary_dashboard.png"
    _chart_summary_dashboard(
        n_trades, sig_level,
        mean_return, std_return, ci_lower, ci_upper,
        t_stat, p_ttest, sig_ttest, norm_test_name, norm_p, is_normal,
        perm_percentile, p_perm, sig_perm,
        cohens_d, cohens_d_size, power, power_interp, min_n_for_power,
        sharpe_annual, sr_p_value, sharpe_sig,
        verdict, p3
    )
    paths.append(str(p3))

    return paths


def _chart_permutation_null(
    null_dist:       np.ndarray,
    observed_mean:   float,
    sig_level:       float,
    perm_percentile: float,
    p_perm:          float,
    save_path:       Path,
) -> None:
    """Histogram of null distribution with observed mean marked."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(11, 6))

        n_bins = min(80, max(30, len(null_dist) // 100))
        counts, bin_edges, _ = ax.hist(
            null_dist, bins=n_bins, color="#546E7A", alpha=0.85, edgecolor="none"
        )

        # Rejection region threshold: use the empirical quantile
        lower_thresh = float(np.percentile(null_dist, sig_level / 2 * 100))
        upper_thresh = float(np.percentile(null_dist, (1 - sig_level / 2) * 100))

        # Shade rejection regions
        ax.axvspan(bin_edges[0], lower_thresh, alpha=0.3, color="coral", label=f"Rejection region (α={sig_level})")
        ax.axvspan(upper_thresh, bin_edges[-1], alpha=0.3, color="coral")

        # Observed mean line
        ax.axvline(observed_mean, color="white", linewidth=2.0, linestyle="-",
                   label=f"Observed mean = {observed_mean * 100:.3f}%")

        y_max = float(counts.max()) * 1.15
        ax.set_ylim(0, max(y_max, 1.0))

        ax.set_xlabel("Permuted mean trade return")
        ax.set_ylabel("Count")
        ax.set_title(
            "Permutation Test — Null Distribution vs Observed Mean Return",
            fontsize=11
        )
        ax.grid(alpha=0.15)

        # Annotate p-value and percentile
        ax.text(
            0.97, 0.92,
            f"p-value = {p_perm:.4f}\nPercentile = {perm_percentile:.1f}th",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="white",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#263238", "alpha": 0.8},
        )

        ax.legend(fontsize=9, loc="upper left")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_return_distribution(
    returns:         np.ndarray,
    mean_return:     float,
    std_return:      float,
    ci_lower:        float,
    ci_upper:        float,
    norm_test_name:  str,
    norm_p:          float,
    is_normal:       bool,
    save_path:       Path,
) -> None:
    """Histogram of trade returns with fitted normal curve overlay."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(11, 6))

        n_bins = min(50, max(10, len(returns) // 3))
        r_min, r_max = float(returns.min()), float(returns.max())

        # Degenerate guard: all returns identical
        if abs(r_max - r_min) < 1e-10:
            pad = max(abs(mean_return) * 0.5, 0.001)
            bin_range = (r_min - pad, r_max + pad)
        else:
            bin_range = (r_min, r_max)

        counts, bin_edges, _ = ax.hist(
            returns, bins=n_bins, range=bin_range,
            color="#1565C0", alpha=0.85, edgecolor="none",
            density=True, label="Trade returns"
        )

        # Fitted normal overlay
        x_curve = np.linspace(bin_range[0] - std_return, bin_range[1] + std_return, 300)
        if std_return > 0:
            from scipy.stats import norm as sp_norm
            normal_pdf = sp_norm.pdf(x_curve, loc=mean_return, scale=std_return)
            ax.plot(x_curve, normal_pdf, color="#42A5F5", linewidth=1.8,
                    label="Fitted normal")

        # Zero line
        ax.axvline(0.0, color="white", linewidth=1.2, linestyle="--",
                   alpha=0.7, label="Zero return")

        # 95% CI bounds
        ax.axvline(ci_lower, color="coral", linewidth=1.2, linestyle="--",
                   alpha=0.9, label="95% CI bounds")
        ax.axvline(ci_upper, color="coral", linewidth=1.2, linestyle="--", alpha=0.9)

        # Mean line
        ax.axvline(mean_return, color="#FFEE58", linewidth=1.5, linestyle="-",
                   label=f"Mean = {mean_return * 100:.3f}%")

        ax.set_xlabel("Trade return (decimal)")
        ax.set_ylabel("Density")
        ax.set_title("OOS Trade Return Distribution", fontsize=11)
        ax.grid(alpha=0.15)

        norm_label = "Normal" if is_normal else "Non-normal"
        ax.text(
            0.97, 0.92,
            f"{norm_test_name}\np = {norm_p:.4f} ({norm_label})\n"
            f"Mean = {mean_return * 100:.3f}%\nStd = {std_return * 100:.3f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="white",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#263238", "alpha": 0.8},
        )

        ax.legend(fontsize=8, loc="upper left")
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def _chart_summary_dashboard(
    n_trades:         int,
    sig_level:        float,
    mean_return:      float,
    std_return:       float,
    ci_lower:         float,
    ci_upper:         float,
    t_stat:           float,
    p_ttest:          float,
    sig_ttest:        bool,
    norm_test_name:   str,
    norm_p:           float,
    is_normal:        bool,
    perm_percentile:  float,
    p_perm:           float,
    sig_perm:         bool,
    cohens_d:         float,
    cohens_d_size:    str,
    power:            float,
    power_interp:     str,
    min_n_for_power:  int | None,
    sharpe_annual:    float,
    sr_p_value:       float,
    sharpe_sig:       bool,
    verdict:          str,
    save_path:        Path,
) -> None:
    """Clean text-based summary card — suitable for direct inclusion in the report."""
    _VERDICT_COLORS = {
        "STATISTICALLY ROBUST":       "#4CAF50",
        "SIGNIFICANT BUT UNDERPOWERED": "#FFA726",
        "MARGINAL":                   "#FFA726",
        "INCONCLUSIVE":               "#FFA726",
        "NOT SIGNIFICANT":            "#E53935",
    }
    verdict_color = _VERDICT_COLORS.get(verdict, "#9E9E9E")

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    yesno = lambda b: ("✓ YES" if b else "✗ NO")

    with plt.style.context("dark_background"):
        fig = plt.figure(figsize=(12, 8))
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Background
        fig.patch.set_facecolor("#111111")

        # Title
        ax.text(0.5, 0.95, "Statistical Significance Summary",
                ha="center", va="top", fontsize=16, fontweight="bold", color="white")
        ax.text(0.5, 0.90, f"FALSIFY  |  N = {n_trades} trades  |  α = {sig_level}",
                ha="center", va="top", fontsize=11, color="#9E9E9E")

        # Horizontal divider
        ax.axhline(0.87, color="#333333", linewidth=1)

        # Three-column layout
        col_x = [0.05, 0.38, 0.70]
        row_start = 0.81
        row_step  = 0.072

        def label(x: float, y: float, text: str, bold: bool = False,
                  color: str = "#9E9E9E", size: int = 9) -> None:
            ax.text(x, y, text, ha="left", va="top",
                    fontsize=size, color=color,
                    fontweight="bold" if bold else "normal")

        def value(x: float, y: float, text: str, color: str = "white",
                  size: int = 10) -> None:
            ax.text(x + 0.01, y - 0.025, text, ha="left", va="top",
                    fontsize=size, color=color, fontweight="bold")

        # Column 1 — Return stats
        label(col_x[0], row_start,           "RETURN STATISTICS", bold=True, color="white", size=10)
        label(col_x[0], row_start - 1*row_step, "Mean return:")
        value(col_x[0], row_start - 1*row_step, pct(mean_return),
              color="#4CAF50" if mean_return > 0 else "#E53935")
        label(col_x[0], row_start - 2*row_step, "Std dev:")
        value(col_x[0], row_start - 2*row_step, pct(std_return))
        label(col_x[0], row_start - 3*row_step, "95% CI:")
        value(col_x[0], row_start - 3*row_step, f"[{pct(ci_lower)}, {pct(ci_upper)}]")
        label(col_x[0], row_start - 4*row_step, "Cohen's d:")
        value(col_x[0], row_start - 4*row_step, f"{cohens_d:.3f} ({cohens_d_size})")
        label(col_x[0], row_start - 5*row_step, "Power:")
        power_color = "#4CAF50" if power >= 0.8 else ("#FFA726" if power >= 0.5 else "#E53935")
        value(col_x[0], row_start - 5*row_step, f"{power:.2f} ({power_interp[:8]})", color=power_color)
        if min_n_for_power:
            label(col_x[0], row_start - 6*row_step, "Need N for 80% pwr:")
            value(col_x[0], row_start - 6*row_step, str(min_n_for_power), color="#FFA726")

        # Column 2 — Test results
        label(col_x[1], row_start,           "TEST RESULTS", bold=True, color="white", size=10)
        label(col_x[1], row_start - 1*row_step, "t-statistic:")
        value(col_x[1], row_start - 1*row_step, f"{t_stat:.3f}")
        label(col_x[1], row_start - 2*row_step, "t-test p-value:")
        value(col_x[1], row_start - 2*row_step, f"{p_ttest:.4f}",
              color="#4CAF50" if sig_ttest else "#E53935")
        label(col_x[1], row_start - 3*row_step, "t-test significant:")
        value(col_x[1], row_start - 3*row_step, yesno(sig_ttest),
              color="#4CAF50" if sig_ttest else "#E53935")
        label(col_x[1], row_start - 4*row_step, "Perm percentile:")
        value(col_x[1], row_start - 4*row_step, f"{perm_percentile:.1f}th")
        label(col_x[1], row_start - 5*row_step, "Perm p-value:")
        value(col_x[1], row_start - 5*row_step, f"{p_perm:.4f}",
              color="#4CAF50" if sig_perm else "#E53935")
        label(col_x[1], row_start - 6*row_step, "Perm significant:")
        value(col_x[1], row_start - 6*row_step, yesno(sig_perm),
              color="#4CAF50" if sig_perm else "#E53935")

        # Column 3 — Sharpe & normality
        label(col_x[2], row_start,           "SHARPE & NORMALITY", bold=True, color="white", size=10)
        label(col_x[2], row_start - 1*row_step, "Annual Sharpe:")
        sharpe_color = "#4CAF50" if sharpe_annual > 1.0 else ("#FFA726" if sharpe_annual > 0 else "#E53935")
        value(col_x[2], row_start - 1*row_step, f"{sharpe_annual:.3f}", color=sharpe_color)
        label(col_x[2], row_start - 2*row_step, "Sharpe p-value:")
        value(col_x[2], row_start - 2*row_step, f"{sr_p_value:.4f}",
              color="#4CAF50" if sharpe_sig else "#E53935")
        label(col_x[2], row_start - 3*row_step, "Sharpe significant:")
        value(col_x[2], row_start - 3*row_step, yesno(sharpe_sig),
              color="#4CAF50" if sharpe_sig else "#E53935")
        label(col_x[2], row_start - 4*row_step, f"Normality ({norm_test_name[:8]}):")
        value(col_x[2], row_start - 4*row_step,
              f"{'Normal' if is_normal else 'Non-normal'} (p={norm_p:.3f})",
              color="#4CAF50" if is_normal else "#FFA726")

        # Verdict box
        ax.add_patch(mpatches.FancyBboxPatch(
            (0.05, 0.04), 0.90, 0.10,
            boxstyle="round,pad=0.01",
            facecolor="#1A1A1A", edgecolor=verdict_color,
            linewidth=2, transform=ax.transAxes, clip_on=False,
        ))
        ax.text(0.50, 0.12, f"VERDICT: {verdict}",
                ha="center", va="center", fontsize=14,
                color=verdict_color, fontweight="bold")

        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
