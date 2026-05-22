"""
FALSIFY Interactive Dashboard.

Launches a PyQt6 desktop window with 7 pages of Plotly charts after
the full validation pipeline completes.

Entry point: launch_dashboard(wfa, mc, sensitivity, regime, stats, config)
"""
from __future__ import annotations

import json
import sys
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ── PyQt6 graceful import ─────────────────────────────────────────────────────
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
        QStackedWidget, QSplitter, QPushButton, QLabel, QSizePolicy,
    )
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtCore import Qt, QUrl
    from PyQt6.QtGui import QFont
    _PYQT_OK = True
except ImportError:
    _PYQT_OK = False


# ──────────────────────────────────────────────────────────────────────────────
# JSON encoder — handles numpy/pandas types that json.dumps can't handle
# ──────────────────────────────────────────────────────────────────────────────

class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return str(obj)
        if isinstance(obj, pd.Series):
            return obj.tolist()
        return super().default(obj)


def _j(obj) -> str:
    """Compact JSON dump safe for embedding in HTML <script> tags."""
    return json.dumps(obj, cls=_Encoder, separators=(",", ":"))


# ──────────────────────────────────────────────────────────────────────────────
# Plotly dark layout base
# ──────────────────────────────────────────────────────────────────────────────

_DARK = {
    "paper_bgcolor": "#141414",
    "plot_bgcolor":  "#141414",
    "font": {"color": "#F5F5F5", "family": "-apple-system,'Segoe UI',sans-serif", "size": 12},
    "xaxis": {"gridcolor": "#222222", "linecolor": "#333333", "zerolinecolor": "#333333"},
    "yaxis": {"gridcolor": "#222222", "linecolor": "#333333", "zerolinecolor": "#333333"},
    "legend": {"bgcolor": "#1A1A1A", "bordercolor": "#333333", "borderwidth": 1},
    "transition": {"duration": 400, "easing": "cubic-in-out"},
    "margin": {"l": 48, "r": 16, "t": 32, "b": 40},
}

_CONFIG = {"responsive": True, "displayModeBar": False}


def _merge_layout(extra: dict) -> dict:
    import copy
    base = copy.deepcopy(_DARK)
    base.update(extra)
    return base


# ──────────────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────────────

def launch_dashboard(
    wfa_results:         dict | None,
    mc_results:          dict | None,
    sensitivity_results: dict | None,
    regime_results:      dict | None,
    stats_results:       dict | None,
    config:              dict,
    verdict_results:     dict | None = None,
    sizing_results:      dict | None = None,
    pre_check_results:   dict | None = None,
) -> None:
    """
    Build and show the FALSIFY dashboard window.
    Blocks until the user closes the window.
    """
    if not _PYQT_OK:
        print(
            "\nDashboard requires PyQt6 and PyQt6-WebEngine.\n"
            "Install with: pip install PyQt6 PyQt6-WebEngine\n"
            "Skipping dashboard launch."
        )
        return

    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyleSheet(_QT_STYLESHEET)

    window = _FALSIFYDashboard(
        wfa_results, mc_results, sensitivity_results,
        regime_results, stats_results, config,
        verdict_results, sizing_results, pre_check_results,
    )
    window.show()
    sys.exit(app.exec())


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────

class _FALSIFYDashboard(QMainWindow):

    _NAV_LABELS = [
        "Overview",
        "Walk-Forward Analysis",
        "Monte Carlo",
        "Parameter Sensitivity",
        "Regime Analysis",
        "Statistical Significance",
        "Master Verdict",
    ]

    def __init__(
        self,
        wfa:     dict | None,
        mc:      dict | None,
        sens:    dict | None,
        reg:     dict | None,
        stat:    dict | None,
        cfg:     dict,
        verdict: dict | None = None,
        sizing:  dict | None = None,
        precheck: dict | None = None,
    ):
        super().__init__()
        self._wfa     = wfa
        self._mc      = mc
        self._sens    = sens
        self._reg     = reg
        self._stat    = stat
        self._cfg     = cfg
        self._verdict = verdict
        self._sizing  = sizing
        self._precheck = precheck

        self.setWindowTitle("FALSIFY — Strategy Validation Dashboard")
        self.resize(1400, 900)
        self.setMinimumSize(1100, 700)
        self._centre_on_screen()

        self._build_ui()

    def _centre_on_screen(self):
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width()  - 1400) // 2
        y = (screen.height() - 900)  // 2
        self.move(max(x, 0), max(y, 0))

    def _build_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)

        # Sidebar
        self._nav_buttons: list[QPushButton] = []
        sidebar = self._build_sidebar()
        sidebar.setFixedWidth(220)
        splitter.addWidget(sidebar)

        # Page stack
        self._stack = QStackedWidget()
        pages = [
            self._build_overview_page(),
            self._build_wfa_page(),
            self._build_mc_page(),
            self._build_sensitivity_page(),
            self._build_regime_page(),
            self._build_stats_page(),
            self._build_verdict_page(),
        ]
        for page in pages:
            view = QWebEngineView()
            view.setHtml(page, QUrl("about:blank"))
            self._stack.addWidget(view)

        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)
        self._set_active_nav(0)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setStyleSheet("QWidget#sidebar { background: #0A0A0A; }")

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 24, 0, 16)
        layout.setSpacing(0)

        # Title
        title = QLabel("FALSIFY")
        title.setStyleSheet("color: white; font-size: 22px; font-weight: bold; padding-left: 20px;")
        layout.addWidget(title)

        sub = QLabel("Strategy Validation")
        sub.setStyleSheet("color: #9E9E9E; font-size: 11px; padding-left: 20px; padding-bottom: 4px;")
        layout.addWidget(sub)

        # Red divider
        divider = QWidget()
        divider.setFixedHeight(2)
        divider.setStyleSheet("background: #E53935; margin: 12px 0px;")
        layout.addWidget(divider)

        # Nav buttons
        for i, label in enumerate(self._NAV_LABELS):
            btn = QPushButton(label)
            btn.setFixedHeight(48)
            btn.setCheckable(False)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, idx=i: self._navigate(idx))
            btn.setStyleSheet(_NAV_BTN_INACTIVE)
            self._nav_buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

        # Bottom info
        strategy_name = self._cfg.get("strategy", {}).get("name", "Strategy")
        date_str = datetime.date.today().strftime("%d %b %Y")

        for txt, style in [
            (strategy_name, "color: white; font-size: 11px; padding-left: 20px;"),
            (date_str,      "color: #757575; font-size: 10px; padding-left: 20px;"),
            ("FALSIFY v1.0", "color: #424242; font-size: 10px; padding-left: 20px; padding-bottom: 8px;"),
        ]:
            lbl = QLabel(txt)
            lbl.setStyleSheet(style)
            layout.addWidget(lbl)

        # ── Built-by footer ───────────────────────────────────────────────────
        footer_divider = QWidget()
        footer_divider.setFixedHeight(1)
        footer_divider.setStyleSheet("background: #1E1E1E; margin: 6px 16px;")
        layout.addWidget(footer_divider)

        builtby = QLabel("Built by")
        builtby.setStyleSheet("color:#757575;font-size:10px;padding-left:20px;padding-top:6px;")
        layout.addWidget(builtby)

        name_lbl = QLabel("Aarav Juneja")
        name_lbl.setStyleSheet("color:white;font-size:13px;font-weight:bold;padding-left:20px;")
        layout.addWidget(name_lbl)

        link_lbl = QLabel(
            '<a href="https://www.linkedin.com/in/aarav-juneja1" '
            'style="color:#E53935;text-decoration:none;">LinkedIn ↗</a>'
        )
        link_lbl.setOpenExternalLinks(True)
        link_lbl.setStyleSheet("font-size:11px;padding-left:20px;padding-bottom:10px;")
        layout.addWidget(link_lbl)

        return sidebar

    def _navigate(self, idx: int):
        self._stack.setCurrentIndex(idx)
        self._set_active_nav(idx)
        # Replay the entrance animations each time a page is shown.
        view = self._stack.widget(idx)
        if isinstance(view, QWebEngineView):
            view.page().runJavaScript("window.__falsifyReplay && window.__falsifyReplay();")

    def _set_active_nav(self, active: int):
        for i, btn in enumerate(self._nav_buttons):
            btn.setStyleSheet(_NAV_BTN_ACTIVE if i == active else _NAV_BTN_INACTIVE)

    # ──────────────────────────────────────────────────────────────────────────
    # HTML helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _metric_card(self, label: str, value: str, colour: str = "neutral") -> str:
        col_map = {"positive": "#4CAF50", "negative": "#EF5350", "neutral": "#F5F5F5",
                   "amber": "#FFA726", "red": "#EF5350", "green": "#4CAF50"}
        colour_hex = col_map.get(colour, "#F5F5F5")
        return (
            f'<div class="metric-card">'
            f'<div class="metric-value" style="color:{colour_hex}">{value}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>'
        )

    def _verdict_badge(self, verdict: str) -> str:
        v = (verdict or "").upper()
        if "ROBUST" in v or "TRADEABLE" in v or "STATISTICALLY ROBUST" in v:
            bg, text = "#4CAF50", "white"
        elif "FRAGILE" in v or "NOT SIGNIFICANT" in v or "LUCKY" in v or "DO NOT TRADE" in v:
            bg, text = "#E53935", "white"
        else:
            bg, text = "#FFA726", "black"
        return (
            f'<span class="verdict-badge" style="background:{bg};color:{text}">'
            f'{verdict}</span>'
        )

    def _narrative(self, text: str) -> str:
        return f'<p class="narrative">{text}</p>'

    def _warning_box(self, text: str) -> str:
        return f'<div class="warning-box">{text}</div>'

    def _plotly_chart(self, fig: dict, div_id: str, height: str = "360px") -> str:
        return (
            f'<div id="{div_id}" style="width:100%;height:{height}"></div>'
            f'<script>Plotly.newPlot("{div_id}",{_j(fig["data"])},{_j(fig["layout"])},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )

    def _stagger_js(self) -> str:
        # Page fade/slide-in, staggered cards, 0->value metric counters, and a
        # spinner. __falsifyReplay() is re-invoked on every nav switch (see
        # _navigate) so the animations replay each time a page is shown.
        return """<script>
function falsifyShowSpin(){var s=document.getElementById('falsify-spin');
  if(s){s.style.display='flex';setTimeout(function(){s.style.display='none';},420);}}
function falsifyStagger(){var cards=document.querySelectorAll('.card');
  cards.forEach(function(c,i){c.classList.remove('in');
    setTimeout(function(){c.classList.add('in');}, i*80);});}
function falsifyCounters(){
  document.querySelectorAll('.metric-value').forEach(function(el){
    var fin=el.getAttribute('data-final');
    if(fin===null){fin=el.textContent;el.setAttribute('data-final',fin);}
    var m=fin.match(/-?[\\d,]*\\.?\\d+/);
    if(!m){el.textContent=fin;return;}
    var numStr=m[0].replace(/,/g,'');var target=parseFloat(numStr);
    if(isNaN(target)){el.textContent=fin;return;}
    var decimals=(numStr.split('.')[1]||'').length;
    var prefix=fin.slice(0,m.index);var suffix=fin.slice(m.index+m[0].length);
    var hasComma=m[0].indexOf(',')>=0;var start=null,dur=800;
    function fmt(v){var s=decimals?v.toFixed(decimals):Math.round(v).toString();
      if(hasComma){var p=s.split('.');p[0]=p[0].replace(/\\B(?=(\\d{3})+(?!\\d))/g,',');s=p.join('.');}
      return prefix+s+suffix;}
    function step(ts){if(!start)start=ts;var p=Math.min((ts-start)/dur,1);
      var e=1-Math.pow(1-p,3);el.textContent=fmt(target*e);
      if(p<1)requestAnimationFrame(step);else el.textContent=fin;}
    requestAnimationFrame(step);});}
window.__falsifyReplay=function(){falsifyShowSpin();
  document.body.classList.remove('loaded');void document.body.offsetWidth;
  document.body.classList.add('loaded');falsifyStagger();falsifyCounters();};
document.addEventListener('DOMContentLoaded',function(){window.__falsifyReplay();});
</script>"""

    def _build_page_html(self, title: str, subtitle: str, content: str) -> str:
        return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{background:#0A0A0A;color:#F5F5F5;font-family:-apple-system,'Segoe UI',sans-serif;
     padding:16px 20px;overflow-x:hidden}}
.page-title{{font-size:18px;font-weight:bold;color:white;margin-bottom:2px}}
.page-subtitle{{font-size:11px;color:#9E9E9E;margin-bottom:12px}}
.card{{background:#141414;border-radius:6px;padding:12px 16px;margin-bottom:10px;
       border:1px solid #1E1E1E}}
.metric-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px}}
.metric-card{{background:#141414;border-radius:6px;padding:10px 12px;text-align:center;
              border:1px solid #1E1E1E}}
.metric-value{{font-size:18px;font-weight:bold;margin-bottom:3px}}
.metric-label{{font-size:10px;color:#9E9E9E;text-transform:uppercase;letter-spacing:.5px}}
.section-title{{font-size:11px;color:#9E9E9E;text-transform:uppercase;
                letter-spacing:1px;margin-bottom:8px}}
.verdict-badge{{display:inline-block;font-weight:bold;font-size:11px;
                padding:4px 12px;border-radius:20px}}
.narrative{{font-size:12px;color:#9E9E9E;line-height:1.5;max-width:840px;
            margin-top:6px}}
.warning-box{{background:rgba(229,57,53,.1);border-left:3px solid #E53935;
              padding:8px 12px;border-radius:4px;font-size:11px;color:#EF9A9A;
              margin-bottom:8px}}
.positive{{color:#4CAF50}}.negative{{color:#EF5350}}.neutral{{color:#F5F5F5}}
table{{width:100%;border-collapse:collapse;font-size:11px}}
th{{text-align:left;padding:6px 10px;color:#9E9E9E;text-transform:uppercase;
    font-size:10px;letter-spacing:.5px;border-bottom:1px solid #222}}
td{{padding:5px 10px;border-bottom:1px solid #1A1A1A}}
tr:nth-child(even){{background:#1A1A1A}}
/* red scrollbars */
::-webkit-scrollbar{{width:8px;height:8px}}
::-webkit-scrollbar-track{{background:#111}}
::-webkit-scrollbar-thumb{{background:#333;border-radius:4px}}
::-webkit-scrollbar-thumb:hover{{background:#E53935}}
/* page + card entrance transitions */
body{{opacity:0;transform:translateY(10px)}}
body.loaded{{opacity:1;transform:translateY(0);
    transition:opacity .45s ease,transform .45s ease}}
.card{{opacity:0;transform:translateY(16px)}}
.card.in{{opacity:1;transform:translateY(0);
    transition:opacity .35s ease,transform .35s ease}}
/* TIER 1 pulsing green glow */
@keyframes tier1pulse{{0%{{box-shadow:0 0 0 0 rgba(76,175,80,.65)}}
    70%{{box-shadow:0 0 0 18px rgba(76,175,80,0)}}
    100%{{box-shadow:0 0 0 0 rgba(76,175,80,0)}}}}
.tier1-glow{{animation:tier1pulse 1.8s infinite}}
/* page-switch spinner */
#falsify-spin{{position:fixed;inset:0;display:none;align-items:center;
    justify-content:center;background:rgba(10,10,10,.5);z-index:9999}}
#falsify-spin div{{width:34px;height:34px;border:3px solid #2a2a2a;
    border-top-color:#E53935;border-radius:50%;animation:spin .7s linear infinite}}
@keyframes spin{{to{{transform:rotate(360deg)}}}}
</style>
</head>
<body>
<div id="falsify-spin"><div></div></div>
<div class="page-title">{title}</div>
<div class="page-subtitle">{subtitle}</div>
{content}
{self._stagger_js()}
</body>
</html>"""

    # ──────────────────────────────────────────────────────────────────────────
    # Module result accessors — safe, return None on missing data
    # ──────────────────────────────────────────────────────────────────────────

    def _wfa_agg(self) -> dict:
        return (self._wfa or {}).get("aggregate_results", {})

    def _wfa_windows(self) -> list:
        return (self._wfa or {}).get("per_window_results", [])

    # ──────────────────────────────────────────────────────────────────────────
    # Page 1 — Overview
    # ──────────────────────────────────────────────────────────────────────────

    def _build_overview_page(self) -> str:
        agg = self._wfa_agg()

        sharpe = agg.get("avg_oos_sharpe")
        cagr   = agg.get("avg_oos_cagr")
        dd     = agg.get("worst_oos_drawdown")
        n_tr   = agg.get("total_oos_trades")
        eq     = agg.get("equity_curve")  # pd.Series

        # Colour helpers
        def sh_col(v):
            return "positive" if (v or 0) > 1 else ("amber" if (v or 0) > 0.5 else "negative")
        def cagr_col(v):
            return "positive" if (v or 0) > 0.10 else ("amber" if (v or 0) >= 0 else "negative")
        def dd_col(v):
            return "positive" if (v or 1) < 0.15 else ("amber" if (v or 1) < 0.30 else "negative")

        metrics_html = (
            '<div class="metric-grid">'
            + self._metric_card("OOS Sharpe",    f"{sharpe:.3f}" if sharpe is not None else "N/A", sh_col(sharpe))
            + self._metric_card("OOS CAGR",      f"{cagr*100:.2f}%" if cagr is not None else "N/A", cagr_col(cagr))
            + self._metric_card("Max Drawdown",  f"{dd*100:.2f}%" if dd is not None else "N/A", dd_col(dd))
            + self._metric_card("Total Trades",  str(n_tr) if n_tr is not None else "N/A")
            + '</div>'
        )

        # Scorecard table
        def mod_row(name, result_dict, verdict_key, metric_label, metric_val):
            v = (result_dict or {}).get(verdict_key, "—") if result_dict else "—"
            icon = "✓" if "ROBUST" in str(v).upper() or "STATISTICALLY" in str(v).upper() \
                   else ("✗" if "FRAGILE" in str(v).upper() or "NOT SIG" in str(v).upper() else "⚠")
            icon_col = "#4CAF50" if icon == "✓" else ("#E53935" if icon == "✗" else "#FFA726")
            return (
                f"<tr><td>{name}</td>"
                f"<td>{self._verdict_badge(v)}</td>"
                f"<td>{metric_label}: {metric_val}</td>"
                f"<td style='color:{icon_col};font-size:18px'>{icon}</td></tr>"
            )

        mc_ror = f"{self._mc['reshuffle']['ror_pct']:.1f}%" if self._mc else "—"
        sens_rob = f"{self._sens.get('robustness_score', 0)*100:.0f}%" if self._sens else "—"
        reg_conc = f"{self._reg['aggregate'].get('profit_concentration') or 0:.1f}%" if self._reg else "—"
        stat_p = f"{self._stat.get('p_value_ttest', 1):.4f}" if self._stat else "—"
        wfe_val = agg.get("wfe")
        wfe_str = f"{wfe_val:.1f}%" if wfe_val is not None else "—"

        wfa_verdict = self._derive_wfa_verdict()
        mc_verdict  = (self._mc or {}).get("reshuffle", {}).get("verdict", "—")
        sens_verdict = (self._sens or {}).get("verdict", "—")
        reg_verdict  = (self._reg or {}).get("verdict", "—")
        stat_verdict = (self._stat or {}).get("verdict", "—")

        scorecard = (
            '<div class="card"><div class="section-title">Validation Scorecard</div>'
            '<table>'
            '<tr><th>Module</th><th>Verdict</th><th>Key Metric</th><th>Status</th></tr>'
            + mod_row("Walk-Forward Analysis", {"verdict": wfa_verdict}, "verdict", "WFE", wfe_str)
            + mod_row("Monte Carlo", {"verdict": mc_verdict}, "verdict", "RoR", mc_ror)
            + mod_row("Parameter Sensitivity", {"verdict": sens_verdict}, "verdict", "Robustness", sens_rob)
            + mod_row("Regime Analysis", {"verdict": reg_verdict}, "verdict", "Conc", reg_conc)
            + mod_row("Statistical Significance", {"verdict": stat_verdict}, "verdict", "p-value", stat_p)
            + '</table></div>'
        )

        # OOS equity chart
        eq_chart = ""
        if eq is not None and not eq.empty:
            dates = [d.strftime("%Y-%m-%d") for d in eq.index]
            vals  = [float(v) for v in eq.values]
            start = vals[0] if vals else 1
            fig = {
                "data": [{
                    "type": "scatter", "mode": "lines", "x": dates, "y": vals,
                    "line": {"color": "#E53935", "width": 2},
                    "fill": "tozeroy", "fillcolor": "rgba(229,57,53,0.06)",
                    "name": "OOS Equity",
                    "hovertemplate": "%{x}<br>₹%{y:,.0f}<extra></extra>",
                }],
                "layout": _merge_layout({
                    "title": {"text": "Out-of-Sample Equity Curve", "font": {"size": 13}},
                    "xaxis": {**_DARK["xaxis"], "title": "Date"},
                    "yaxis": {**_DARK["yaxis"], "title": "Portfolio Value (₹)", "tickprefix": "₹", "tickformat": ",.0f"},
                }),
            }
            eq_chart = '<div class="card">' + self._plotly_chart(fig, "ov_equity", "240px") + '</div>'

            final = vals[-1]
            total_ret = (final - start) / start * 100
            narr = (
                f"Your strategy turned ₹{start:,.0f} into ₹{final:,.0f} over the OOS period, "
                f"a total return of {total_ret:.1f}%. "
            )
            if sharpe is not None:
                narr += (
                    f"The average OOS Sharpe ratio of {sharpe:.2f} means you earned "
                    f"{sharpe:.2f} units of return per unit of risk. "
                )
            if dd is not None:
                narr += f"Worst OOS drawdown was {dd*100:.1f}%."
            eq_chart += self._narrative(narr)

        # Pre-check status bar
        precheck_bar = ""
        if self._precheck:
            pc = self._precheck
            if pc.get("passed"):
                wr  = pc.get("win_rate", 0)
                pf  = pc.get("profit_factor", 0)
                nt  = pc.get("n_trades", 0)
                pf_str = "∞" if pf == float("inf") else f"{pf:.2f}"
                precheck_bar = (
                    f'<div style="background:#1B5E20;border:1px solid #4CAF50;border-radius:6px;'
                    f'padding:8px 14px;margin-bottom:10px;font-size:11px;color:#A5D6A7">'
                    f'PRE-CHECK PASSED ✓ — {nt} trades &nbsp;|&nbsp; Win rate {wr:.1f}% &nbsp;|&nbsp; PF {pf_str}'
                    f'</div>'
                )
            else:
                reasons = "; ".join(pc.get("fail_reasons", ["Unknown failure"]))
                precheck_bar = (
                    f'<div style="background:#B71C1C;border:1px solid #E53935;border-radius:6px;'
                    f'padding:8px 14px;margin-bottom:10px;font-size:11px;color:#FFCDD2">'
                    f'PRE-CHECK FAILED ✗ — {reasons}'
                    f'</div>'
                )

        # Add trade count row to scorecard
        agg_tc = self._wfa_agg()
        n_oos = agg_tc.get("total_oos_trades", 0)
        tc_ok = (n_oos or 0) >= 30
        tc_icon = "✓" if tc_ok else "✗"
        tc_col  = "#4CAF50" if tc_ok else "#E53935"
        trade_count_row = (
            f"<tr><td>Trade Count</td>"
            f"<td>{self._verdict_badge('SUFFICIENT' if tc_ok else 'INSUFFICIENT')}</td>"
            f"<td>OOS trades: {n_oos} (min 30)</td>"
            f"<td style='color:{tc_col};font-size:18px'>{tc_icon}</td></tr>"
        )
        scorecard_with_tc = scorecard.replace("</table></div>", trade_count_row + "</table></div>")

        # ── Survivorship-bias warning (portfolio mode only) ───────────────────
        survivorship_card = ""
        if self._cfg.get("data", {}).get("mode") == "portfolio":
            survivorship_card = (
                '<div class="warning-box" style="border-left-color:#FFA726;'
                'background:rgba(255,167,38,.08);color:#FFCC80;margin-bottom:10px">'
                '⚠ SURVIVORSHIP BIAS: This backtest uses the CURRENT Nifty 200 '
                'constituents. Stocks delisted or removed from the index between '
                '2015 and 2024 are excluded. This likely overstates historical '
                'returns — treat all results as upper-bound estimates.'
                '</div>'
            )

        # ── Benchmark comparison card (Strategy vs Nifty 50 buy & hold) ───────
        benchmark_card = ""
        bench_cagr = agg.get("benchmark_cagr")
        strat_cagr = agg.get("strategy_cagr")
        alpha      = agg.get("alpha_cagr")
        sym        = agg.get("benchmark_label") or agg.get("benchmark_symbol") or "Nifty 50"
        if alpha is not None and bench_cagr is not None and strat_cagr is not None:
            alpha_col = "#4CAF50" if alpha > 0 else "#EF5350"
            sign      = "+" if alpha >= 0 else ""
            verb      = "outperformed" if alpha > 0 else "underperformed"
            benchmark_card = (
                '<div class="card"><div class="section-title">'
                f'Benchmark — Strategy vs {sym} Buy &amp; Hold</div>'
                '<div class="metric-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:6px">'
                + self._metric_card("Strategy CAGR", f"{strat_cagr*100:.2f}%",
                                     "positive" if strat_cagr > 0 else "negative")
                + self._metric_card(f"{sym} CAGR", f"{bench_cagr*100:.2f}%", "neutral")
                + f'<div class="metric-card"><div class="metric-value" style="color:{alpha_col}">'
                  f'{sign}{alpha*100:.2f}%</div><div class="metric-label">Alpha (annual)</div></div>'
                + '</div>'
                + self._narrative(
                    f"Your strategy {verb} a buy-and-hold of {sym} by "
                    f"{abs(alpha)*100:.2f}% CAGR over the out-of-sample period."
                    + ("" if alpha > 0 else " Consider whether the added complexity is justified.")
                  )
                + '</div>'
            )

        content = (precheck_bar + survivorship_card + metrics_html
                   + benchmark_card + scorecard_with_tc + eq_chart)
        strategy_name = self._cfg.get("strategy", {}).get("name", "Strategy")
        return self._build_page_html("Overview", strategy_name, content)

    # ──────────────────────────────────────────────────────────────────────────
    # Page 2 — Walk-Forward Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def _build_wfa_page(self) -> str:
        agg = self._wfa_agg()
        wins = self._wfa_windows()

        if not agg:
            return self._not_run_page("Walk-Forward Analysis")

        wfe        = agg.get("wfe")
        cons_score = agg.get("consistency_score")
        avg_sharpe = agg.get("avg_oos_sharpe")
        avg_cagr   = agg.get("avg_oos_cagr")

        metrics_html = (
            '<div class="metric-grid">'
            + self._metric_card("WFE Score",        f"{wfe:.1f}%" if wfe else "N/A",
                                 "positive" if (wfe or 0) > 60 else ("amber" if (wfe or 0) > 40 else "negative"))
            + self._metric_card("Consistency",      f"{cons_score:.1f}%" if cons_score is not None else "N/A",
                                 "positive" if (cons_score or 0) > 70 else ("amber" if (cons_score or 0) > 50 else "negative"))
            + self._metric_card("Avg OOS Sharpe",   f"{avg_sharpe:.3f}" if avg_sharpe is not None else "N/A",
                                 "positive" if (avg_sharpe or 0) > 1 else ("amber" if (avg_sharpe or 0) > 0.5 else "negative"))
            + self._metric_card("Avg OOS CAGR",     f"{avg_cagr*100:.2f}%" if avg_cagr is not None else "N/A",
                                 "positive" if (avg_cagr or 0) > 0.10 else ("amber" if (avg_cagr or 0) >= 0 else "negative"))
            + '</div>'
        )

        # IS vs OOS Sharpe bar chart
        bar_chart = ""
        if wins:
            x_labels  = [f"W{w['window_number']}" for w in wins]
            is_sharpes  = [float(w.get("is_sharpe", 0))  for w in wins]
            oos_sharpes = [float(w.get("oos_sharpe", 0)) for w in wins]
            fig = {
                "data": [
                    {"type": "bar", "name": "IS Sharpe",  "x": x_labels, "y": is_sharpes,
                     "marker": {"color": "#546E7A"}, "opacity": 0.85},
                    {"type": "bar", "name": "OOS Sharpe", "x": x_labels, "y": oos_sharpes,
                     "marker": {"color": "#E53935"}, "opacity": 0.9},
                ],
                "layout": _merge_layout({
                    "title": {"text": "In-Sample vs Out-of-Sample Sharpe Per Window"},
                    "barmode": "group",
                    "xaxis": {**_DARK["xaxis"], "title": "Window"},
                    "yaxis": {**_DARK["yaxis"], "title": "Sharpe Ratio",
                              "zeroline": True, "zerolinecolor": "#444"},
                }),
            }
            bar_chart = '<div class="card">' + self._plotly_chart(fig, "wfa_bars", "220px") + '</div>'

        # Parameter table
        table_rows = ""
        for w in wins:
            sh = w.get("oos_sharpe", 0)
            sh_col = "#4CAF50" if sh > 1 else ("#FFA726" if sh > 0.5 else "#EF5350")
            params_str = ", ".join(f"{k}={v}" for k, v in (w.get("best_params") or {}).items())
            oos_start = w.get("oos_start")
            oos_end   = w.get("oos_end")
            period = (
                f"{oos_start.strftime('%b %Y')}→{oos_end.strftime('%b %Y')}"
                if oos_start and oos_end else "—"
            )
            table_rows += (
                f"<tr><td>W{w['window_number']} ({period})</td>"
                f"<td style='font-size:11px;color:#9E9E9E'>{params_str or '—'}</td>"
                f"<td style='color:{sh_col}'>{sh:.3f}</td>"
                f"<td>{w.get('oos_cagr', 0)*100:.1f}%</td>"
                f"<td>{w.get('oos_max_drawdown', 0)*100:.1f}%</td>"
                f"<td>{w.get('oos_num_trades', 0)}</td>"
                f"</tr>"
            )
        param_table = (
            '<div class="card"><div class="section-title">Per-Window Results</div>'
            '<table>'
            '<tr><th>Window</th><th>Best Params</th><th>OOS Sharpe</th>'
            '<th>OOS CAGR</th><th>Max DD</th><th>Trades</th></tr>'
            + table_rows
            + '</table></div>'
        ) if table_rows else ""

        # Narrative
        wfe_narr = ""
        if wfe is not None:
            wfe_narr = self._narrative(
                f"WFE score of {wfe:.1f}% means that for every 1% return generated in training, "
                f"the strategy produced {wfe/100:.2f}% out-of-sample. "
                + (f"Consistency of {cons_score:.1f}%: your strategy was profitable in "
                   f"{cons_score:.0f}% of OOS windows." if cons_score is not None else "")
            )

        n_neg = sum(1 for w in wins if w.get("oos_sharpe", 0) < 0)
        if n_neg:
            wfe_narr += self._warning_box(
                f"{n_neg} of {len(wins)} OOS windows had negative Sharpe. "
                "Review those windows for structural breaks."
            )

        label = agg.get("wfa_label", agg.get("wfe_label", ""))
        subtitle = f"{len(wins)} windows | {label}" if label else f"{len(wins)} windows"
        content = metrics_html + bar_chart + param_table + wfe_narr
        return self._build_page_html("Walk-Forward Analysis", subtitle, content)

    # ──────────────────────────────────────────────────────────────────────────
    # Page 3 — Monte Carlo
    # ──────────────────────────────────────────────────────────────────────────

    def _build_mc_page(self) -> str:
        if not self._mc:
            return self._not_run_page("Monte Carlo Simulation")

        mA    = self._mc["reshuffle"]
        mB    = self._mc["resample"]
        shared = self._mc["shared"]
        n_sims = shared.get("n_simulations", 5000)

        def mc_metrics(m, label):
            return (
                f'<div class="card" style="flex:1;margin:0 6px">'
                f'<div class="section-title">{label}</div>'
                '<div class="metric-grid" style="grid-template-columns:repeat(2,1fr)">'
                + self._metric_card("RoR %",       f"{m['ror_pct']:.1f}%",
                                     "positive" if m['ror_pct'] < 5 else ("amber" if m['ror_pct'] < 15 else "negative"))
                + self._metric_card("Sharpe P50",   f"{m['sharpe_p50']:.3f}")
                + self._metric_card("DD P95",        f"{m['dd_p95']:.1f}%",
                                     "positive" if m['dd_p95'] < 15 else ("amber" if m['dd_p95'] < 30 else "negative"))
                + self._metric_card("Sharpe Rank",  f"{m['sharpe_percentile_rank']:.1f}th",
                                     "positive" if m['sharpe_percentile_rank'] > 60 else ("amber" if m['sharpe_percentile_rank'] > 40 else "negative"))
                + '</div>'
                + self._verdict_badge(m["verdict"])
                + '</div>'
            )

        metrics_row = (
            '<div style="display:flex;gap:0;margin-bottom:16px">'
            + mc_metrics(mA, "Reshuffle (Method A)")
            + mc_metrics(mB, "Resample (Method B)")
            + '</div>'
        )

        # Approximate equity fan from percentile final returns
        n_steps = max(shared.get("n_trades", 30), 5)
        x_axis = list(range(n_steps + 1))

        def _fan_curve(ret_pct: float) -> list[float]:
            r = ret_pct / 100.0
            step_r = (1.0 + r) ** (1.0 / n_steps) - 1.0
            return [float((1.0 + step_r) ** t) for t in range(n_steps + 1)]

        fan_traces = []
        for pct, name, dash in [
            (mA["ret_p5"], "P5 (worst 5%)", "dot"),
            (mA["ret_p95"], "P95 (best 5%)", "dot"),
            (mA["ret_p50"], "P50 (median)", "solid"),
        ]:
            fan_traces.append({
                "type": "scatter", "mode": "lines", "name": name,
                "x": x_axis, "y": _fan_curve(pct),
                "line": {"color": "#E53935", "dash": dash,
                         "width": 1 if dash == "dot" else 2},
            })

        fig_fan = {
            "data": fan_traces,
            "layout": _merge_layout({
                "title": {"text": "Monte Carlo Equity Fan (Reshuffle) — Approximate P5/P50/P95"},
                "xaxis": {**_DARK["xaxis"], "title": "Trade Number"},
                "yaxis": {**_DARK["yaxis"], "title": "Portfolio Multiple (×1.0 start)"},
            }),
        }
        fan_chart = '<div class="card">' + self._plotly_chart(fig_fan, "mc_fan", "220px") + '</div>'

        # DD distribution histogram (both methods)
        dd_A = [float(v) for v in mA["drawdowns"]]
        dd_B = [float(v) for v in mB["drawdowns"]]
        fig_dd = {
            "data": [
                {"type": "histogram", "x": dd_A, "name": "Reshuffle", "opacity": 0.65,
                 "marker": {"color": "#E53935"}, "nbinsx": 50},
                {"type": "histogram", "x": dd_B, "name": "Resample",  "opacity": 0.50,
                 "marker": {"color": "#546E7A"}, "nbinsx": 50},
                {"type": "scatter", "mode": "lines", "name": "Actual DD",
                 "x": [mA["dd_backtest"], mA["dd_backtest"]],
                 "y": [0, len(dd_A) // 10], "line": {"color": "white", "width": 2}},
                {"type": "scatter", "mode": "lines", "name": "P95 DD",
                 "x": [mA["dd_p95"], mA["dd_p95"]],
                 "y": [0, len(dd_A) // 10], "line": {"color": "#E53935", "dash": "dash", "width": 1.5}},
            ],
            "layout": _merge_layout({
                "title": {"text": "Max Drawdown Distribution"},
                "barmode": "overlay",
                "xaxis": {**_DARK["xaxis"], "title": "Max Drawdown (%)"},
                "yaxis": {**_DARK["yaxis"], "title": "Count"},
            }),
        }
        dd_chart = '<div class="card">' + self._plotly_chart(fig_dd, "mc_dd", "210px") + '</div>'

        narr = self._narrative(
            f"Your strategy was simulated {n_sims:,} times by reshuffling trade order. "
            f"In {mA['ror_pct']:.1f}% of alternative histories it would have hit the ruin threshold. "
            f"Your actual Sharpe sits at the {mA['sharpe_percentile_rank']:.1f}th percentile — "
            f"meaning {mA['sharpe_percentile_rank']:.0f}% of random orderings of your own trades "
            f"produced a lower Sharpe. "
            + ("This strongly suggests skill over luck." if mA["sharpe_percentile_rank"] > 70
               else "This result may partly reflect lucky sequencing." if mA["sharpe_percentile_rank"] > 40
               else "This result is likely driven by lucky sequencing.")
        )

        content = metrics_row + fan_chart + dd_chart + narr
        return self._build_page_html(
            "Monte Carlo Simulation",
            f"{n_sims:,} simulations | Reshuffle + Resample",
            content
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Page 4 — Parameter Sensitivity
    # ──────────────────────────────────────────────────────────────────────────

    def _build_sensitivity_page(self) -> str:
        if not self._sens:
            return self._not_run_page("Parameter Sensitivity")

        rob  = self._sens.get("robustness_score", 0)
        peak = self._sens.get("peak_sharpe", 0)
        opt  = self._sens.get("optimal_sharpe", 0)
        verdict = self._sens.get("verdict", "—")
        df   = self._sens.get("results_df")
        n_combos = len(df) if df is not None else "—"

        pa = self._sens.get("plateau_analysis", {})
        primary_param = list(pa.keys())[0] if pa else None
        plateau_pct = pa[primary_param].get("plateau_pct", 0) if primary_param else 0

        metrics_html = (
            '<div class="metric-grid">'
            + self._metric_card("Robustness Score", f"{rob*100:.1f}%",
                                 "positive" if rob > 0.5 else ("amber" if rob > 0.25 else "negative"))
            + self._metric_card("Peak Sharpe",  f"{peak:.3f}")
            + self._metric_card("Optimal Sharpe", f"{opt:.3f}" if opt is not None else "N/A")
            + self._metric_card("Plateau Width",  f"{plateau_pct:.1f}%",
                                 "positive" if plateau_pct > 50 else ("amber" if plateau_pct > 25 else "negative"))
            + '</div>'
        )

        # Interactive Sharpe landscape.
        #   1 param  -> line chart      2 params -> heatmap
        #   3+ params -> pairwise heatmaps (others held at the optimal combo)
        # The optimal (peak-Sharpe) combination is marked with a white star.
        param_names = list(pa.keys())                       # reliable param names
        peak_params = self._sens.get("peak_sharpe_params", {}) or {}
        heatmap_html = ""

        if df is not None and not df.empty and "sharpe" in df.columns and param_names:
            if len(param_names) == 1:
                heatmap_html = self._sens_line_fig(df, param_names[0], peak_params, "sens_1d")
            elif len(param_names) == 2:
                p1, p2 = param_names[0], param_names[1]
                heatmap_html = self._sens_heatmap_fig(df, p1, p2, {}, peak_params, "sens_heat")
            else:
                pairs = [
                    (param_names[i], param_names[j])
                    for i in range(len(param_names))
                    for j in range(i + 1, len(param_names))
                ]
                blocks = []
                for k, (p1, p2) in enumerate(pairs):
                    hold = {hp: peak_params.get(hp) for hp in param_names if hp not in (p1, p2)}
                    blocks.append(self._sens_heatmap_fig(df, p1, p2, hold, peak_params, f"sens_heat_{k}"))
                heatmap_html = "".join(blocks)

        narr = self._narrative(
            f"Your strategy was tested across {n_combos} parameter combinations. "
            f"{rob*100:.0f}% of combinations produced a Sharpe within 20% of the peak, "
            f"indicating a {'wide plateau' if rob > 0.5 else 'narrow spike'}. "
            + ("The strategy does not heavily depend on precise parameter tuning." if rob > 0.5
               else "Results are sensitive to parameter choice — tune carefully." if rob > 0.25
               else "The strategy depends critically on precise parameters — high overfitting risk.")
        )

        badge = self._verdict_badge(verdict)
        content = metrics_html + f'<div class="card" style="margin-bottom:16px">{badge}</div>' + heatmap_html + narr
        return self._build_page_html(
            "Parameter Sensitivity",
            f"{n_combos} combinations tested",
            content
        )

    _SHARPE_SCALE = [[0, "#B71C1C"], [0.5, "#FFA726"], [1, "#4CAF50"]]

    def _sens_heatmap_fig(self, df, p1: str, p2: str, hold: dict,
                          star: dict, div_id: str) -> str:
        """Interactive Sharpe heatmap for (p1, p2). Other params held at `hold`.
        Optimal combo marked with a white star. Returns a card HTML block."""
        sub = df
        for hp, hv in hold.items():
            if hv is not None:
                sub = sub[sub[hp] == hv]
        pivot = sub.pivot_table(index=p2, columns=p1, values="sharpe", aggfunc="mean")
        xvals = [str(v) for v in pivot.columns.tolist()]
        yvals = [str(v) for v in pivot.index.tolist()]
        zvals = [[(None if v != v else float(v)) for v in row] for row in pivot.values.tolist()]

        data = [{
            "type": "heatmap", "z": zvals, "x": xvals, "y": yvals,
            "colorscale": self._SHARPE_SCALE, "colorbar": {"title": "Sharpe"},
            "hoverongaps": False,
            "hovertemplate": f"{p1}=%{{x}}<br>{p2}=%{{y}}<br>Sharpe=%{{z:.3f}}<extra></extra>",
        }]
        sx = star.get(p1); sy = star.get(p2)
        if sx is not None and sy is not None and str(sx) in xvals and str(sy) in yvals:
            data.append({
                "type": "scatter", "mode": "markers+text",
                "x": [str(sx)], "y": [str(sy)],
                "marker": {"symbol": "star", "size": 20, "color": "white",
                           "line": {"color": "#000", "width": 1}},
                "text": ["★ optimal"], "textposition": "top center",
                "textfont": {"color": "white", "size": 10}, "showlegend": False,
                "hovertemplate": f"Optimal<br>{p1}=%{{x}}<br>{p2}=%{{y}}<extra></extra>",
            })

        title = f"Sharpe Heatmap — {p1} × {p2}"
        if hold:
            title += "  (" + ", ".join(f"{k}={v}" for k, v in hold.items() if v is not None) + ")"
        fig = {
            "data": data,
            "layout": _merge_layout({
                "title": {"text": title, "font": {"size": 12}},
                "xaxis": {**_DARK["xaxis"], "title": p1, "type": "category"},
                "yaxis": {**_DARK["yaxis"], "title": p2, "type": "category"},
            }),
        }
        return '<div class="card">' + self._plotly_chart(fig, div_id, "300px") + '</div>'

    def _sens_line_fig(self, df, p: str, star: dict, div_id: str) -> str:
        """Single-parameter Sharpe line chart with the optimal value starred."""
        grouped = df.groupby(p)["sharpe"].mean().reset_index().sort_values(p)
        xs = [float(v) for v in grouped[p].tolist()]
        ys = [float(v) for v in grouped["sharpe"].tolist()]
        data = [{
            "type": "scatter", "mode": "lines+markers", "x": xs, "y": ys,
            "line": {"color": "#E53935", "width": 2},
            "marker": {"color": "#E53935", "size": 6}, "name": "Sharpe",
            "hovertemplate": f"{p}=%{{x}}<br>Sharpe=%{{y:.3f}}<extra></extra>",
        }]
        sx = star.get(p)
        if sx is not None:
            sy_series = grouped.loc[grouped[p] == sx, "sharpe"]
            sy = float(sy_series.iloc[0]) if not sy_series.empty else (max(ys) if ys else 0.0)
            data.append({
                "type": "scatter", "mode": "markers+text", "x": [float(sx)], "y": [sy],
                "marker": {"symbol": "star", "size": 20, "color": "white",
                           "line": {"color": "#000", "width": 1}},
                "text": ["★ optimal"], "textposition": "top center",
                "textfont": {"color": "white", "size": 10}, "showlegend": False,
                "hovertemplate": f"Optimal {p}=%{{x}}<extra></extra>",
            })
        fig = {
            "data": data,
            "layout": _merge_layout({
                "title": {"text": f"Sharpe vs {p}", "font": {"size": 12}},
                "xaxis": {**_DARK["xaxis"], "title": p},
                "yaxis": {**_DARK["yaxis"], "title": "Sharpe"},
            }),
        }
        return '<div class="card">' + self._plotly_chart(fig, div_id, "280px") + '</div>'

    # ──────────────────────────────────────────────────────────────────────────
    # Page 5 — Regime Analysis
    # ──────────────────────────────────────────────────────────────────────────

    def _build_regime_page(self) -> str:
        if not self._reg:
            return self._not_run_page("Regime Analysis")

        agg     = self._reg.get("aggregate", {})
        results = self._reg.get("regime_results", {})
        verdict = self._reg.get("verdict", "—")
        wfa_eq  = self._wfa_agg().get("equity_curve")

        profitable = agg.get("profitable_regimes", 0)
        total_rep  = agg.get("total_reported", 0)
        worst      = agg.get("worst_regime") or {}
        bear_sh    = agg.get("bear_sharpe")
        conc       = agg.get("profit_concentration")

        metrics_html = (
            '<div class="metric-grid">'
            + self._metric_card("Profitable Regimes", f"{profitable}/{total_rep}",
                                 "positive" if total_rep > 0 and profitable / max(total_rep, 1) >= 0.7 else "amber")
            + self._metric_card("Worst Regime Sharpe",
                                 f"{worst.get('sharpe', 0):.3f}" if worst else "—",
                                 "positive" if (worst.get("sharpe") or 0) > 0 else "negative")
            + self._metric_card("Profit Concentration",
                                 f"{conc:.1f}%" if conc is not None else "—",
                                 "positive" if (conc or 0) < 50 else ("amber" if (conc or 0) < 70 else "negative"))
            + self._metric_card("Bear/Crash Sharpe",
                                 f"{bear_sh:.3f}" if bear_sh is not None else "—",
                                 "positive" if (bear_sh or -1) > 0 else "negative")
            + '</div>'
        )

        # Equity overlay with regime bands
        overlay_html = ""
        if wfa_eq is not None and not wfa_eq.empty:
            dates = [d.strftime("%Y-%m-%d") for d in wfa_eq.index]
            vals  = [float(v) for v in wfa_eq.values]
            traces = [{"type": "scatter", "mode": "lines", "x": dates, "y": vals,
                       "line": {"color": "white", "width": 1.5}, "name": "OOS Equity"}]
            shapes = []
            _REG_COLS = {"bull": "rgba(76,175,80,0.15)", "bear": "rgba(229,57,53,0.15)",
                         "crash": "rgba(183,28,28,0.25)", "recovery": "rgba(66,165,245,0.15)",
                         "sideways": "rgba(255,167,38,0.15)"}
            for rname, rres in results.items():
                reg = rres.get("regime", {})
                if not reg:
                    continue
                rtype = reg.get("type", "")
                r_start = reg.get("start")
                r_end   = reg.get("end")
                if r_start and r_end:
                    shapes.append({
                        "type": "rect", "xref": "x", "yref": "paper",
                        "x0": r_start.strftime("%Y-%m-%d"), "x1": r_end.strftime("%Y-%m-%d"),
                        "y0": 0, "y1": 1,
                        "fillcolor": _REG_COLS.get(rtype, "rgba(158,158,158,0.1)"),
                        "line": {"width": 0},
                    })
            fig = {
                "data": traces,
                "layout": _merge_layout({
                    "title": {"text": "OOS Equity — Regime Overlay"},
                    "shapes": shapes,
                    "xaxis": {**_DARK["xaxis"], "title": "Date"},
                    "yaxis": {**_DARK["yaxis"], "title": "Portfolio Value (₹)", "tickprefix": "₹"},
                }),
            }
            overlay_html = '<div class="card">' + self._plotly_chart(fig, "reg_eq", "230px") + '</div>'

        # Regime Sharpe bar chart
        computed = [(n, r) for n, r in results.items() if r.get("status") == "computed"]
        bar_html = ""
        if computed:
            _COL = {"bull": "#4CAF50", "bear": "#E53935", "crash": "#B71C1C",
                    "recovery": "#42A5F5", "sideways": "#FFA726"}
            names   = [n for n, _ in computed]
            sharpes = [float((r.get("sharpe") or 0)) for _, r in computed]
            colours = [_COL.get(r.get("regime", {}).get("type", ""), "#9E9E9E") for _, r in computed]
            fig = {
                "data": [{"type": "bar", "orientation": "h", "x": sharpes, "y": names,
                          "marker": {"color": colours}, "text": [f"n={r.get('num_trades',0)}" for _,r in computed],
                          "textposition": "outside"}],
                "layout": _merge_layout({
                    "title": {"text": "Sharpe Ratio by Regime"},
                    "xaxis": {**_DARK["xaxis"], "title": "Sharpe Ratio",
                              "zeroline": True, "zerolinecolor": "#444"},
                    "yaxis": {**_DARK["yaxis"]},
                    "margin": {**_DARK["margin"], "l": 160},
                }),
            }
            bar_html = '<div class="card">' + self._plotly_chart(fig, "reg_bars", "200px") + '</div>'

        # Regime narratives
        narr_text = ""
        for rname, rres in results.items():
            if rres.get("status") != "computed":
                continue
            reg = rres.get("regime", {})
            r_start = reg.get("start")
            r_end   = reg.get("end")
            period = (f"({r_start.strftime('%b %Y')} — {r_end.strftime('%b %Y')})"
                      if r_start and r_end else "")
            sh = rres.get("sharpe")
            ct = rres.get("contribution_pct")
            narr_text += (
                f"<b>{rname}</b> {period}: {rres.get('num_trades', 0)} trades, "
                f"Sharpe {sh:.2f}" + (f", contributed {ct:.1f}% of total profit." if ct is not None else ".") + "<br>"
            )
        if narr_text:
            narr_text = self._narrative(narr_text)

        badge = self._verdict_badge(verdict)
        content = (metrics_html
                   + f'<div class="card" style="margin-bottom:16px">{badge}</div>'
                   + overlay_html + bar_html + narr_text)
        n_regimes = len(results)
        return self._build_page_html(
            "Regime Analysis", f"{n_regimes} regimes | Full OOS period", content
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Page 6 — Statistical Significance
    # ──────────────────────────────────────────────────────────────────────────

    def _build_stats_page(self) -> str:
        if not self._stat:
            return self._not_run_page("Statistical Significance")

        s = self._stat
        sig_level = self._cfg.get("stats", {}).get("significance_level", 0.05)

        metrics_html = (
            '<div class="metric-grid">'
            + self._metric_card("t-test p-value",   f"{s['p_value_ttest']:.4f}",
                                 "positive" if s['p_value_ttest'] < sig_level else "negative")
            + self._metric_card("Permutation p",    f"{s['p_value_permutation']:.4f}",
                                 "positive" if s['p_value_permutation'] < sig_level else "negative")
            + self._metric_card("Cohen's d",        f"{s['cohens_d']:.3f} ({s['cohens_d_size']})")
            + self._metric_card("Statistical Power", f"{s['power']:.2f}",
                                 "positive" if s['power'] >= 0.8 else ("amber" if s['power'] >= 0.5 else "negative"))
            + '</div>'
        )

        # Permutation null distribution
        null_dist = [float(v) for v in s["null_distribution"]]
        obs_mean  = float(s["mean_return"])
        n_iters   = len(null_dist)

        # Rejection region thresholds from empirical distribution
        lo_thresh = float(np.percentile(null_dist, sig_level / 2 * 100))
        hi_thresh = float(np.percentile(null_dist, (1 - sig_level / 2) * 100))

        fig_null = {
            "data": [
                {"type": "histogram", "x": null_dist, "nbinsx": 60, "name": "Null distribution",
                 "marker": {"color": "#546E7A"}, "opacity": 0.85},
                {"type": "scatter", "mode": "lines", "name": "Rejection region (low)",
                 "x": [min(null_dist), lo_thresh, lo_thresh, min(null_dist)],
                 "y": [0, 0, n_iters // 8, n_iters // 8],
                 "fill": "toself", "fillcolor": "rgba(229,57,53,0.25)", "line": {"width": 0}},
                {"type": "scatter", "mode": "lines", "name": "Rejection region (high)",
                 "x": [hi_thresh, max(null_dist), max(null_dist), hi_thresh],
                 "y": [0, 0, n_iters // 8, n_iters // 8],
                 "fill": "toself", "fillcolor": "rgba(229,57,53,0.25)", "line": {"width": 0}},
                {"type": "scatter", "mode": "lines", "name": f"Observed mean ({obs_mean*100:.3f}%)",
                 "x": [obs_mean, obs_mean], "y": [0, n_iters // 5],
                 "line": {"color": "white", "width": 2}},
            ],
            "layout": _merge_layout({
                "title": {"text": "Permutation Null Distribution vs Observed Mean"},
                "xaxis": {**_DARK["xaxis"], "title": "Permuted mean trade return"},
                "yaxis": {**_DARK["yaxis"], "title": "Count"},
            }),
        }
        null_chart = '<div class="card">' + self._plotly_chart(fig_null, "stat_null", "220px") + '</div>'

        # Return distribution
        mean_r = float(s["mean_return"])
        std_r  = float(s["std_return"])
        ci_lo  = float(s["ci_95"][0])
        ci_hi  = float(s["ci_95"][1])
        x_norm = [mean_r - 3 * std_r + (6 * std_r * i / 200) for i in range(201)]
        from scipy.stats import norm as _sp_norm
        y_norm = [float(_sp_norm.pdf(x, mean_r, std_r)) if std_r > 0 else 0.0 for x in x_norm]

        # For histogram we need the actual returns
        _tl = self._wfa_agg().get("full_trade_log")
        returns_raw = _tl["return_pct"] if (_tl is not None and "return_pct" in _tl.columns) else pd.Series(dtype=float)
        ret_list = [float(v) for v in returns_raw.values] if not returns_raw.empty else []

        ret_traces = []
        if ret_list:
            ret_traces.append({"type": "histogram", "x": ret_list, "nbinsx": 40, "name": "Trade returns",
                               "marker": {"color": "#1565C0"}, "opacity": 0.85, "histnorm": "probability density"})
        if std_r > 0:
            ret_traces.append({"type": "scatter", "mode": "lines", "name": "Fitted normal",
                               "x": x_norm, "y": y_norm, "line": {"color": "#42A5F5", "width": 2}})
        ret_traces += [
            {"type": "scatter", "mode": "lines", "name": "Zero",
             "x": [0, 0], "y": [0, max(y_norm) if y_norm else 0.1],
             "line": {"color": "#E53935", "dash": "dash", "width": 1.5}},
            {"type": "scatter", "mode": "lines", "name": "95% CI lo",
             "x": [ci_lo, ci_lo], "y": [0, max(y_norm) if y_norm else 0.1],
             "line": {"color": "white", "dash": "dash", "width": 1.2}},
            {"type": "scatter", "mode": "lines", "name": "95% CI hi",
             "x": [ci_hi, ci_hi], "y": [0, max(y_norm) if y_norm else 0.1],
             "line": {"color": "white", "dash": "dash", "width": 1.2}},
        ]
        fig_ret = {
            "data": ret_traces,
            "layout": _merge_layout({
                "title": {"text": "OOS Trade Return Distribution"},
                "xaxis": {**_DARK["xaxis"], "title": "Return (decimal)"},
                "yaxis": {**_DARK["yaxis"], "title": "Density"},
            }),
        }
        ret_chart = '<div class="card">' + self._plotly_chart(fig_ret, "stat_ret", "210px") + '</div>'

        # Narrative
        sig_word = "can" if (s["significant_ttest"] and s["significant_permutation"]) else "cannot"
        power_comment = (
            "You have adequate statistical power."
            if s["power"] >= 0.8
            else f"You would need at least {s.get('min_n_for_power') or '?'} trades for 80% power."
        )
        narr = self._narrative(
            f"Mean trade return of {mean_r*100:.3f}%. "
            f"The t-test gives p = {s['p_value_ttest']:.4f}: we {sig_word} reject the null "
            f"hypothesis at the {sig_level*100:.0f}% level. "
            f"Cohen's d = {s['cohens_d']:.2f} ({s['cohens_d_size']} effect). "
            f"Statistical power = {s['power']:.2f}. {power_comment}"
        )
        if s.get("low_power_warning"):
            narr += self._warning_box(f"Low power: only {s['n_trades']} trades. Collect more data.")
        if not s.get("is_normal"):
            narr += self._warning_box("Returns are non-normal. Weight permutation test result more heavily.")

        badge = self._verdict_badge(s["verdict"])
        content = metrics_html + f'<div class="card" style="margin-bottom:16px">{badge}</div>' + null_chart + ret_chart + narr
        return self._build_page_html(
            "Statistical Significance",
            f"One-sample t-test + Permutation test | α = {sig_level}",
            content
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Page 7 — Master Verdict (tiered system)
    # ──────────────────────────────────────────────────────────────────────────

    def _build_verdict_page(self) -> str:
        v = self._verdict or {}
        s = self._sizing  or {}

        tier       = v.get("tier", 0)
        label      = v.get("label", "INCONCLUSIVE")
        colour     = v.get("colour", "grey")
        summary    = v.get("summary", "Verdict not computed.")
        gates      = v.get("gates", {})
        diagnoses  = v.get("diagnoses", [])
        next_steps = v.get("next_steps", [])
        sufficient = v.get("sufficient_trades", False)
        n_oos      = v.get("total_oos_trades", 0)

        # Tier → background colour
        _TIER_BG = {
            1: "#4CAF50", 2: "#FFA726", 3: "#FF7043", 4: "#E53935", 0: "#616161"
        }
        _TIER_FG = {1: "white", 2: "black", 3: "white", 4: "white", 0: "white"}
        tier_bg = _TIER_BG.get(tier, "#616161")
        tier_fg = _TIER_FG.get(tier, "white")

        # ── Insufficient trades warning banner ────────────────────────────────
        warn_banner = ""
        if not sufficient:
            warn_banner = (
                f'<div style="background:#B71C1C;color:white;padding:12px 16px;'
                f'border-radius:6px;margin-bottom:12px;font-size:12px;font-weight:bold;text-align:center">'
                f'⚠ INCONCLUSIVE: Only {n_oos} OOS trades. Minimum 30 required. '
                f'All results below are statistically unreliable.'
                f'</div>'
            )

        # ── Tier badge ────────────────────────────────────────────────────────
        gates_passed = v.get("gates_passed", 0)
        glow_class = " tier1-glow" if tier == 1 else ""
        badge_block = f"""
<div style="text-align:center;padding:12px 0 10px">
  <div class="{glow_class.strip()}" style="display:inline-block;background:{tier_bg};color:{tier_fg};
              font-size:22px;font-weight:bold;padding:10px 36px;border-radius:8px;
              letter-spacing:2px;text-transform:uppercase">
    TIER {tier} &nbsp;—&nbsp; {label}
  </div>
  <p style="font-size:12px;color:#BDBDBD;margin-top:10px;max-width:700px;
            margin-left:auto;margin-right:auto">{summary}</p>
  <p style="font-size:11px;color:#757575;margin-top:4px">{gates_passed}/5 gates passed</p>
</div>"""

        # ── Gate scorecard ────────────────────────────────────────────────────
        gate_rows = ""
        gate_defs = [
            ("Walk-Forward Analysis", "wfa"),
            ("Monte Carlo",           "monte_carlo"),
            ("Parameter Sensitivity", "sensitivity"),
            ("Regime Analysis",       "regime"),
            ("Statistical Tests",     "stats"),
        ]
        for module_name, key in gate_defs:
            g = gates.get(key, {})
            passed = g.get("passed", False)
            detail = g.get("detail", "—")
            icon   = "✓" if passed else "✗"
            color  = "#4CAF50" if passed else "#E53935"
            result_txt = "PASS" if passed else "FAIL"
            gate_rows += (
                f"<tr>"
                f"<td>{module_name}</td>"
                f"<td style='color:{color};font-weight:bold'>{result_txt}</td>"
                f"<td style='color:#9E9E9E'>{detail}</td>"
                f"<td style='color:{color};font-size:16px;text-align:center'>{icon}</td>"
                f"</tr>"
            )

        scorecard = (
            '<div class="card"><div class="section-title">Gate Scorecard</div>'
            '<table>'
            '<tr><th>Module</th><th>Result</th><th>Detail</th><th style="text-align:center">Status</th></tr>'
            + gate_rows
            + '</table></div>'
        )

        # ── Diagnoses (only if any gates failed) ──────────────────────────────
        diagnoses_card = ""
        if diagnoses:
            items = "".join(f"<li style='margin-bottom:6px'>{d}</li>" for d in diagnoses)
            diagnoses_card = (
                '<div class="card" style="border-left:3px solid #E53935;margin-bottom:10px">'
                '<div class="section-title" style="color:#EF5350">What Went Wrong</div>'
                f'<ul style="padding-left:18px;font-size:12px;color:#BDBDBD;line-height:1.6">{items}</ul>'
                '</div>'
            )

        # ── Next steps ────────────────────────────────────────────────────────
        ns_border = "#4CAF50" if tier == 1 else ("#FFA726" if tier == 2 else "#E53935")
        if next_steps:
            items = "".join(f"<li style='margin-bottom:6px'>{ns}</li>" for ns in next_steps)
            next_steps_card = (
                f'<div class="card" style="border-left:3px solid {ns_border};margin-bottom:10px">'
                '<div class="section-title">What To Do Next</div>'
                f'<ul style="padding-left:18px;font-size:12px;color:#BDBDBD;line-height:1.6">{items}</ul>'
                '</div>'
            )
        else:
            next_steps_card = ""

        # ── Kelly position sizing card ────────────────────────────────────────
        kelly_card = ""
        if tier in (1, 2) and s:
            if s.get("no_edge_flag"):
                kelly_card = (
                    '<div class="card" style="border-left:3px solid #E53935">'
                    '<div class="section-title">Suggested Position Sizing (Quarter Kelly)</div>'
                    '<div class="warning-box">Kelly formula returns 0 — strategy has no mathematical edge '
                    'despite passing other gates. Do not size positions using this output.</div>'
                    '</div>'
                )
            else:
                rec_pct = s.get("recommended_pct", 0)
                rec_inr = s.get("recommended_inr", 0)
                tot_inr = s.get("total_exposure_inr", 0)
                res_inr = s.get("cash_reserve_inr", 0)
                exp_txt = s.get("explanation", "")
                warn_txt = ""
                if s.get("capital_warning"):
                    warn_txt = '<div class="warning-box">Full Kelly exceeds capital with max positions. Use Half Kelly or Quarter Kelly.</div>'

                kelly_card = f"""
<div class="card" style="border-left:3px solid {ns_border}">
  <div class="section-title">Suggested Position Sizing (Quarter Kelly)</div>
  <div class="metric-grid" style="grid-template-columns:repeat(3,1fr);margin-bottom:10px">
    <div class="metric-card">
      <div class="metric-value" style="color:#4CAF50">₹{rec_inr:,.0f}</div>
      <div class="metric-label">Per Trade ({rec_pct:.1f}%)</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:#FFA726">₹{tot_inr:,.0f}</div>
      <div class="metric-label">Total Exposure</div>
    </div>
    <div class="metric-card">
      <div class="metric-value" style="color:#9E9E9E">₹{res_inr:,.0f}</div>
      <div class="metric-label">Cash Reserve</div>
    </div>
  </div>
  <p style="font-size:11px;color:#757575;line-height:1.5">{exp_txt}</p>
  {warn_txt}
</div>"""

        elif tier in (3, 4):
            kelly_card = (
                '<div class="card"><div style="font-size:12px;color:#757575;padding:8px 0">'
                'Position sizing not shown — fix strategy issues first.'
                '</div></div>'
            )

        content = warn_banner + badge_block + scorecard + diagnoses_card + next_steps_card + kelly_card
        return self._build_page_html("Master Verdict", "Full validation summary", content)

    # ──────────────────────────────────────────────────────────────────────────
    # Utility: not-run placeholder page
    # ──────────────────────────────────────────────────────────────────────────

    def _not_run_page(self, module_name: str) -> str:
        content = (
            '<div class="card" style="text-align:center;padding:60px;opacity:0.6">'
            '<div style="font-size:48px;margin-bottom:16px">—</div>'
            f'<div style="font-size:18px;color:#9E9E9E">{module_name} was not run</div>'
            '<div style="font-size:13px;color:#757575;margin-top:8px">'
            'Run this module and pass its results to launch_dashboard() to see results here.'
            '</div></div>'
        )
        return self._build_page_html(module_name, "Module not run", content)

    # ──────────────────────────────────────────────────────────────────────────
    # Verdict derivation helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _derive_wfa_verdict(self) -> str:
        agg = self._wfa_agg()
        if not agg:
            return "NOT RUN"
        label = agg.get("wfe_label", "")
        if "Robust" in label:
            return "ROBUST"
        if "Acceptable" in label:
            return "MARGINAL"
        if label:
            return "FRAGILE"
        return "—"

    def _overall_verdict(self, *verdicts) -> str:
        _BAD = {"FRAGILE", "NOT SIGNIFICANT", "LUCKY"}
        _GOOD = {"ROBUST", "STATISTICALLY ROBUST"}
        present = [v for v in verdicts if v and v not in ("NOT RUN", "—", None)]
        if not present:
            return "CONDITIONAL"
        if any(v in _BAD for v in present):
            return "DO NOT TRADE"
        if all(v in _GOOD for v in present):
            return "TRADEABLE"
        return "CONDITIONAL"


# ──────────────────────────────────────────────────────────────────────────────
# Qt stylesheets
# ──────────────────────────────────────────────────────────────────────────────

_QT_STYLESHEET = """
QMainWindow { background: #0A0A0A; }
QSplitter { background: #0A0A0A; }
QSplitter::handle { background: #1E1E1E; width: 1px; }
QScrollBar:vertical { background: #111; width: 8px; border-radius: 4px; }
QScrollBar::handle:vertical { background: #333; border-radius: 4px; min-height: 20px; }
QScrollBar::handle:vertical:hover { background: #E53935; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal { background: #111; height: 8px; border-radius: 4px; }
QScrollBar::handle:horizontal { background: #333; border-radius: 4px; min-width: 20px; }
QScrollBar::handle:horizontal:hover { background: #E53935; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
"""

_NAV_BTN_INACTIVE = """
QPushButton {
    background: transparent;
    color: #9E9E9E;
    border: none;
    border-left: 3px solid transparent;
    text-align: left;
    padding-left: 20px;
    font-size: 13px;
}
QPushButton:hover {
    background: rgba(255,255,255,0.05);
    color: white;
}
"""

_NAV_BTN_ACTIVE = """
QPushButton {
    background: rgba(229,57,53,0.08);
    color: #E53935;
    border: none;
    border-left: 3px solid #E53935;
    text-align: left;
    padding-left: 20px;
    font-size: 13px;
    font-weight: bold;
}
"""
