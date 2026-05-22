"""
Internal long-only backtester shared between WFA and sensitivity analysis.

FALSIFY enforces T+1 entry HERE: a signal at close of bar T is executed at
the OPEN of bar T+1 via the pending-order mechanism. Callers pass RAW,
unshifted signals. Strategies must NOT shift internally.
"""
from __future__ import annotations

import pandas as pd

_EMPTY_TRADES = pd.DataFrame(
    columns=["entry_date", "exit_date", "entry_price",
             "exit_price", "return_pct", "direction"]
)


def backtest_signals(
    ohlcv:           pd.DataFrame,
    signals:         pd.Series,
    config:          dict,
    initial_capital: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Simulate a long-only strategy from pre-computed signals.

    ohlcv   : DataFrame with 'open' and 'close' columns, indexed by date.
    signals : Series aligned to ohlcv.index; 1 = long, else = flat.
              Pass RAW (unshifted) signals. The backtester executes a pending
              signal at the OPEN of the next bar, so a signal at close of bar T
              produces an entry at the open of bar T+1.
    config  : full config dict — costs read from config['costs'],
              starting capital from config['capital']['starting_capital'].
    initial_capital : override starting capital (used by WFA carry-forward).
                      If None, reads from config.

    Transaction costs applied on every entry and exit:
        Entry  : effective_buy  = open × (1 + brokerage + slippage)
        Exit   : effective_sell = open × (1 − brokerage − slippage)

    An open position at the end of the window is force-closed at the last
    day's close price (with transaction costs).

    Returns
    -------
    equity : pd.Series  — daily portfolio value at close, indexed by date
    trades : pd.DataFrame — columns: entry_date, exit_date, entry_price,
             exit_price, return_pct, direction
    """
    costs_cfg      = config.get("costs", {})
    brokerage      = float(costs_cfg.get("brokerage", 0.001))
    slippage       = float(costs_cfg.get("slippage",  0.0005))
    total_cost_pct = brokerage + slippage

    if initial_capital is None:
        initial_capital = float(config["capital"]["starting_capital"])

    if ohlcv.empty:
        return pd.Series(dtype=float), _EMPTY_TRADES.copy()

    signals = signals.reindex(ohlcv.index).fillna(0)

    cash          = float(initial_capital)
    shares        = 0.0
    in_pos        = False
    eff_entry     = float("nan")
    raw_entry     = float("nan")
    entry_date: pd.Timestamp | None = None
    pending: str | None = None

    equity_vals:   list[float] = []
    trade_records: list[dict]  = []

    for i in range(len(ohlcv)):
        dt       = ohlcv.index[i]
        open_px  = float(ohlcv["open"].iat[i])
        close_px = float(ohlcv["close"].iat[i])

        # ── OPEN: execute yesterday's pending order ───────────────────────────
        if pending == "enter" and not in_pos:
            eff_buy   = open_px * (1.0 + total_cost_pct)
            shares    = cash / eff_buy
            cash      = 0.0
            eff_entry = eff_buy
            raw_entry = open_px
            entry_date = dt
            in_pos    = True

        elif pending == "exit" and in_pos:
            eff_sell  = open_px * (1.0 - total_cost_pct)
            ret_pct   = (eff_sell / eff_entry) - 1.0
            trade_records.append({
                "entry_date":  entry_date,
                "exit_date":   dt,
                "entry_price": raw_entry,
                "exit_price":  open_px,
                "return_pct":  ret_pct,
                "direction":   "long",
            })
            cash      = shares * eff_sell
            shares    = 0.0
            in_pos    = False
            eff_entry = float("nan")
            raw_entry = float("nan")
            entry_date = None

        pending = None

        # ── MARK equity at close ──────────────────────────────────────────────
        equity_vals.append(shares * close_px + cash)

        # ── SIGNAL: set pending action for next bar ───────────────────────────
        sig = int(signals.iat[i])
        if sig == 1 and not in_pos:
            pending = "enter"
        elif sig != 1 and in_pos:
            pending = "exit"

    # ── Force-close any open position at last day's close ────────────────────
    if in_pos and shares > 0:
        last_close = float(ohlcv["close"].iat[-1])
        eff_sell   = last_close * (1.0 - total_cost_pct)
        ret_pct    = (eff_sell / eff_entry) - 1.0
        trade_records.append({
            "entry_date":  entry_date,
            "exit_date":   ohlcv.index[-1],
            "entry_price": raw_entry,
            "exit_price":  last_close,
            "return_pct":  ret_pct,
            "direction":   "long",
        })
        equity_vals[-1] = shares * eff_sell

    equity = pd.Series(equity_vals, index=ohlcv.index, name="equity")
    trades = pd.DataFrame(trade_records) if trade_records else _EMPTY_TRADES.copy()
    return equity, trades
