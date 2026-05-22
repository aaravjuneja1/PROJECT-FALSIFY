"""
Multi-position portfolio backtester for FALSIFY (portfolio mode).

Simulates a swing-trading book across many tickers at once:

  • Up to `max_positions` simultaneous long positions (config capital.max_positions).
  • PROPORTIONAL sizing per slot: position_size = starting_capital / max_positions,
    where starting_capital is this run's opening capital. WFA passes the prior
    window's OOS ending equity, so the slot size scales with the book across
    windows. It is computed ONCE at the start of the run and fixed for its
    duration — never resized mid-run by current equity or open-position count.
    Unfilled slots stay in cash (config capital.cash_when_idle).
  • T+1 entry: a signal at the close of bar T is executed at the OPEN of bar T+1
    via the same pending-order mechanism as the single-ticker backtester. Pass
    RAW (unshifted) signals.
  • When more buy signals fire than there are free slots, candidates are ranked
    by the SIGNAL-DAY volume (a liquidity proxy, no look-ahead) and the extras
    are skipped entirely (they are NOT re-queued).
  • Optional intraday stop-loss / take-profit read from the strategy
    (stop_loss_pct / take_profit_pct). Stop is checked first (worst case). Fills
    model gap risk: a long stop fills at min(open, stop); a target at
    max(open, target).
  • After a stop/target exit the ticker is blocked from re-entry until its signal
    goes flat and re-triggers (prevents whipsaw re-entry while still in-state).
    Same-day re-entry is never allowed.

Costs (config costs.brokerage + costs.slippage) are applied once per side:
    entry  fill = open  × (1 + cost)
    exit   fill = price × (1 − cost)

Returns
-------
equity : pd.Series   — daily portfolio value (cash + marked positions) at close
trades : pd.DataFrame — columns: entry_date, exit_date, entry_price, exit_price,
         return_pct, direction, ticker, exit_reason, pnl_abs
"""
from __future__ import annotations

import pandas as pd

_TRADE_COLS = [
    "entry_date", "exit_date", "entry_price", "exit_price",
    "return_pct", "direction", "ticker", "exit_reason", "pnl_abs",
]


def _isnan(x: float) -> bool:
    return x != x


def backtest_portfolio(
    prices:           dict[str, pd.DataFrame],
    signals:          dict[str, pd.Series],
    config:           dict,
    strategy=None,
    starting_capital: float | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    costs_cfg = config.get("costs", {})
    cost = float(costs_cfg.get("brokerage", 0.001)) + float(costs_cfg.get("slippage", 0.0005))

    cap_cfg        = config.get("capital", {})
    config_start   = float(cap_cfg.get("starting_capital", 1_000_000))
    max_positions  = int(cap_cfg.get("max_positions", 6))
    # Proportional sizing: slot = (this run's opening capital) / max_positions.
    # Computed once here; fixed for the run. WFA passes the prior window's OOS
    # ending equity, so the slot size scales with the book window to window.
    window_capital = float(starting_capital) if starting_capital is not None else config_start
    position_size  = window_capital / max_positions

    cash = window_capital

    sl_pct = strategy.stop_loss_pct()   if strategy is not None else None
    tp_pct = strategy.take_profit_pct() if strategy is not None else None

    if not prices:
        return pd.Series(dtype=float), pd.DataFrame(columns=_TRADE_COLS)

    # ── Build master index (union of all dates) and per-ticker numpy arrays ────
    master_index = None
    for df in prices.values():
        master_index = df.index if master_index is None else master_index.union(df.index)
    master_index = master_index.sort_values()
    n = len(master_index)
    if n == 0:
        return pd.Series(dtype=float), pd.DataFrame(columns=_TRADE_COLS)

    arr: dict[str, dict] = {}
    for t, df in prices.items():
        d = df.reindex(master_index)
        s = signals.get(t)
        if s is None:
            sig = [0] * n
        else:
            sig = (s.reindex(master_index).fillna(0).to_numpy() > 0).astype(int).tolist()
        arr[t] = {
            "open":   d["open"].to_numpy(dtype=float),
            "high":   d["high"].to_numpy(dtype=float),
            "low":    d["low"].to_numpy(dtype=float),
            "close":  d["close"].to_numpy(dtype=float),
            "volume": d["volume"].to_numpy(dtype=float),
            "sig":    sig,
        }
    tickers = list(arr.keys())

    positions: dict[str, dict] = {}          # ticker -> {shares, raw_entry, eff_entry, entry_idx, cost_basis, last_close}
    pending_entries: dict[str, float] = {}   # ticker -> signal-day volume (rank key)
    pending_exits: set[str] = set()          # signal-driven exits to run at next open
    blocked_until_flat: set[str] = set()     # blocked from re-entry until signal resets

    equity_vals: list[float] = []
    trades: list[dict] = []
    cash_floor_warned = False

    def _record(t, pos, exit_date, raw_exit, eff_sell, reason):
        ret     = (eff_sell / pos["eff_entry"]) - 1.0
        proceeds = pos["shares"] * eff_sell
        trades.append({
            "entry_date":  master_index[pos["entry_idx"]],
            "exit_date":   exit_date,
            "entry_price": pos["raw_entry"],
            "exit_price":  raw_exit,
            "return_pct":  ret,
            "direction":   "long",
            "ticker":      t,
            "exit_reason": reason,
            "pnl_abs":     proceeds - pos["cost_basis"],
        })
        return proceeds

    for p in range(n):
        dt = master_index[p]
        exited_today: set[str] = set()

        # ── 1. OPEN: signal-driven exits (free slots + return cash) ───────────
        still_pending: set[str] = set()
        for t in pending_exits:
            if t not in positions:
                continue                      # already closed (e.g. stopped out)
            op = arr[t]["open"][p]
            if _isnan(op):
                still_pending.add(t)          # no price today; retry next day
                continue
            pos = positions.pop(t)
            cash += _record(t, pos, dt, op, op * (1.0 - cost), "signal")
            exited_today.add(t)
        pending_exits = still_pending

        # ── 2. OPEN: signal-driven entries (rank by volume, fill free slots) ──
        free = max_positions - len(positions)
        if pending_entries and free > 0:
            cands = [
                (t, v) for t, v in pending_entries.items()
                if t not in positions and t not in exited_today
            ]
            # highest signal-day volume first (NaN volume sinks to the bottom)
            cands.sort(key=lambda kv: (-1.0 if _isnan(kv[1]) else kv[1]), reverse=True)
            filled = 0
            for t, _vol in cands:
                if filled >= free:
                    break
                op = arr[t]["open"][p]
                if _isnan(op) or op <= 0:
                    continue
                if cash + 1e-6 < position_size:    # fixed sizing needs a full slot
                    continue
                eff_buy = op * (1.0 + cost)
                positions[t] = {
                    "shares":     position_size / eff_buy,
                    "raw_entry":  op,
                    "eff_entry":  eff_buy,
                    "entry_idx":  p,
                    "cost_basis": position_size,
                    "last_close": op,
                }
                cash -= position_size
                filled += 1
        pending_entries = {}                  # extras are skipped entirely

        # ── 3. INTRADAY: stop-loss / take-profit (stop first, gap-aware fill) ─
        if sl_pct is not None or tp_pct is not None:
            for t in list(positions):
                if t in exited_today:
                    continue
                pos = positions[t]
                op = arr[t]["open"][p]; hi = arr[t]["high"][p]; lo = arr[t]["low"][p]
                if _isnan(hi) or _isnan(lo) or _isnan(op):
                    continue
                raw_entry    = pos["raw_entry"]
                stop_price   = raw_entry * (1.0 - sl_pct) if sl_pct is not None else None
                target_price = raw_entry * (1.0 + tp_pct) if tp_pct is not None else None

                raw_exit = None; reason = None
                if stop_price is not None and lo <= stop_price:
                    raw_exit = min(op, stop_price)        # gap through stop → fill at open
                    reason   = "stop_loss"
                elif target_price is not None and hi >= target_price:
                    raw_exit = max(op, target_price)      # gap through target → fill at open
                    reason   = "take_profit"

                if raw_exit is not None:
                    positions.pop(t)
                    cash += _record(t, pos, dt, raw_exit, raw_exit * (1.0 - cost), reason)
                    exited_today.add(t)
                    pending_exits.discard(t)
                    blocked_until_flat.add(t)             # wait for a fresh signal

        # ── 4. Cash guard (FIX 2) ─────────────────────────────────────────────
        if cash < 0:
            if cash < -1e-6 and not cash_floor_warned:
                print(f"  ⚠ [portfolio] cash went negative ({cash:.4f}) on "
                      f"{dt.date()} — flooring at 0 (rounding). Check cost logic.")
                cash_floor_warned = True
            cash = max(cash, 0.0)
        assert cash >= 0, f"Cash went negative on {dt}. Check cost calculation."

        # ── 5. MARK equity at close ───────────────────────────────────────────
        port_val = cash
        for t, pos in positions.items():
            cl = arr[t]["close"][p]
            if _isnan(cl):
                cl = pos["last_close"]         # carry last known close for valuation
            else:
                pos["last_close"] = cl
            port_val += pos["shares"] * cl
        equity_vals.append(port_val)

        # ── 6. SIGNAL: read today's signal, set pending action for next day ───
        for t in tickers:
            s = arr[t]["sig"][p]
            held = t in positions
            if s == 0:
                blocked_until_flat.discard(t)  # signal reset → fresh entries allowed
            if s == 1 and not held and t not in exited_today and t not in blocked_until_flat:
                pending_entries[t] = arr[t]["volume"][p]
            elif s != 1 and held:
                pending_exits.add(t)

    # ── Force-close any open positions at the last bar's close ────────────────
    if positions:
        last_p  = n - 1
        last_dt = master_index[last_p]
        liquidation = cash
        for t in list(positions):
            pos = positions.pop(t)
            cl = arr[t]["close"][last_p]
            if _isnan(cl):
                cl = pos["last_close"]
            eff_sell = cl * (1.0 - cost)
            liquidation += pos["shares"] * eff_sell
            _record(t, pos, last_dt, cl, eff_sell, "force_close")
        equity_vals[-1] = liquidation          # final mark reflects liquidation costs

    equity = pd.Series(equity_vals, index=master_index, name="equity")
    trades_df = pd.DataFrame(trades, columns=_TRADE_COLS) if trades else pd.DataFrame(columns=_TRADE_COLS)
    return equity, trades_df
