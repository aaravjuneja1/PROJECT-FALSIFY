"""
20/50 moving-average crossover strategy for FALSIFY.

Buy  : fast MA crosses ABOVE slow MA (golden cross)
Sell : fast MA crosses BELOW slow MA (death cross)

The .shift(1) calls below detect the CROSSOVER EVENT (comparing today's MA
relationship to yesterday's). They are computed from data available at the
close of bar T and do NOT delay the signal — FALSIFY's backtester applies
the T+1 entry delay automatically.

Stop loss and take profit are FIXED risk-management rules (5% / 10%), not
optimised parameters. See BaseStrategy for why SL/TP must never go in
param_grid (optimising them is curve fitting).
"""
from __future__ import annotations

import pandas as pd

from strategy.base import BaseStrategy


class MACrossoverStrategy(BaseStrategy):

    def indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        fast = getattr(self, 'fast_ma', 20)
        slow = getattr(self, 'slow_ma', 50)
        data['fast_ma'] = data['close'].rolling(fast).mean()
        data['slow_ma'] = data['close'].rolling(slow).mean()
        return data

    def buy_signal(self, data: pd.DataFrame) -> pd.Series:
        return (data['fast_ma'] > data['slow_ma']) & \
               (data['fast_ma'].shift(1) <= data['slow_ma'].shift(1))

    def sell_signal(self, data: pd.DataFrame) -> pd.Series:
        return (data['fast_ma'] < data['slow_ma']) & \
               (data['fast_ma'].shift(1) >= data['slow_ma'].shift(1))

    def stop_loss_pct(self) -> float:
        return 0.05   # 5% stop loss — fixed risk rule, NOT optimised

    def take_profit_pct(self) -> float:
        return 0.10   # 10% take profit — fixed risk rule, NOT optimised

    @staticmethod
    def param_grid() -> dict:
        return {
            'fast_ma': [10, 20, 30],
            'slow_ma': [50, 100, 150]
        }
