"""Buy and Hold strategy — FALSIFY reference implementation."""

from __future__ import annotations

import pandas as pd

from strategy.base import BaseStrategy


class BuyHoldStrategy(BaseStrategy):

    def indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        return data  # no indicators needed

    def buy_signal(self, data: pd.DataFrame) -> pd.Series:
        # Always buy on first bar, never again
        signal = pd.Series(False, index=data.index)
        signal.iloc[0] = True
        return signal

    def sell_signal(self, data: pd.DataFrame) -> pd.Series:
        # Never sell
        return pd.Series(False, index=data.index)

    @staticmethod
    def param_grid() -> dict:
        return {}
