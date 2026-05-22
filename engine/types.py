"""Main data shapes used across the project."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class PricePanel:
    """Price table: rows = dates, columns = tickers."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    adj_close: pd.DataFrame

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.adj_close.index

    @property
    def tickers(self) -> list[str]:
        return list(self.adj_close.columns)

    def validate(self) -> None:
        frames = [self.open, self.high, self.low, self.close, self.volume, self.adj_close]
        idx = self.adj_close.index
        cols = self.adj_close.columns
        for f in frames:
            if not f.index.equals(idx) or not f.columns.equals(cols):
                raise ValueError("Price table columns/dates are not aligned")


@dataclass
class MarketSnapshot:
    """What the strategy is allowed to see on a given day (no future)."""

    as_of: pd.Timestamp
    prices: pd.DataFrame
    opens: pd.DataFrame | None = None

    @property
    def history(self) -> pd.DataFrame:
        return self.prices.loc[: self.as_of]

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, float]

    def copy(self) -> PortfolioState:
        return PortfolioState(cash=self.cash, positions=dict(self.positions))
