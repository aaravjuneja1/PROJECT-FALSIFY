from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):
    """
    FALSIFY Base Strategy Interface
    ════════════════════════════════

    To create a strategy for FALSIFY, inherit from
    this class and implement all four methods below.

    IMPORTANT — LOOK-AHEAD RULE:
    Do NOT call .shift(1) on your signals.
    FALSIFY enforces T+1 entry automatically: the backtester executes
    your signal at the OPEN of the NEXT bar (a signal computed from the
    close of day T fills at the open of day T+1). Compute indicators on
    raw data and return signals directly — the engine handles the delay.
    (Using .shift(1) on an INDICATOR to detect a crossover — comparing
    today's MA to yesterday's — is fine; that is not shifting the signal.)

    STOP LOSS / TAKE PROFIT:
    Optionally override stop_loss_pct() and take_profit_pct() to add risk
    management. Each returns a decimal fraction (0.05 = 5%) or None. The
    portfolio backtester reads them after params are applied and exits a
    position intraday when price pierces the level (stop checked first):

        class MyStrategy(BaseStrategy):
            def stop_loss_pct(self)   -> float: return 0.05   # 5% stop
            def take_profit_pct(self) -> float: return 0.10   # 10% target

    Defaults return None → exit on signal only (no SL/TP).

    ⚠ DO NOT add stop loss or take profit to param_grid. These are risk
    management decisions that should be fixed based on your trading logic,
    not optimised by WFA. Optimising SL/TP alongside strategy parameters is
    curve fitting — WFA will find values that worked historically but have
    no forward-looking validity. Define SL/TP as fixed values using the
    stop_loss_pct() and take_profit_pct() methods. Change them only when
    your risk management rules change, not when backtest results change.

    EXAMPLE — 200-day Moving Average strategy:

        class MA200Strategy(BaseStrategy):

            def indicators(self, data):
                data['ma200'] = data['close'].rolling(200).mean()
                return data

            def buy_signal(self, data):
                return data['close'] > data['ma200']

            def sell_signal(self, data):
                return data['close'] < data['ma200']

            @staticmethod
            def param_grid():
                return {'period': [100, 150, 200, 250]}

    SIGNAL RULES:
    - buy_signal returns pd.Series of bool (True = buy)
    - sell_signal returns pd.Series of bool (True = sell)
    - Both must have same index as data
    - NaN values are treated as no signal

    PARAM GRID RULES:
    - Return {} if strategy has no tunable parameters
    - Keys are parameter names passed to generate_signals
    - Values are lists of values to test in WFA grid search
    - Maximum 3 parameters recommended (more = overfitting risk)
    """

    @abstractmethod
    def indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators needed.
        Add them as new columns to data and return it.

        Args:
            data: OHLCV DataFrame with columns:
                  open, high, low, close, volume
        Returns:
            Same DataFrame with indicator columns added
        """
        pass

    @abstractmethod
    def buy_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Define when to enter a long position.

        Args:
            data: OHLCV DataFrame with indicator columns
                  already computed by indicators()
        Returns:
            pd.Series of bool, same index as data
            True = buy signal on this bar
        """
        pass

    @abstractmethod
    def sell_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        Define when to exit a long position.

        Args:
            data: OHLCV DataFrame with indicator columns
                  already computed by indicators()
        Returns:
            pd.Series of bool, same index as data
            True = sell signal on this bar
        """
        pass

    @staticmethod
    @abstractmethod
    def param_grid() -> dict:
        """
        Define parameters to optimise in WFA grid search.

        Returns:
            dict of param_name -> list of values
            Example: {'fast': [10,20,30], 'slow': [50,100,200]}
            Return {} if strategy has no parameters.
        """
        pass

    def generate_signals(self,
                         data: pd.DataFrame,
                         params: dict) -> pd.Series:
        """
        DO NOT OVERRIDE THIS METHOD.

        Called by FALSIFY internally. Applies params,
        computes indicators, generates buy/sell signals,
        and returns a combined signal series.

        Signal values:
            1  = long (buy or hold)
            0  = flat (out of market)

        Return RAW signals — do NOT shift. FALSIFY enforces T+1 entry via the
        backtester's pending-order mechanism: a signal at the close of bar T is
        executed at the OPEN of bar T+1. There is no .shift() anywhere in the
        engine; adding one would push entry to T+2.
        """
        # Apply params to self
        for key, value in params.items():
            setattr(self, key, value)

        # Compute indicators
        data = data.copy()
        data = self.indicators(data)

        # Generate raw signals
        buys  = self.buy_signal(data).fillna(False)
        sells = self.sell_signal(data).fillna(False)

        # Convert to position series (1=long, 0=flat)
        # Enter on buy, exit on sell, hold in between
        position = pd.Series(0, index=data.index, dtype=int)
        in_position = False

        for i in range(len(data)):
            if not in_position and buys.iloc[i]:
                in_position = True
            elif in_position and sells.iloc[i]:
                in_position = False
            position.iloc[i] = 1 if in_position else 0

        return position

    # ──────────────────────────────────────────────────────────────────
    # Optional risk management — override to enable. Default: disabled.
    # ──────────────────────────────────────────────────────────────────
    def stop_loss_pct(self) -> float | None:
        """
        Fractional stop-loss below entry price (0.05 = 5%), or None to disable.
        Read by the portfolio backtester AFTER params are applied.

        Do NOT add this to param_grid — a stop loss is a fixed risk-management
        rule, not a parameter to optimise. Optimising it is curve fitting.
        """
        return None

    def take_profit_pct(self) -> float | None:
        """
        Fractional take-profit above entry price (0.10 = 10%), or None to disable.
        Read by the portfolio backtester AFTER params are applied.

        Do NOT add this to param_grid — a take profit is a fixed risk-management
        rule, not a parameter to optimise. Optimising it is curve fitting.
        """
        return None

    # ──────────────────────────────────────────────────────────────────
    # Portfolio mode — generate signals across a universe of tickers.
    # ──────────────────────────────────────────────────────────────────
    def generate_signals_universe(
        self,
        prices: dict[str, pd.DataFrame],
        params: dict,
    ) -> dict[str, pd.Series]:
        """
        Generate signals for every ticker in a universe.

        Default behaviour: run generate_signals() on each ticker
        independently. Override ONLY for cross-stock logic (e.g. ranking
        by relative strength across the universe).

        Args:
            prices : {ticker: OHLCV DataFrame}
            params : strategy parameters (applied per ticker)
        Returns:
            {ticker: signal Series (1=long, 0=flat)} aligned to each
            ticker's own index.
        """
        out: dict[str, pd.Series] = {}
        for ticker, df in prices.items():
            out[ticker] = self.generate_signals(df, params)
        return out
