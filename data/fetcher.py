"""Download and clean prices from Yahoo; save under DATA/."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from engine.types import PricePanel

_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CACHE = _ROOT / "data" / "cache"

_ADJ_NOTE_PRINTED = False


def _adj_note() -> None:
    """Print the dividend-adjustment note once per process."""
    global _ADJ_NOTE_PRINTED
    if not _ADJ_NOTE_PRINTED:
        print("Using dividend-adjusted prices.")
        _ADJ_NOTE_PRINTED = True


@dataclass
class QualityReport:
    warnings: list[str] = field(default_factory=list)
    tickers: list[str] = field(default_factory=list)
    start: str = ""
    end: str = ""
    rows: int = 0

    def ok(self) -> bool:
        return len(self.warnings) == 0


def load_prices(
    tickers: list[str] | tuple[str, ...],
    start: str,
    end: str,
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
    align: str = "inner",
) -> tuple[PricePanel, QualityReport]:
    _adj_note()
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    tickers = [t.upper() for t in tickers]
    report = QualityReport(tickers=tickers, start=start, end=end)

    panels: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        path = cache / f"{ticker}_{start}_{end}.parquet"
        if path.exists() and not force_refresh:
            raw = pd.read_parquet(path)
        else:
            raw = _fetch_yahoo(ticker, start, end)
            raw.to_parquet(path)
        panels[ticker] = raw

    panel, align_warnings = _build_panel(panels, align=align)
    report.warnings.extend(align_warnings)
    report.warnings.extend(_qa_panel(panel))
    report.rows = len(panel.dates)
    return panel, report


def load_universe_prices(
    tickers: list[str] | tuple[str, ...],
    start: str,
    end: str,
    min_bars: int = 250,
    cache_dir: Path | str | None = None,
    force_refresh: bool = False,
) -> tuple[PricePanel, QualityReport]:
    """
    Fetch OHLCV for a whole universe (portfolio mode).

    Resilient per ticker: a fetch error or a ticker with fewer than `min_bars`
    valid rows is SKIPPED (not fatal). Each ticker is cached individually.
    Progress is printed every 20 tickers. The panel is OUTER-aligned: the
    index is the union of all trading days, with NaN where a ticker has no data
    (different listing dates, suspensions, etc.).
    """
    _adj_note()
    cache = Path(cache_dir) if cache_dir else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    tickers = [t.upper() for t in tickers]
    report = QualityReport(tickers=tickers, start=start, end=end)

    per_ticker: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    total = len(tickers)
    print(f"Fetching {total} tickers  {start} → {end} ...")

    for i, ticker in enumerate(tickers, start=1):
        path = cache / f"{ticker}_{start}_{end}.parquet"
        try:
            if path.exists() and not force_refresh:
                raw = pd.read_parquet(path)
            else:
                raw = _fetch_yahoo(ticker, start, end)
                raw.to_parquet(path)
            if raw is None or len(raw.dropna(subset=["adj_close"])) < min_bars:
                skipped.append(ticker)
            else:
                per_ticker[ticker] = raw
        except Exception as e:
            skipped.append(ticker)
            report.warnings.append(f"{ticker}: skipped ({e})")

        if i % 20 == 0 or i == total:
            print(f"  Progress: {i}/{total} fetched  "
                  f"({len(per_ticker)} kept, {len(skipped)} skipped)")

    if not per_ticker:
        raise ValueError(
            "No tickers had sufficient data — cannot build the universe panel."
        )

    panel, align_warnings = _build_panel(per_ticker, align="outer")
    report.warnings.extend(align_warnings)
    report.rows = len(panel.dates)
    report.tickers = list(per_ticker.keys())
    if skipped:
        report.warnings.append(
            f"Skipped {len(skipped)} tickers (insufficient data or fetch error)."
        )
        preview = ", ".join(skipped[:10]) + (" ..." if len(skipped) > 10 else "")
        print(f"  Skipped {len(skipped)} tickers: {preview}")
    print(f"Universe panel: {len(per_ticker)} tickers × {len(panel.dates)} dates.")
    return panel, report


def _fetch_yahoo(ticker: str, start: str, end: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("Install: pip install -e '.[full]'") from e

    # auto_adjust=True → OHLC are already dividend/split adjusted; yfinance
    # then usually returns NO separate 'Adj Close' column.
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data for {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "Adj Close": "adj_close",
    })
    df.index = pd.to_datetime(df.index).tz_localize(None)

    # With auto_adjust=True 'close' is already adjusted. If an explicit
    # 'Adj Close' was returned, use it; otherwise 'close' IS the adjusted price.
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    return df[["open", "high", "low", "close", "volume", "adj_close"]].sort_index()


def _build_panel(per_ticker: dict[str, pd.DataFrame], align: str = "inner") -> tuple[PricePanel, list[str]]:
    warnings: list[str] = []
    fields = ["open", "high", "low", "close", "volume", "adj_close"]
    wide: dict[str, pd.DataFrame] = {}
    for field in fields:
        wide[field] = pd.concat(
            [df[field].rename(t) for t, df in per_ticker.items()], axis=1
        ).sort_index()

    if align == "inner":
        mask = wide["adj_close"].notna().all(axis=1)
        dropped = (~mask).sum()
        if dropped:
            warnings.append(f"Dropped {dropped} days where not every stock had a price.")
        for f in fields:
            wide[f] = wide[f].loc[mask]

    panel = PricePanel(**wide)
    panel.validate()
    return panel, warnings


def _qa_panel(panel: PricePanel) -> list[str]:
    warnings: list[str] = []
    if panel.adj_close.empty:
        return ["Price table is empty."]
    for ticker in panel.tickers:
        s = panel.adj_close[ticker].dropna()
        if len(s) < 20:
            warnings.append(f"{ticker}: only {len(s)} days of data.")
        ret = s.pct_change().dropna()
        if len(ret) and ret.abs().max() > 0.5:
            warnings.append(f"{ticker}: one-day move over 50% — check for bad data.")
    return warnings
