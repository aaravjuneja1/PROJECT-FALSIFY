"""
Nifty 200 universe resolver for FALSIFY portfolio mode.

Resolution order:
  0. Explicit list in config (data.universe_tickers) — for testing / custom sets
  1. Fresh local cache (< universe_cache_days old)
  2. NSE official constituent CSV (live)
  3. Wikipedia constituent table (fallback)
  4. Stale local cache (any age) — better than nothing
  5. Raise — portfolio mode cannot run without a universe

Tickers are returned with the '.NS' suffix for yfinance (NSE listings).
"""
from __future__ import annotations

import io
import json
import datetime as _dt
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_CACHE_FILE = _ROOT / "data" / "cache" / "nifty200_universe.json"

_NSE_URL  = "https://nsearchives.nseindia.com/content/indices/ind_nifty200list.csv"
_WIKI_URL = "https://en.wikipedia.org/wiki/NIFTY_200"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,application/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def get_nifty200_universe(config: dict) -> list[str]:
    """Resolve the Nifty 200 ticker universe (see module docstring for order)."""
    data_cfg   = config.get("data", {})
    cache_days = int(data_cfg.get("universe_cache_days", 7))

    # 0. Explicit list (testing / custom universe)
    explicit = data_cfg.get("universe_tickers")
    if explicit:
        syms = _normalise(explicit)
        print(f"Universe: {len(syms)} tickers from config (explicit list).")
        return syms

    # 1. Fresh cache
    cached = _read_cache()
    if cached is not None and cached["tickers"] and cached["age_days"] <= cache_days:
        print(f"Universe: {len(cached['tickers'])} tickers from cache "
              f"(source={cached['source']}, age={cached['age_days']:.1f}d).")
        return cached["tickers"]

    # 2. NSE live
    tickers = _try_nse()
    if tickers:
        _write_cache(tickers, "nse")
        print(f"Universe: {len(tickers)} tickers from NSE (live).")
        return tickers

    # 3. Wikipedia
    tickers = _try_wikipedia()
    if tickers:
        _write_cache(tickers, "wikipedia")
        print(f"Universe: {len(tickers)} tickers from Wikipedia (fallback).")
        return tickers

    # 4. Stale cache
    if cached is not None and cached["tickers"]:
        print(f"⚠ Universe: NSE + Wikipedia failed. Using STALE cache "
              f"({len(cached['tickers'])} tickers, age={cached['age_days']:.1f}d, "
              f"source={cached['source']}).")
        return cached["tickers"]

    # 5. Fail loudly
    raise RuntimeError(
        "Could not fetch the Nifty 200 universe — NSE, Wikipedia, and the local "
        "cache all failed. Portfolio mode cannot run. Connect to the internet, "
        "or set data.universe_tickers in config.yml, or place a valid list at "
        f"{_CACHE_FILE}."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Sources
# ──────────────────────────────────────────────────────────────────────────────

def _try_nse() -> list[str] | None:
    try:
        import requests
        sess = requests.Session()
        # NSE blocks bare requests; prime cookies from the homepage first.
        try:
            sess.get("https://www.nseindia.com", headers=_HEADERS, timeout=10)
        except Exception:
            pass
        resp = sess.get(_NSE_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        return _extract_symbols(df) or None
    except Exception as e:
        print(f"  [universe] NSE fetch failed: {e}")
        return None


def _try_wikipedia() -> list[str] | None:
    try:
        tables = pd.read_html(_WIKI_URL)
        for df in tables:
            syms = _extract_symbols(df)
            if syms and len(syms) >= 100:
                return syms
        return None
    except Exception as e:
        print(f"  [universe] Wikipedia fetch failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_symbols(df: pd.DataFrame) -> list[str]:
    """Find a symbol-like column and normalise to '.NS' tickers."""
    candidates = [
        c for c in df.columns
        if str(c).strip().lower() in ("symbol", "ticker", "nse code", "nse symbol")
    ]
    if not candidates:
        return []
    return _normalise(df[candidates[0]].dropna().tolist())


def _normalise(raw_list) -> list[str]:
    """Uppercase, strip, append '.NS', drop blanks, de-duplicate (order-preserving)."""
    seen: set[str] = set()
    out: list[str] = []
    for s in raw_list:
        s = str(s).strip().upper().replace(" ", "")
        if not s or s == "NAN":
            continue
        if not s.endswith(".NS"):
            s = s + ".NS"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _read_cache() -> dict | None:
    if not _CACHE_FILE.exists():
        return None
    try:
        data = json.loads(_CACHE_FILE.read_text())
        fetched = _dt.datetime.fromisoformat(data["fetched"])
        age_days = (_dt.datetime.now() - fetched).total_seconds() / 86400.0
        return {
            "tickers":  list(data.get("tickers", [])),
            "source":   data.get("source", "?"),
            "age_days": age_days,
        }
    except Exception:
        return None


def _write_cache(tickers: list[str], source: str) -> None:
    try:
        _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps({
            "fetched": _dt.datetime.now().isoformat(),
            "source":  source,
            "tickers": tickers,
        }, indent=2))
    except Exception as e:
        print(f"  [universe] could not write cache: {e}")
