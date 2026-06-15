"""Data acquisition and snapshot loading.

By default the analysis loads a committed parquet snapshot so results are
deterministic and network-free. ``refresh=True`` re-pulls live from Yahoo Finance.
"""

from __future__ import annotations

import pandas as pd

from .config import load_config, resolve_path

OHLC = ["Open", "High", "Low", "Close"]


def download_fx(tickers: dict[str, str], start: str, end: str) -> pd.DataFrame:
    """Download daily OHLC for each pair from Yahoo Finance into a long DataFrame.

    Returns columns: Date, Open, High, Low, Close, pair. Only real trading days
    are returned (no calendar fabrication).
    """
    import yfinance as yf  # imported lazily; not needed for the offline path

    frames = []
    for name, ticker in tickers.items():
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            interval="1d",
            auto_adjust=False,
            multi_level_index=False,
        )
        df = df.reset_index()[["Date", *OHLC]].sort_values("Date")
        df["pair"] = name
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    long["Date"] = pd.to_datetime(long["Date"])
    return long


def load_ohlc(refresh: bool = False) -> pd.DataFrame:
    """Load the long-format OHLC table (Date, OHLC, pair).

    Loads the committed snapshot unless ``refresh=True``, in which case it
    re-downloads live and overwrites the snapshot.
    """
    cfg = load_config()
    snapshot = resolve_path(cfg["data"]["snapshot"])

    if refresh or not snapshot.exists():
        d = cfg["data"]
        long = download_fx(d["tickers"], d["start"], d["end"])
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        long.to_parquet(snapshot, index=False)
        return long

    long = pd.read_parquet(snapshot)
    long["Date"] = pd.to_datetime(long["Date"])
    return long


def to_wide_close(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long table to a wide close-price matrix (index=Date, cols=pair)."""
    wide = long.pivot(index="Date", columns="pair", values="Close").sort_index()
    wide.columns.name = None
    return wide


def to_wide_ohlc(long: pd.DataFrame) -> pd.DataFrame:
    """Pivot to a wide OHLC matrix with ``{pair}_{field}`` columns."""
    pairs = list(dict.fromkeys(long["pair"]))
    wide = pd.DataFrame(index=sorted(long["Date"].unique()))
    for name in pairs:
        sub = long[long["pair"] == name].set_index("Date").sort_index()
        for col in OHLC:
            wide[f"{name}_{col}"] = sub[col]
    wide.index.name = "Date"
    return wide
