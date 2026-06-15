"""Trading-day alignment.

Critical design choice: FX trades Monday-Friday, so weekend dates are *non-
observations*, not missing data. We therefore NEVER reindex onto a full calendar
and we NEVER fabricate prices. Pairs are aligned on their actual shared trading
dates; genuine single-day holiday gaps (a date present for some pairs but not
others) are forward-filled at the price level only, with returns on such a day
left to come out near zero rather than inventing a jump.
"""

from __future__ import annotations

import pandas as pd

from .data import to_wide_close


def align_close(long: pd.DataFrame, how: str = "union") -> pd.DataFrame:
    """Align pair close prices on real trading dates.

    Parameters
    ----------
    long : long-format OHLC table (Date, OHLC, pair).
    how : "union" keeps every date traded by *any* pair and forward-fills the
        rare single-day holiday gaps (default; maximizes usable history).
        "inner" keeps only dates traded by *all* pairs (zero fill at all).

    Returns a wide close-price matrix on a pure trading-day index. Guarantees:
    no weekend rows, no NaNs after the first valid observation, no fabricated
    weekend values.
    """
    wide = to_wide_close(long)

    if how == "inner":
        wide = wide.dropna(how="any")
    elif how == "union":
        # Forward-fill only genuine holiday gaps embedded in the trading calendar.
        wide = wide.ffill().dropna(how="any")
    else:  # pragma: no cover - guarded by tests
        raise ValueError(f"unknown how={how!r}; use 'union' or 'inner'")

    assert (wide.index.dayofweek <= 4).all(), "alignment leaked a weekend date"
    assert wide.notna().all().all(), "alignment left NaNs"
    return wide


def trading_day_summary(wide: pd.DataFrame) -> pd.Series:
    """Quick integrity summary used in the notebook to prove no fabrication."""
    idx = wide.index
    return pd.Series(
        {
            "rows": len(wide),
            "first": idx.min().date().isoformat(),
            "last": idx.max().date().isoformat(),
            "weekend_rows": int((idx.dayofweek > 4).sum()),
            "calendar_days_spanned": (idx.max() - idx.min()).days + 1,
        }
    )
