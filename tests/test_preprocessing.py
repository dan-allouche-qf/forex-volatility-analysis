"""Data-integrity tests on the committed offline snapshot (no network)."""

from fxvol import config, data
from fxvol import preprocessing as pp


def test_snapshot_loads_offline():
    long = data.load_ohlc()
    assert set(long["pair"].unique()) == set(config.pairs())
    assert {"Open", "High", "Low", "Close"}.issubset(long.columns)


def test_alignment_has_no_weekend_or_nan():
    long = data.load_ohlc()
    close = pp.align_close(long, how="union")
    assert (close.index.dayofweek <= 4).all(), "weekend row leaked into the series"
    assert close.notna().all().all(), "NaNs left after alignment"
    assert close.index.is_monotonic_increasing


def test_no_fabricated_rows_vs_raw_trading_days():
    long = data.load_ohlc()
    raw_days = long["Date"].nunique()
    close = pp.align_close(long, how="union")
    # Union alignment must never create MORE rows than the real trading calendar.
    assert len(close) <= raw_days
    # Sanity: the 2022-2024 window has far fewer rows than calendar days (~1093).
    span = (close.index.max() - close.index.min()).days + 1
    assert len(close) < span * 0.8


def test_union_and_inner_agree_when_calendars_match():
    long = data.load_ohlc()
    u = pp.align_close(long, how="union")
    i = pp.align_close(long, how="inner")
    # The three majors share the same trading calendar in this sample.
    assert len(u) == len(i)


def test_invalid_how():
    long = data.load_ohlc()
    try:
        pp.align_close(long, how="bogus")
        raised = False
    except ValueError:
        raised = True
    assert raised
