"""Plotting helpers. Pure presentation; all numbers come from the other modules."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from .config import resolve_path


def apply_style() -> None:
    """Consistent, dependency-free plot styling (no external style sheets)."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "font.size": 9,
        }
    )


def savefig(fig, name: str) -> Path:
    """Save a figure into the configured ``figures/`` directory."""
    out = resolve_path("figures")
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, bbox_inches="tight")
    return path


def dashboard(pair: str, close: pd.Series, sma_fast: pd.Series, sma_slow: pd.Series,
              vol_short: pd.Series, vol_long: pd.Series, returns_pct: pd.Series,
              rsi: pd.Series):
    """2x2 per-pair dashboard: price+SMA, rolling vol, return histogram, RSI."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    ax[0, 0].plot(close.index, close, label="Close", lw=1.4)
    ax[0, 0].plot(sma_fast.index, sma_fast, label="SMA fast", lw=1)
    ax[0, 0].plot(sma_slow.index, sma_slow, label="SMA slow", lw=1)
    ax[0, 0].set_title(f"{pair} - Price with SMAs")
    ax[0, 0].legend()

    ax[0, 1].plot(vol_short.index, vol_short * 100, label="short window", lw=1.4)
    ax[0, 1].plot(vol_long.index, vol_long * 100, label="long window", lw=1.4)
    ax[0, 1].set_title(f"{pair} - Annualized rolling volatility (%)")
    ax[0, 1].legend()

    ax[1, 0].hist(returns_pct.dropna(), bins=50, alpha=0.75, edgecolor="black")
    ax[1, 0].set_title(f"{pair} - Daily returns distribution (%)")
    ax[1, 0].set_xlabel("Daily return (%)")

    ax[1, 1].plot(rsi.index, rsi, lw=1.3)
    ax[1, 1].axhline(70, color="r", ls="--", alpha=0.5)
    ax[1, 1].axhline(30, color="g", ls="--", alpha=0.5)
    ax[1, 1].set_ylim(0, 100)
    ax[1, 1].set_title(f"{pair} - RSI")

    fig.tight_layout()
    return fig


def correlation_heatmap(corr: pd.DataFrame, title: str = "Return correlation"):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def rolling_correlation(roll_corr: pd.DataFrame, title: str = "Rolling correlation"):
    fig, ax = plt.subplots(figsize=(11, 4))
    for col in roll_corr.columns:
        ax.plot(roll_corr.index, roll_corr[col], label=col, lw=1.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-1, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def conditional_vol_overlay(pair: str, garch_vol: pd.Series, realized_vol: pd.Series,
                            ewma_vol: pd.Series | None = None):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(realized_vol.index, realized_vol * 100, label="realized (rolling)", lw=1, alpha=0.7)
    if ewma_vol is not None:
        ax.plot(ewma_vol.index, ewma_vol * 100, label="EWMA", lw=1, alpha=0.8)
    ax.plot(garch_vol.index, garch_vol * 100, label="GARCH(1,1)-t", lw=1.4)
    ax.set_title(f"{pair} - Conditional volatility (annualized %)")
    ax.legend()
    fig.tight_layout()
    return fig


def qq_plot(pair: str, returns: pd.Series):
    """QQ plot of standardized returns vs Normal and Student-t."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    r = returns.dropna()
    stats.probplot(r, dist="norm", plot=ax[0])
    ax[0].set_title(f"{pair} - QQ vs Normal")
    df, loc, scale = stats.t.fit(r)
    stats.probplot(r, dist=stats.t, sparams=(df,), plot=ax[1])
    ax[1].set_title(f"{pair} - QQ vs Student-t (df={df:.1f})")
    fig.tight_layout()
    return fig


def equity_curve(result, pair: str, label: str = "SMA crossover"):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(result.equity.index, result.equity, label=label, lw=1.5)
    ax.plot(result.benchmark.index, result.benchmark, label="Buy & hold", lw=1.2, alpha=0.8)
    ax.set_title(f"{pair} - Strategy vs buy-and-hold (growth of 1)")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    return fig


def annotate_drawdown(returns: pd.Series):  # small helper used in notebook narration
    from .risk import drawdown_curve

    dd = drawdown_curve(returns)
    fig, ax = plt.subplots(figsize=(11, 3))
    ax.fill_between(dd.index, dd * 100, 0, color="crimson", alpha=0.4)
    ax.set_title("Drawdown (%)")
    fig.tight_layout()
    return fig
