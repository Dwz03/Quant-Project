"""Explicit, disjoint periods for feature-alpha research."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchPeriod:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None


RESEARCH_PERIOD = ResearchPeriod(
    name="research",
    start=pd.Timestamp("2021-01-01"),
    end=pd.Timestamp("2023-12-31"),
)
VALIDATION_PERIOD = ResearchPeriod(
    name="validation",
    start=pd.Timestamp("2024-01-01"),
    end=pd.Timestamp("2025-12-31"),
)
HOLDOUT_PERIOD = ResearchPeriod(
    name="holdout",
    start=pd.Timestamp("2026-01-01"),
    end=None,
)

RESEARCH_PERIODS = {
    period.name: period
    for period in (RESEARCH_PERIOD, VALIDATION_PERIOD, HOLDOUT_PERIOD)
}


def get_research_period(
    name: str,
    allow_holdout: bool = False,
) -> ResearchPeriod:
    try:
        period = RESEARCH_PERIODS[name]
    except KeyError as exc:
        valid_names = ", ".join(RESEARCH_PERIODS)
        raise ValueError(f"period must be one of: {valid_names}") from exc

    if name == "holdout" and not allow_holdout:
        raise ValueError("holdout access requires explicit allow_holdout=True")

    return period


def filter_panel_by_period(
    panel: pd.DataFrame,
    period_name: str,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Filter a (date, symbol) panel using inclusive feature-date boundaries."""
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("panel must use a (date, symbol) MultiIndex")
    if panel.index.names != ["date", "symbol"]:
        raise ValueError("panel index levels must be named date and symbol")

    period = get_research_period(period_name, allow_holdout=allow_holdout)
    dates = panel.index.get_level_values("date")
    mask = dates >= period.start
    if period.end is not None:
        mask &= dates <= period.end

    return panel.loc[mask].copy()
