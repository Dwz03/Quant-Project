import pandas as pd
import pytest

from src.research.periods import (
    HOLDOUT_PERIOD,
    RESEARCH_PERIOD,
    VALIDATION_PERIOD,
    filter_panel_by_period,
    get_research_period,
)
from src.research.universes import (
    LARGE_LIQUID_US_EQUITIES_V1,
    LARGE_LIQUID_US_EQUITIES_V1_BY_SECTOR,
    UNIVERSE_METADATA,
)


def test_large_universe_is_fixed_unique_and_sector_diversified():
    assert len(LARGE_LIQUID_US_EQUITIES_V1) == 88
    assert len(set(LARGE_LIQUID_US_EQUITIES_V1)) == 88
    assert len(LARGE_LIQUID_US_EQUITIES_V1_BY_SECTOR) == 11
    assert {
        len(symbols)
        for symbols in LARGE_LIQUID_US_EQUITIES_V1_BY_SECTOR.values()
    } == {8}
    assert UNIVERSE_METADATA["large"]["number_of_symbols"] == 88
    assert UNIVERSE_METADATA["large"]["point_in_time"] is False
    assert UNIVERSE_METADATA["large"]["contains_survivorship_bias"] is True


def test_research_validation_and_holdout_periods_are_disjoint():
    assert RESEARCH_PERIOD.start == pd.Timestamp("2021-01-01")
    assert RESEARCH_PERIOD.end == pd.Timestamp("2023-12-31")
    assert VALIDATION_PERIOD.start == pd.Timestamp("2024-01-01")
    assert VALIDATION_PERIOD.end == pd.Timestamp("2025-12-31")
    assert HOLDOUT_PERIOD.start == pd.Timestamp("2026-01-01")
    assert HOLDOUT_PERIOD.end is None
    assert RESEARCH_PERIOD.end < VALIDATION_PERIOD.start
    assert VALIDATION_PERIOD.end < HOLDOUT_PERIOD.start


def test_period_filtering_uses_inclusive_boundaries_and_guards_holdout():
    dates = pd.to_datetime([
        "2020-12-31",
        "2021-01-01",
        "2023-12-31",
        "2024-01-01",
        "2025-12-31",
        "2026-01-01",
    ])
    index = pd.MultiIndex.from_product(
        [dates, ["AAA"]],
        names=["date", "symbol"],
    )
    panel = pd.DataFrame({"value": range(len(index))}, index=index)

    research = filter_panel_by_period(panel, "research")
    validation = filter_panel_by_period(panel, "validation")

    assert research.index.get_level_values("date").tolist() == list(dates[1:3])
    assert validation.index.get_level_values("date").tolist() == list(dates[3:5])
    with pytest.raises(ValueError, match="explicit"):
        get_research_period("holdout")

    holdout = filter_panel_by_period(
        panel,
        "holdout",
        allow_holdout=True,
    )
    assert holdout.index.get_level_values("date").tolist() == [dates[5]]
