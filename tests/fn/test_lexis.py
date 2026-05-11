"""Tests for morie.fn.lexis -- Lexis diagram data."""

import pytest
from morie.fn.lexis import lexis_diagram_data


class TestLexisDiagram:
    def test_basic(self):
        res = lexis_diagram_data(
            birth_dates=[1990, 1985, 2000],
            event_dates=[2020, 2020, 2020],
        )
        assert res.extra["mean_age_at_event"] == pytest.approx((30 + 35 + 20) / 3)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            lexis_diagram_data([1990], [2020, 2021])
