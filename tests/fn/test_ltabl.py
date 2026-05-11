"""Tests for morie.fn.ltabl -- life table."""

import pytest
from morie.fn.ltabl import life_table_full


class TestLifeTable:
    def test_basic(self):
        res = life_table_full(
            age_groups=[0, 5, 10, 15],
            deaths=[50, 10, 5, 200],
            populations=[10000, 9500, 9000, 8000],
            interval=5,
        )
        assert res.name == "life_table"
        assert res.extra["e0"] > 0

    def test_short_raises(self):
        with pytest.raises(ValueError):
            life_table_full([0], [10], [1000])
