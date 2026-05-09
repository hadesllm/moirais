"""Tests for moirais.fn.lftbl -- Complete life table."""

import pytest
from moirais.fn.lftbl import life_table_complete


class TestLifeTableComplete:
    def test_basic(self):
        ages = list(range(5))
        deaths = [10, 5, 3, 8, 20]
        pops = [1000, 900, 850, 800, 500]
        res = life_table_complete(ages, deaths, pops)
        assert res.measure == "life_table_complete"
        assert res.estimate > 0

    def test_qx_last_is_one(self):
        ages = list(range(3))
        deaths = [10, 5, 50]
        pops = [1000, 900, 500]
        res = life_table_complete(ages, deaths, pops)
        df = res.extra["table"]
        assert df["qx"].iloc[-1] == pytest.approx(1.0)

    def test_mismatch(self):
        with pytest.raises(ValueError):
            life_table_complete([0, 1], [10], [1000, 900])
