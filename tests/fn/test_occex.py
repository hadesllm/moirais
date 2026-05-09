"""Tests for moirais.fn.occex -- occupational exposure."""

import pytest
from moirais.fn.occex import occupational_exposure


class TestOccupationalExposure:
    def test_under_oel(self):
        res = occupational_exposure([5, 8, 6, 7], oel=25)
        assert res.extra["category"] == "low"

    def test_overexposed(self):
        res = occupational_exposure([30, 35, 28], oel=25)
        assert res.extra["category"] == "overexposed"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            occupational_exposure([], oel=10)
