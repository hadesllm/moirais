"""Tests for moirais.fn.eqrid — racial disparity index."""

import pytest
from moirais.fn.eqrid import racial_disparity_index
from moirais.fn._containers import ESRes


class TestRdi:
    def test_basic(self):
        r = racial_disparity_index({"White": 2.0, "Black": 6.0, "Indigenous": 8.0})
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(4.0)

    def test_custom_reference(self):
        r = racial_disparity_index({"A": 5.0, "B": 10.0}, reference_group="A")
        assert r.extra["ratios"]["B"] == pytest.approx(2.0)

    def test_too_few(self):
        with pytest.raises(ValueError):
            racial_disparity_index({"A": 1.0})
