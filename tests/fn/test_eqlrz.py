"""Tests for moirais.fn.eqlrz — Lorenz curve."""

import pytest
from moirais.fn.eqlrz import lorenz_curve
from moirais.fn._containers import DescriptiveResult


class TestLorenz:
    def test_basic(self):
        r = lorenz_curve([1, 2, 3, 4, 5])
        assert isinstance(r, DescriptiveResult)
        assert r.extra["cumulative_share"][-1] == pytest.approx(1.0)
        assert r.extra["proportions"][0] == 0.0

    def test_too_few(self):
        with pytest.raises(ValueError):
            lorenz_curve([1])
