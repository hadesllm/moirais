"""Tests for moirais.fn.evals — E-value."""

import pytest

from moirais.fn.evals import e_value


class TestEValue:
    def test_rr_2(self):
        res = e_value(2.0)
        assert res.estimate > 2.0

    def test_rr_1(self):
        res = e_value(1.0)
        assert res.estimate == pytest.approx(1.0)

    def test_with_ci(self):
        res = e_value(2.5, ci_lower=1.5)
        assert res.extra["e_value_ci"] is not None
        assert res.extra["e_value_ci"] < res.estimate

    def test_invalid(self):
        with pytest.raises(ValueError):
            e_value(-1.0)
