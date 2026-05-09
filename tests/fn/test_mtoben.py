"""Tests for moirais.fn.mtoben — benefit cost."""

import pytest
from moirais.fn.mtoben import mto_benefit_cost
from moirais.fn._containers import ESRes


class TestBenefitCost:
    def test_basic(self):
        r = mto_benefit_cost(10, 500000)
        assert isinstance(r, ESRes)
        assert r.estimate == pytest.approx(4.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            mto_benefit_cost(5, 0)
