"""Tests for moirais.fn.psafs -- Population attributable fraction."""

import pytest
from moirais.fn.psafs import population_attributable_fraction


class TestPAF:
    def test_known(self):
        res = population_attributable_fraction(rr=2.0, prevalence=0.3)
        expected = 0.3 * 1 / (1 + 0.3 * 1)
        assert res.estimate == pytest.approx(expected)

    def test_no_risk(self):
        res = population_attributable_fraction(rr=1.0, prevalence=0.5)
        assert res.estimate == pytest.approx(0.0)

    def test_invalid(self):
        with pytest.raises(ValueError):
            population_attributable_fraction(rr=-1, prevalence=0.5)
