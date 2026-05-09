"""Tests for moirais.fn.suprv -- substance use prevalence."""

import pytest
from moirais.fn.suprv import substance_prevalence


class TestSubstancePrevalence:
    def test_known(self):
        res = substance_prevalence(n_users=30, n_surveyed=200)
        assert res.measure == "substance_prevalence"
        assert res.estimate == pytest.approx(0.15)

    def test_ci_brackets(self):
        res = substance_prevalence(n_users=30, n_surveyed=200)
        assert res.ci_lower < res.estimate < res.ci_upper

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            substance_prevalence(n_users=0, n_surveyed=0)
