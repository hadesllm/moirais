"""Tests for morie.fn.ismrt -- Indirect standardization."""

import pytest
from morie.fn.ismrt import indirect_standardization


class TestIndirectStandardization:
    def test_known(self):
        res = indirect_standardization(
            observed=50,
            populations=[1000, 2000],
            reference_rates=[0.01, 0.02],
        )
        assert res.measure == "indirect_std_rate"
        assert res.extra["SMR"] > 0

    def test_smr_unity(self):
        res = indirect_standardization(
            observed=50,
            populations=[1000, 2000],
            reference_rates=[0.01, 0.02],
        )
        expected = 1000 * 0.01 + 2000 * 0.02
        assert res.extra["SMR"] == pytest.approx(50 / expected)

    def test_invalid(self):
        with pytest.raises(ValueError):
            indirect_standardization(observed=-1, populations=[100], reference_rates=[0.01])
