"""Tests for morie.fn.mxent — maximum entropy distribution."""

import numpy as np
import pytest

from morie.fn.mxent import mxent


class TestMxent:
    def test_no_constraints_uniform(self):
        support = np.arange(5, dtype=float)
        result = mxent(support, [])
        expected = np.ones(5) / 5
        np.testing.assert_allclose(result["pmf"], expected, atol=1e-4)

    def test_mean_constraint(self):
        support = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = mxent(support, [(lambda x: x, 3.0)])
        actual_mean = np.sum(result["pmf"] * support)
        assert actual_mean == pytest.approx(3.0, abs=0.05)

    def test_pmf_sums_to_one(self):
        support = np.arange(4, dtype=float)
        result = mxent(support, [(lambda x: x, 1.5)])
        assert np.sum(result["pmf"]) == pytest.approx(1.0, abs=1e-8)

    def test_entropy_positive(self):
        support = np.arange(3, dtype=float)
        result = mxent(support, [])
        assert result["entropy"] > 0

    def test_constraint_errors_small(self):
        support = np.array([0.0, 1.0, 2.0])
        result = mxent(support, [(lambda x: x, 1.0)])
        for err in result["constraint_errors"]:
            assert abs(err) < 0.1

    def test_empty_support_error(self):
        with pytest.raises(ValueError):
            mxent(np.array([]), [])
