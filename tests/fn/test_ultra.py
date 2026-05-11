"""Tests for morie.fn.ultra -- ensemble aggregation."""

import numpy as np
from morie.fn.ultra import ensemble_aggregate, ultra
from morie.fn._containers import DescriptiveResult


class TestUltra:
    def test_alias(self):
        assert ultra is ensemble_aggregate

    def test_mean(self):
        preds = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        r = ensemble_aggregate(preds, method="mean")
        assert isinstance(r, DescriptiveResult)
        np.testing.assert_allclose(r.value, [2.0, 2.0, 2.0])

    def test_weighted(self):
        preds = np.array([[0.0, 0.0], [10.0, 10.0]])
        r = ensemble_aggregate(preds, weights=np.array([0.25, 0.75]))
        np.testing.assert_allclose(r.value, [7.5, 7.5])
