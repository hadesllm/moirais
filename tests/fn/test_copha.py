"""Tests for moirais.fn.copha -- Cophenetic correlation."""

import numpy as np
from moirais.fn.copha import cophenetic_correlation, copha
from moirais.fn._containers import DescriptiveResult


class TestCopheneticCorrelation:
    def test_alias(self):
        assert copha is cophenetic_correlation

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        res = cophenetic_correlation(X)
        assert isinstance(res, DescriptiveResult)

    def test_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        res = cophenetic_correlation(X)
        assert -1.0 <= res.value <= 1.0

    def test_high_for_well_structured(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (10, 2)), rng.normal(5, 0.5, (10, 2))])
        res = cophenetic_correlation(X, method="average")
        assert res.value > 0.5
