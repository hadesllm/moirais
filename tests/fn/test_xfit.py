"""Test cross_fit."""
import numpy as np
from moirais.fn.xfit import cross_fit, xfit
from moirais.fn._containers import DescriptiveResult


class TestCrossFit:
    def test_basic(self):
        X = np.random.default_rng(42).standard_normal((100, 5))
        y = np.random.default_rng(42).standard_normal(100)
        result = cross_fit(X, y, n_folds=5, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cross_fit"

    def test_fold_count(self):
        X = np.random.default_rng(42).standard_normal((100, 5))
        y = np.random.default_rng(42).standard_normal(100)
        result = cross_fit(X, y, n_folds=5, seed=42)
        assert len(result.extra["folds"]) == 5

    def test_no_overlap(self):
        X = np.random.default_rng(42).standard_normal((20, 3))
        y = np.random.default_rng(42).standard_normal(20)
        result = cross_fit(X, y, n_folds=4, seed=42)
        all_test = np.concatenate([f["test"] for f in result.extra["folds"]])
        assert len(set(all_test)) == 20

    def test_alias(self):
        assert xfit is cross_fit
