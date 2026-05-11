"""Test nested_cv."""
import numpy as np
from morie.fn.nestc import nested_cv, nestc
from morie.fn._containers import DescriptiveResult


class TestNestedCv:
    def test_basic(self):
        X = np.random.default_rng(42).standard_normal((50, 3))
        y = np.random.default_rng(42).standard_normal(50)
        result = nested_cv(X, y, inner_k=3, outer_k=5, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "nested_cv"

    def test_outer_fold_count(self):
        X = np.random.default_rng(42).standard_normal((50, 3))
        y = np.random.default_rng(42).standard_normal(50)
        result = nested_cv(X, y, inner_k=3, outer_k=5, seed=42)
        assert len(result.extra["folds"]) == 5

    def test_inner_fold_count(self):
        X = np.random.default_rng(42).standard_normal((50, 3))
        y = np.random.default_rng(42).standard_normal(50)
        result = nested_cv(X, y, inner_k=3, outer_k=5, seed=42)
        for fold in result.extra["folds"]:
            assert len(fold["inner_folds"]) == 3

    def test_alias(self):
        assert nestc is nested_cv
