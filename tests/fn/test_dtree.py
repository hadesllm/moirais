"""Test decision_tree (dtree)."""
import numpy as np
from morie.fn.dtree import decision_tree, dtree
from morie.fn._containers import DescriptiveResult


class TestDtree:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        y = (X[:, 0] > 0).astype(int)
        result = decision_tree(X[:50], y[:50], X[50:])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "decision_tree"

    def test_predictions_shape(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        y = (X[:, 0] > 0).astype(int)
        result = decision_tree(X[:50], y[:50], X[50:])
        assert len(result.extra["predictions"]) == 10

    def test_feature_importance(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        y = (X[:, 0] > 0).astype(int)
        result = decision_tree(X[:50], y[:50], X[50:])
        imp = result.extra["feature_importance"]
        assert len(imp) == 4
        assert np.isclose(imp.sum(), 1.0, atol=0.01) or imp.sum() == 0

    def test_alias(self):
        assert dtree is decision_tree
