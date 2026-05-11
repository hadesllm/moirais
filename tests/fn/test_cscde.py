"""Test cascade_classify (cscde)."""
import numpy as np
from morie.fn.cscde import cascade_classify, cscde
from morie.fn._containers import DescriptiveResult


class TestCscde:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 3))
        clf1 = lambda x: x[:, 0]
        clf2 = lambda x: x[:, 1]
        result = cascade_classify(X, [clf1, clf2], [0.5, 0.3])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cascade_classify"
        assert len(result.extra["predictions"]) == 20

    def test_all_accepted(self):
        X = np.ones((5, 2))
        clf = lambda x: np.ones(len(x)) * 10
        result = cascade_classify(X, [clf], [0.0])
        assert result.extra["n_accepted"] == 5

    def test_alias(self):
        assert cscde is cascade_classify
