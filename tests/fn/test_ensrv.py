"""Test ensemble_variance (ensrv)."""
import numpy as np
from moirais.fn.ensrv import ensemble_variance, ensrv
from moirais.fn._containers import DescriptiveResult


class TestEnsembleVariance:
    def test_basic(self):
        segments = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = ensemble_variance(segments)
        assert isinstance(result, DescriptiveResult)
        assert result.value is not None
        assert len(result.value) == 2

    def test_alias(self):
        assert ensrv is ensemble_variance
