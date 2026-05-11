"""Test weight_decay."""
import numpy as np
from morie.fn.wdecm import weight_decay, wdecm
from morie.fn._containers import DescriptiveResult


class TestWeightDecay:
    def test_basic(self):
        params = [np.array([1.0, 2.0, 3.0])]
        result = weight_decay(params, lambda_=0.01)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "weight_decay"

    def test_penalty_value(self):
        params = [np.array([1.0, 0.0])]
        result = weight_decay(params, lambda_=0.1)
        assert abs(result.value - 0.05) < 1e-10

    def test_decay_shrinks(self):
        params = [np.array([1.0, 2.0])]
        result = weight_decay(params, lambda_=0.1)
        assert np.allclose(result.extra["decayed"][0], [0.9, 1.8])

    def test_alias(self):
        assert wdecm is weight_decay
