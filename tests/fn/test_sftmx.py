"""Tests for moirais.fn.sftmx — softmax with temperature scaling."""
import numpy as np
import pytest
from moirais.fn.sftmx import softmax


class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        res = softmax(x)
        probs = res.extra["probabilities"]
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_low_temperature_sharper(self):
        x = np.array([1.0, 2.0, 3.0])
        res_high = softmax(x, temperature=2.0)
        res_low = softmax(x, temperature=0.1)
        assert res_low.extra["probabilities"][2] > res_high.extra["probabilities"][2]

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            softmax(np.array([1.0, 2.0]), temperature=0.0)
        with pytest.raises(ValueError):
            softmax(np.array([1.0, 2.0]), temperature=-1.0)
