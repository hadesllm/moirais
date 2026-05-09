"""Test soft_threshold_fn (softt)."""
import numpy as np
from moirais.fn.softt import soft_threshold_fn, softt
from moirais.fn._containers import DescriptiveResult


class TestSoftt:
    def test_basic(self):
        x = np.array([1.0, -0.5, 0.05, 2.0])
        result = soft_threshold_fn(x, tau=0.2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "soft_threshold_fn"

    def test_l1_value(self):
        x = np.array([1.0, -1.0, 0.0])
        r = soft_threshold_fn(x, tau=0.3)
        expected_prox = np.array([0.7, -0.7, 0.0])
        np.testing.assert_allclose(r.extra["prox"], expected_prox, atol=1e-10)
        np.testing.assert_allclose(r.value, 1.4, atol=1e-10)

    def test_sparsity(self):
        x = np.array([0.05, -0.05, 0.01, 3.0])
        r = soft_threshold_fn(x, tau=0.1)
        assert r.extra["sparsity"] == 0.75

    def test_alias(self):
        assert softt is soft_threshold_fn
