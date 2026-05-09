"""Test ccf_normalized (ccfn)."""
import numpy as np
from moirais.fn.ccfn import ccf_normalized, ccfn
from moirais.fn._containers import DescriptiveResult


class TestCcfn:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        y = rng.standard_normal(64)
        result = ccf_normalized(x, y)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ccf_normalized"

    def test_self_correlation_unity(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = ccf_normalized(x, x)
        assert np.isclose(result.extra["ccf"][0], 1.0, atol=1e-10)

    def test_alias(self):
        assert ccfn is ccf_normalized
