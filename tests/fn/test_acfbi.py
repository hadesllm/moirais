"""Test acf_biased (acfbi)."""
import numpy as np
from moirais.fn.acfbi import acf_biased, acfbi
from moirais.fn._containers import DescriptiveResult


class TestAcfbi:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = acf_biased(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "acf_biased"

    def test_zero_lag_max(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = acf_biased(x)
        acf = result.extra["acf"]
        assert acf[0] >= np.max(np.abs(acf[1:]))

    def test_alias(self):
        assert acfbi is acf_biased
