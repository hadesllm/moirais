"""Test curvelet (curlt)."""
import numpy as np
from moirais.fn.curlt import curvelet, curlt
from moirais.fn._containers import DescriptiveResult


class TestCurlt:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        result = curvelet(x, n_scales=4)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "curvelet"
        assert result.value == 4

    def test_coefficients_count(self):
        x = np.random.default_rng(0).standard_normal(128)
        r = curvelet(x, n_scales=3)
        assert len(r.extra["coefficients"]) == 3
        assert len(r.extra["energies"]) == 3

    def test_alias(self):
        assert curlt is curvelet
