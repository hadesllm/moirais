"""Test ar_normalize (arnrm)."""
import numpy as np
from moirais.fn.arnrm import ar_normalize, arnrm
from moirais.fn._containers import DescriptiveResult


class TestArnrm:
    def test_basic(self):
        result = ar_normalize([2.0, -1.0, 0.4])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ar_normalize"
        normed = result.extra["ar_normalized"]
        assert abs(normed[0] - 1.0) < 1e-10

    def test_already_normalized(self):
        result = ar_normalize([1.0, -0.5, 0.2])
        normed = result.extra["ar_normalized"]
        assert abs(normed[0] - 1.0) < 1e-10
        assert abs(normed[1] - (-0.5)) < 1e-10

    def test_alias(self):
        assert arnrm is ar_normalize
