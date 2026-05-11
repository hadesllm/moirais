"""Test ar_to_lsf (ar2ls)."""
import numpy as np
from morie.fn.ar2ls import ar_to_lsf, ar2ls
from morie.fn._containers import DescriptiveResult


class TestAr2ls:
    def test_basic(self):
        result = ar_to_lsf([1.0, -0.5, 0.2])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "ar_to_lsf"
        lsf = result.extra["lsf"]
        assert len(lsf) > 0

    def test_lsf_in_range(self):
        result = ar_to_lsf([1.0, -0.8, 0.3])
        lsf = result.extra["lsf"]
        for w in lsf:
            assert 0 < w < np.pi

    def test_alias(self):
        assert ar2ls is ar_to_lsf
