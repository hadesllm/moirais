"""Test cramer_rao_lower_bound (crlb)."""
import numpy as np

from morie.fn.crlb import cramer_rao_lower_bound, crlb
from morie.fn._containers import DescriptiveResult


class TestCramerRao:
    def test_scalar(self):
        result = cramer_rao_lower_bound(4.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cramer_rao_lower_bound"
        assert result.value == 0.25

    def test_matrix(self):
        fi = np.diag([2.0, 4.0])
        result = cramer_rao_lower_bound(fi)
        assert np.isclose(result.extra["crlb_diag"][0], 0.5)
        assert np.isclose(result.extra["crlb_diag"][1], 0.25)

    def test_alias(self):
        assert crlb is cramer_rao_lower_bound
