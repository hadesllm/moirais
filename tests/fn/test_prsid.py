"""Test parseval_identity (prsid)."""
import numpy as np

from morie.fn.prsid import parseval_identity, prsid
from morie.fn._containers import DescriptiveResult


class TestParsevalIdentity:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(64)
        result = parseval_identity(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "parseval_identity"
        assert result.extra["holds"] is True

    def test_ratio_near_one(self):
        x = np.array([1.0, 0.0, -1.0, 0.0])
        result = parseval_identity(x)
        assert np.isclose(result.value, 1.0, atol=1e-10)

    def test_alias(self):
        assert prsid is parseval_identity
