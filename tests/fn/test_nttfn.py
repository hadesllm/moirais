"""Test ntt_transform."""
import numpy as np
import pytest
from moirais.fn.nttfn import ntt_transform
from moirais.fn._containers import DescriptiveResult


class TestNttTransform:
    def test_basic(self):
        x = [1] + [0] * 255
        result = ntt_transform(poly=x, q=3329, inverse=False)
        assert isinstance(result, DescriptiveResult)

    def test_output_type(self):
        x = [1] + [0] * 255
        result = ntt_transform(poly=x, q=3329, inverse=False)
        assert "result" in result.extra

    def test_roundtrip(self):
        x = [1] + [0] * 255
        fwd = ntt_transform(poly=x, q=3329, inverse=False)
        inv = ntt_transform(poly=fwd.extra["result"], q=3329, inverse=True)
        recovered = list(np.asarray(inv.extra["result"]) % 3329)
        assert recovered[0] == 1
        assert all(v == 0 for v in recovered[1:])
