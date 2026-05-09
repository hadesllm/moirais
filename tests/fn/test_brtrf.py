"""Tests for moirais.fn.brtrf -- Brent's method."""

import numpy as np
import pytest
from moirais.fn.brtrf import brent_root, brtrf
from moirais.fn._containers import DescriptiveResult


class TestBrtrf:
    def test_alias(self):
        assert brtrf is brent_root

    def test_sqrt2(self):
        r = brent_root(lambda x: x**2 - 2, 1.0, 2.0)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value - np.sqrt(2)) < 1e-10

    def test_cosine(self):
        r = brent_root(np.cos, 1.0, 2.0)
        assert abs(r.value - np.pi / 2) < 1e-10

    def test_same_sign_raises(self):
        with pytest.raises(ValueError):
            brent_root(lambda x: x**2 + 1, -1.0, 1.0)
