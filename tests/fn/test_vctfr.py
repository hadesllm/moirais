"""Tests for moirais.fn.vctfr — fear of crime."""

import pytest
import numpy as np
from moirais.fn.vctfr import victim_fear
from moirais.fn._containers import DescriptiveResult


class TestVictimFear:
    def test_basic(self):
        rng = np.random.default_rng(42)
        r = victim_fear(rng.integers(1, 6, 200))
        assert isinstance(r, DescriptiveResult)
        assert 1 <= r.value <= 5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            victim_fear([])
