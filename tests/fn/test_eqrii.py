"""Tests for moirais.fn.eqrii — relative inequality index."""

import pytest
from moirais.fn.eqrii import relative_inequality
from moirais.fn._containers import ESRes


class TestRelativeInequality:
    def test_basic(self):
        r = relative_inequality([10, 8, 6, 4, 2], [0.1, 0.3, 0.5, 0.7, 0.9])
        assert isinstance(r, ESRes)
        assert r.extra["sii"] < 0

    def test_too_few(self):
        with pytest.raises(ValueError):
            relative_inequality([1, 2], [0.3, 0.7])
