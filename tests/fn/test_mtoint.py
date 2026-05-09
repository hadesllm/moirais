"""Tests for moirais.fn.mtoint — intersection."""

import pytest
from moirais.fn.mtoint import mto_intersection
from moirais.fn._containers import CrimeResult


class TestIntersection:
    def test_basic(self):
        r = mto_intersection([5, 10], [100000, 200000])
        assert isinstance(r, CrimeResult)
        assert r.rate > 0

    def test_mismatch(self):
        with pytest.raises(ValueError):
            mto_intersection([1], [1, 2])
