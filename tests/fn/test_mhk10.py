"""Tests for moirais.fn.mhk10 -- K10 score."""

import pytest
from moirais.fn.mhk10 import k10_score


class TestK10:
    def test_low(self):
        res = k10_score([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert res.estimate == 10.0
        assert res.extra["distress_level"] == "low"

    def test_severe(self):
        res = k10_score([4, 4, 4, 3, 3, 3, 3, 3, 3, 3])
        assert res.extra["distress_level"] == "severe"

    def test_wrong_len(self):
        with pytest.raises(ValueError):
            k10_score([1, 2])
