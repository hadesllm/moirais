"""Tests for morie.fn.mhgad -- GAD-7 score."""

import pytest
from morie.fn.mhgad import gad7_score


class TestGAD7:
    def test_minimal(self):
        res = gad7_score([0, 1, 0, 0, 0, 0, 0])
        assert res.extra["severity"] == "minimal"

    def test_severe(self):
        res = gad7_score([3, 3, 3, 3, 2, 1, 1])
        assert res.extra["severity"] == "severe"

    def test_wrong_len(self):
        with pytest.raises(ValueError):
            gad7_score([1, 2, 3])
