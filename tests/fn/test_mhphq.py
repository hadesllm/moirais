"""Tests for moirais.fn.mhphq -- PHQ-9 score."""

import pytest
from moirais.fn.mhphq import phq9_score


class TestPHQ9:
    def test_minimal(self):
        res = phq9_score([0, 0, 1, 0, 0, 0, 0, 0, 0])
        assert res.extra["severity"] == "minimal"

    def test_moderate(self):
        res = phq9_score([2, 1, 1, 1, 1, 1, 1, 1, 1])
        assert res.extra["severity"] == "moderate"

    def test_wrong_len(self):
        with pytest.raises(ValueError):
            phq9_score([1, 2])
