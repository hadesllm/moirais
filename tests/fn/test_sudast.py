"""Tests for moirais.fn.sudast -- DAST score."""

import pytest
from moirais.fn.sudast import dast_score


class TestDASTScore:
    def test_no_problems(self):
        res = dast_score([0] * 10)
        assert res.estimate == 0.0
        assert res.extra["zone"] == "no_problems"

    def test_moderate(self):
        res = dast_score([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        assert res.extra["zone"] == "moderate"

    def test_wrong_length(self):
        with pytest.raises(ValueError):
            dast_score([1, 0])
