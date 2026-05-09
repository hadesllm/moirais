"""Tests for moirais.fn.syndm — Syndromic surveillance score."""

import pytest

from moirais.fn.syndm import syndromic_score


class TestSyndromicScore:
    def test_basic(self):
        res = syndromic_score({"fever": 10, "cough": 5, "fatigue": 3})
        assert res.estimate == pytest.approx(6.0)

    def test_weighted(self):
        res = syndromic_score(
            {"fever": 10, "cough": 5},
            weights={"fever": 2.0, "cough": 1.0},
        )
        assert res.estimate == pytest.approx(25 / 3.0)

    def test_empty(self):
        with pytest.raises(ValueError):
            syndromic_score({})
