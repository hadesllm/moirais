"""Tests for morie.fn.cdccm -- Charlson comorbidity index."""

import pytest
from morie.fn.cdccm import charlson_comorbidity


class TestCharlsonComorbidity:
    def test_basic(self):
        res = charlson_comorbidity({"mi": 1, "chf": 0, "copd": 1})
        assert res.estimate == pytest.approx(2.0)

    def test_with_age(self):
        res = charlson_comorbidity({"mi": 1}, age=65)
        assert res.extra["age_score"] == 2
        assert res.estimate == pytest.approx(3.0)

    def test_empty(self):
        res = charlson_comorbidity({})
        assert res.estimate == 0.0
