"""Tests for morie.fn.scren -- screening test performance."""

import numpy as np
import pytest
from morie.fn.scren import screening_performance


class TestScreening:
    def test_perfect_test(self):
        res = screening_performance(50, 0, 0, 50)
        assert res["sensitivity"] == pytest.approx(1.0)
        assert res["specificity"] == pytest.approx(1.0)
        assert res["ppv"] == pytest.approx(1.0)
        assert res["npv"] == pytest.approx(1.0)

    def test_known_values(self):
        res = screening_performance(80, 10, 20, 90)
        assert res["sensitivity"] == pytest.approx(0.8)
        assert res["specificity"] == pytest.approx(0.9)

    def test_lr_positive(self):
        res = screening_performance(80, 10, 20, 90)
        assert res["lr_positive"] == pytest.approx(0.8 / 0.1)

    def test_youden_index(self):
        res = screening_performance(90, 5, 10, 95)
        se = 90 / 100
        sp = 95 / 100
        assert res["youden_index"] == pytest.approx(se + sp - 1)

    def test_bayesian_ppv_with_prevalence(self):
        res = screening_performance(80, 10, 20, 90, prevalence=0.01)
        assert res["ppv"] < 0.5

    def test_negative_count_raises(self):
        with pytest.raises(ValueError):
            screening_performance(-1, 10, 20, 90)

    def test_accuracy(self):
        res = screening_performance(40, 10, 5, 45)
        assert res["accuracy"] == pytest.approx(85 / 100)
