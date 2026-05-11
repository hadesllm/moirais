"""Tests for morie.fn.bbf — Bayes factor (BIC)."""

import pytest

from morie.fn.bbf import bayes_factor


class TestBayesFactor:
    def test_favors_alternative(self):
        res = bayes_factor(loglik_1=-100, loglik_0=-120, k_1=3, k_0=2, n=100)
        assert res.estimate > 1

    def test_favors_null(self):
        res = bayes_factor(loglik_1=-120, loglik_0=-100, k_1=5, k_0=2, n=100)
        assert res.estimate < 1

    def test_interpretation(self):
        res = bayes_factor(loglik_1=-50, loglik_0=-100, k_1=3, k_0=2, n=200)
        assert res.extra["interpretation"] in ["decisive", "strong", "substantial",
                                                 "barely worth mentioning", "supports null"]
