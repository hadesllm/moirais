"""Tests for morie.fn.lr_ -- Likelihood Ratios."""

import pytest
from morie.fn.lr_ import likelihood_ratios


class TestLikelihoodRatios:
    def test_basic(self):
        result = likelihood_ratios(tp=80, fp=10, fn=20, tn=90)
        assert result["lr_pos"] > 1  # good test has LR+ > 1
        assert result["lr_neg"] < 1  # good test has LR- < 1

    def test_perfect_test(self):
        result = likelihood_ratios(tp=50, fp=0, fn=0, tn=50)
        assert result["lr_pos"] == float("inf")
        assert result["lr_neg"] == 0.0

    def test_ci_tuple(self):
        result = likelihood_ratios(tp=80, fp=10, fn=20, tn=90)
        assert isinstance(result["ci_pos"], tuple)
        assert len(result["ci_pos"]) == 2
        assert result["ci_pos"][0] < result["lr_pos"]

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            likelihood_ratios(tp=80, fp=-1, fn=20, tn=90)
