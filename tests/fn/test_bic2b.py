"""Tests for moirais.fn.bic2b -- BIC-based Bayes factor."""

import math
from moirais.fn.bic2b import bayes_factor_bic


def test_returns_dict():
    result = bayes_factor_bic(100.0, 110.0)
    assert isinstance(result, dict)
    assert "bf01" in result


def test_lower_bic_null_favours_null():
    result = bayes_factor_bic(100.0, 120.0)
    assert result["bf01"] > 1.0


def test_lower_bic_alt_favours_alt():
    result = bayes_factor_bic(120.0, 100.0)
    assert result["bf10"] > 1.0


def test_equal_bic_gives_one():
    result = bayes_factor_bic(100.0, 100.0)
    assert abs(result["bf01"] - 1.0) < 1e-10


def test_bf01_bf10_reciprocal():
    result = bayes_factor_bic(90, 100)
    assert abs(result["bf01"] * result["bf10"] - 1.0) < 1e-10


def test_delta_bic():
    result = bayes_factor_bic(90, 100)
    assert abs(result["delta_bic"] - 10.0) < 1e-10


def test_known_value():
    result = bayes_factor_bic(100, 110)
    expected = math.exp(10 / 2.0)
    assert abs(result["bf01"] - expected) < 1e-6
