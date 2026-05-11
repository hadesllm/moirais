"""Tests for hrvfq -- HRV frequency-domain metrics."""
import numpy as np
import pytest
from morie.fn.hrvfq import hrvfq
from morie.fn._containers import DescriptiveResult


def test_hrvfq_basic():
    rng = np.random.default_rng(42)
    rr = 0.8 + rng.standard_normal(200) * 0.02
    result = hrvfq(rr)
    assert isinstance(result, DescriptiveResult)
    assert "lf" in result.extra
    assert "hf" in result.extra


def test_hrvfq_powers_nonnegative():
    rng = np.random.default_rng(7)
    rr = 0.9 + rng.standard_normal(300) * 0.03
    result = hrvfq(rr)
    assert result.extra["vlf"] >= 0
    assert result.extra["lf"] >= 0
    assert result.extra["hf"] >= 0


def test_hrvfq_too_short():
    with pytest.raises(ValueError):
        hrvfq(np.array([0.8, 0.9, 0.85]))
