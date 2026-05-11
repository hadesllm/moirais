"""Test aicc."""
import numpy as np
import pytest
from morie.fn.aicc import corrected_aic


def test_aicc_basic():
    r = corrected_aic(loglik=-100.0, n=50, k=3)
    assert r.name == "aicc"
    assert r.value > 200


def test_aicc_converges_to_aic():
    r = corrected_aic(loglik=-100.0, n=10000, k=3)
    aic = -2 * (-100.0) + 2 * 3
    assert abs(r.value - aic) < 0.1
