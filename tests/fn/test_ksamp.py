"""Test ksamp."""
import numpy as np
import pytest
from morie.fn.ksamp import k_sample_anderson_darling


def test_ksamp_basic():
    rng = np.random.default_rng(42)
    a = rng.standard_normal(30)
    b = rng.standard_normal(30)
    r = k_sample_anderson_darling(a, b)
    assert r.test_name == "k-sample Anderson-Darling"
    assert r.p_value >= 0


def test_ksamp_different():
    rng = np.random.default_rng(7)
    a = rng.standard_normal(50)
    b = rng.standard_normal(50) + 3
    r = k_sample_anderson_darling(a, b)
    assert r.p_value < 0.05
