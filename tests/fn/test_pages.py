"""Test pages."""
import numpy as np
import pytest
from morie.fn.pages import page_trend_test


def test_pages_basic():
    R = np.array([[1, 2, 3], [1, 3, 2], [2, 1, 3], [1, 2, 3]])
    r = page_trend_test(R)
    assert r.test_name == "Page's L"
    assert r.statistic > 0


def test_pages_no_trend():
    rng = np.random.default_rng(42)
    R = np.zeros((10, 3))
    for i in range(10):
        R[i] = rng.permutation([1, 2, 3])
    r = page_trend_test(R)
    assert r.p_value >= 0
