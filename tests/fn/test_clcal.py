"""Test clcal."""
import numpy as np
import pytest
from morie.fn.clcal import clcal


def test_clcal_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcal(data=data, n=30, k=3)
    assert r.value is not None


def test_clcal_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clcal(data=data, n=30, k=3)
    assert r.name
