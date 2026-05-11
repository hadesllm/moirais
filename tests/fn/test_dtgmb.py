"""Test dtgmb."""
import numpy as np
import pytest
from morie.fn.dtgmb import dtgmb


def test_dtgmb_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtgmb(x=x, n=50)
    assert r.value is not None


def test_dtgmb_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtgmb(x=x, n=50)
    assert r.name
