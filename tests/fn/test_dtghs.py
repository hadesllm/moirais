"""Test dtghs."""
import numpy as np
import pytest
from morie.fn.dtghs import dtghs


def test_dtghs_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtghs(x=x, n=50)
    assert r.value is not None


def test_dtghs_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtghs(x=x, n=50)
    assert r.name
