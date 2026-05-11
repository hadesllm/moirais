"""Test cldiv."""
import numpy as np
import pytest
from morie.fn.cldiv import cldiv


def test_cldiv_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldiv(data=data, n=30, k=3)
    assert r.value is not None


def test_cldiv_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cldiv(data=data, n=30, k=3)
    assert r.name
