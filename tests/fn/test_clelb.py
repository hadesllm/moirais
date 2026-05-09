"""Test clelb."""
import numpy as np
import pytest
from moirais.fn.clelb import clelb


def test_clelb_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clelb(data=data, n=30, k=3)
    assert r.value is not None


def test_clelb_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = clelb(data=data, n=30, k=3)
    assert r.name
