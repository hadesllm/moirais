"""Test dtfsk."""
import numpy as np
import pytest
from morie.fn.dtfsk import dtfsk


def test_dtfsk_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtfsk(x=x, n=50)
    assert r.value is not None


def test_dtfsk_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtfsk(x=x, n=50)
    assert r.name
