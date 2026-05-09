"""Test rsndw."""
import numpy as np
import pytest
from moirais.fn.rsndw import rsndw


def test_rsndw_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndw(pixels=pixels, n=40)
    assert r.value is not None


def test_rsndw_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsndw(pixels=pixels, n=40)
    assert r.name
