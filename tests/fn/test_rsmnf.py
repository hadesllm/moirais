"""Test rsmnf."""
import numpy as np
import pytest
from moirais.fn.rsmnf import rsmnf


def test_rsmnf_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmnf(pixels=pixels, n=40)
    assert r.value is not None


def test_rsmnf_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rsmnf(pixels=pixels, n=40)
    assert r.name
