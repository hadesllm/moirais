"""Test rschg."""
import numpy as np
import pytest
from moirais.fn.rschg import rschg


def test_rschg_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rschg(pixels=pixels, n=40)
    assert r.value is not None


def test_rschg_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rschg(pixels=pixels, n=40)
    assert r.name
