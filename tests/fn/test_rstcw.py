"""Test rstcw."""
import numpy as np
import pytest
from morie.fn.rstcw import rstcw


def test_rstcw_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rstcw(pixels=pixels, n=40)
    assert r.value is not None


def test_rstcw_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rstcw(pixels=pixels, n=40)
    assert r.name
