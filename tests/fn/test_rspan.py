"""Test rspan."""
import numpy as np
import pytest
from morie.fn.rspan import rspan


def test_rspan_basic():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rspan(pixels=pixels, n=40)
    assert r.value is not None


def test_rspan_description():
    rng = np.random.default_rng(42)
    pixels = rng.uniform(0, 10000, (40, 4))
    r = rspan(pixels=pixels, n=40)
    assert r.name
