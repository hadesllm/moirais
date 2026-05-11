"""Test adpsmp."""
import numpy as np
import pytest
from morie.fn.adpsmp import adpsmp


def test_adpsmp_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = adpsmp(coords=coords, n=20)
    assert r.value is not None


def test_adpsmp_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = adpsmp(coords=coords, n=20)
    assert r.name
