"""Test azmeqa."""
import numpy as np
import pytest
from moirais.fn.azmeqa import azmeqa


def test_azmeqa_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = azmeqa(coords=coords, n=20)
    assert r.value is not None


def test_azmeqa_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = azmeqa(coords=coords, n=20)
    assert r.name
