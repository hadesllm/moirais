"""Test grtcrc."""
import numpy as np
import pytest
from moirais.fn.grtcrc import grtcrc


def test_grtcrc_basic():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = grtcrc(coords=coords, n=20)
    assert r.value is not None


def test_grtcrc_description():
    rng = np.random.default_rng(42)
    coords = rng.uniform(-90, 90, (20, 2))
    r = grtcrc(coords=coords, n=20)
    assert r.name
