"""Test ppmkd."""
import numpy as np
import pytest
from morie.fn.ppmkd import ppmkd


def test_ppmkd_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmkd(points=pts, n=30)
    assert isinstance(r.value, float)
    assert r.value > 0, "Point intensity must be positive"
    assert r.value == pytest.approx(30.0, rel=0.01), "Intensity should equal n/area = 30/1 = 30"


def test_ppmkd_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 1, (30, 2))
    r = ppmkd(points=pts, n=30)
    assert r.name
