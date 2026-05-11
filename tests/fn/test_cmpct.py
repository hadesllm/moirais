"""Test cmpct."""
import pytest
import numpy as np
from morie.fn.cmpct import compactify_torus


def test_cmpct_basic():
    r = compactify_torus(radii=[1.0, 1.0])
    assert r.value == pytest.approx((2 * np.pi) ** 2)


def test_cmpct_default():
    r = compactify_torus()
    assert r.extra["dimension"] == 2
