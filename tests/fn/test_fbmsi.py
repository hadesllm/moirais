"""Test fbmsi."""
import numpy as np
import pytest
from morie.fn.fbmsi import fbmsi


def test_fbmsi_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = fbmsi(points=pts, n=40)
    assert r.value is not None


def test_fbmsi_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = fbmsi(points=pts, n=40)
    assert r.name
