"""Test smsim."""
import numpy as np
import pytest
from moirais.fn.smsim import smsim


def test_smsim_basic():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = smsim(points=pts, n=40)
    assert r.value is not None


def test_smsim_description():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 100, (40, 2))
    r = smsim(points=pts, n=40)
    assert r.name
