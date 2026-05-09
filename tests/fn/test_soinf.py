"""Test soinf."""
import numpy as np
import pytest
from moirais.fn.soinf import soinf


def test_soinf_basic():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soinf(data=data, depth=depth, n=20)
    assert r.value is not None


def test_soinf_description():
    rng = np.random.default_rng(42)
    data = rng.uniform(0, 1, 20)
    depth = rng.uniform(0, 2, 20)
    r = soinf(data=data, depth=depth, n=20)
    assert r.name
