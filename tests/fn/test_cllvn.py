"""Test cllvn."""
import numpy as np
import pytest
from morie.fn.cllvn import cllvn


def test_cllvn_basic():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cllvn(data=data, n=30, k=3)
    assert r.value is not None


def test_cllvn_description():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((30, 2))
    r = cllvn(data=data, n=30, k=3)
    assert r.name
