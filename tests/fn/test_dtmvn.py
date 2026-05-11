"""Test dtmvn."""
import numpy as np
import pytest
from morie.fn.dtmvn import dtmvn


def test_dtmvn_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmvn(x=x, n=50)
    assert r.value is not None


def test_dtmvn_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmvn(x=x, n=50)
    assert r.name
