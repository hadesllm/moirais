"""Test dtmvt."""
import numpy as np
import pytest
from moirais.fn.dtmvt import dtmvt


def test_dtmvt_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmvt(x=x, n=50)
    assert r.value is not None


def test_dtmvt_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmvt(x=x, n=50)
    assert r.name
