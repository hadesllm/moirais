"""Test dtmxs."""
import numpy as np
import pytest
from morie.fn.dtmxs import dtmxs


def test_dtmxs_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmxs(x=x, n=50)
    assert r.value is not None


def test_dtmxs_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmxs(x=x, n=50)
    assert r.name
