"""Test dtmvs."""
import numpy as np
import pytest
from morie.fn.dtmvs import dtmvs


def test_dtmvs_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmvs(x=x, n=50)
    assert r.value is not None


def test_dtmvs_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtmvs(x=x, n=50)
    assert r.name
