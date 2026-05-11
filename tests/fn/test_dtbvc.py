"""Test dtbvc."""
import numpy as np
import pytest
from morie.fn.dtbvc import dtbvc


def test_dtbvc_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbvc(x=x, n=50)
    assert r.value is not None


def test_dtbvc_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtbvc(x=x, n=50)
    assert r.name
