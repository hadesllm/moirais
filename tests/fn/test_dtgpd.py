"""Test dtgpd."""
import numpy as np
import pytest
from moirais.fn.dtgpd import dtgpd


def test_dtgpd_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtgpd(x=x, n=50)
    assert r.value is not None


def test_dtgpd_description():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    r = dtgpd(x=x, n=50)
    assert r.name
