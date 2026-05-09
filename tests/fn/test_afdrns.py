"""Test afdrns."""
import numpy as np
import pytest
from moirais.fn.afdrns import afdrns


def test_afdrns_basic():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afdrns(yield_data=yld, soil=soil, n=20)
    assert r.value is not None


def test_afdrns_description():
    rng = np.random.default_rng(42)
    yld = rng.uniform(50, 200, 20)
    soil = rng.uniform(0, 1, 20)
    r = afdrns(yield_data=yld, soil=soil, n=20)
    assert r.name
