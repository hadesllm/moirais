"""Test samov."""
import numpy as np
import pytest
from moirais.fn.samov import samov


def test_samov_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samov(values=vals, n=25)
    assert isinstance(r.value, float)
    assert -1.5 <= r.value <= 1.5, f"Moran's I {r.value} outside plausible range"


def test_samov_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samov(values=vals, n=25)
    assert r.name
