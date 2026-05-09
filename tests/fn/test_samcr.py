"""Test samcr."""
import numpy as np
import pytest
from moirais.fn.samcr import samcr


def test_samcr_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samcr(values=vals, n=25)
    assert r.value is not None


def test_samcr_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samcr(values=vals, n=25)
    assert r.name
