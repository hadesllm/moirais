"""Test samoe."""
import numpy as np
import pytest
from moirais.fn.samoe import samoe


def test_samoe_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samoe(values=vals, n=25)
    assert r.value is not None


def test_samoe_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samoe(values=vals, n=25)
    assert r.name
