"""Test samob."""
import numpy as np
import pytest
from morie.fn.samob import samob


def test_samob_basic():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samob(values=vals, n=25)
    assert r.value is not None


def test_samob_description():
    rng = np.random.default_rng(42)
    vals = rng.standard_normal(25)
    r = samob(values=vals, n=25)
    assert r.name
