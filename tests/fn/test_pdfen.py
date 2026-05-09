"""Tests for pdf_estimate."""
import numpy as np
import pytest
from moirais.fn.pdfen import pdf_estimate, pdfen


def test_basic():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, 1000)
    r = pdf_estimate(x, bins=30)
    assert r.value > 0
    assert len(r.extra["density"]) == 30


def test_alias():
    assert pdfen is pdf_estimate


def test_too_few():
    with pytest.raises(ValueError):
        pdf_estimate([1])
