"""Tests for rcrnc -- Recurrence quantification analysis."""
import numpy as np
import pytest
from moirais.fn.rcrnc import rcrnc
from moirais.fn._containers import DescriptiveResult


def test_rcrnc_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = rcrnc(x, m=2, tau=1)
    assert isinstance(result, DescriptiveResult)
    assert "recurrence_rate" in result.extra
    assert "determinism" in result.extra


def test_rcrnc_periodic_high_rr():
    t = np.arange(200)
    x = np.sin(2 * np.pi * t / 20)
    result = rcrnc(x, m=2, tau=1, epsilon=0.5)
    assert result.extra["recurrence_rate"] > 0


def test_rcrnc_too_short():
    with pytest.raises(ValueError):
        rcrnc(np.array([1.0, 2.0, 3.0]), m=5, tau=1)
