"""Tests for entsg -- Signal entropy."""
import numpy as np
from moirais.fn.entsg import entsg
from moirais.fn._containers import DescriptiveResult


def test_entsg_shannon():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    result = entsg(x, method="shannon")
    assert isinstance(result, DescriptiveResult)
    assert result.value > 0


def test_entsg_sample():
    x = np.sin(2 * np.pi * 10 * np.arange(200) / 100)
    result = entsg(x, method="sample", m=2)
    assert result.extra["method"] == "sample"


def test_entsg_approx():
    x = np.random.default_rng(7).standard_normal(200)
    result = entsg(x, method="approx", m=2)
    assert result.extra["method"] == "approx"


def test_entsg_constant_low():
    x = np.ones(200) * 5.0
    result = entsg(x, method="shannon", n_bins=32)
    assert result.value < 1.0
