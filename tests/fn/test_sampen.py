"""Tests for sampen — Sample entropy."""
import numpy as np
from morie.fn.sampen import sample_entropy
from morie.fn._containers import DescriptiveResult


def test_sampen_basic(rng):
    x = rng.standard_normal(200)
    result = sample_entropy(x, m=2, r=0.2)
    assert isinstance(result, DescriptiveResult)
    assert result.value > 0


def test_sampen_periodic_low():
    x = np.tile([1.0, 2.0, 3.0, 2.0], 50)
    result = sample_entropy(x, m=2, r=0.3)
    assert result.value < 1.0


def test_sampen_random_higher(rng):
    periodic = np.tile([1.0, 2.0, 3.0, 2.0], 50)
    random = rng.standard_normal(200)
    se_periodic = sample_entropy(periodic, m=2, r=0.3).value
    se_random = sample_entropy(random, m=2, r=0.3).value
    assert se_random > se_periodic
