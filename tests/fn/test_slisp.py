"""Tests for moirais.fn.slisp -- Slice sampler."""

import numpy as np
from moirais.fn.slisp import slice_sampler


def test_returns_dict():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=500)
    assert isinstance(result, dict)
    assert "samples" in result


def test_samples_length():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=1000)
    assert len(result["samples"]) == 1000


def test_mean_near_zero():
    result = slice_sampler(lambda x: -0.5 * x ** 2, n_iter=5000, seed=42)
    assert abs(np.mean(result["samples"])) < 0.5
