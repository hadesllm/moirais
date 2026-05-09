"""Tests for moirais.fn.rwmh -- Random walk MH."""

import numpy as np
from moirais.fn.rwmh import random_walk_mh


def test_returns_dict():
    result = random_walk_mh(lambda x: -0.5 * float(x @ x), [0.0])
    assert isinstance(result, dict)
    assert "samples" in result


def test_samples_shape():
    result = random_walk_mh(lambda x: -0.5 * float(x @ x), [0.0], n_iter=500)
    assert result["samples"].shape == (500, 1)


def test_final_step_size_adapted():
    result = random_walk_mh(lambda x: -0.5 * float(x @ x), [0.0], step_size=0.01, n_iter=2000)
    assert result["final_step_size"] != 0.01
