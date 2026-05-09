"""Tests for moirais.fn.bnut -- NUTS sampler."""

import numpy as np
from moirais.fn.bnut import nuts_sampler


def _log_normal(x):
    return -0.5 * float(x @ x)


def _grad_log_normal(x):
    return -x


def test_returns_dict():
    result = nuts_sampler(_log_normal, _grad_log_normal, [0.0], n_iter=50, max_depth=5)
    assert isinstance(result, dict)
    assert "samples" in result


def test_samples_shape():
    result = nuts_sampler(_log_normal, _grad_log_normal, [0.0], n_iter=50, max_depth=5)
    assert result["samples"].shape == (50, 1)


def test_mean_tree_depth_positive():
    result = nuts_sampler(_log_normal, _grad_log_normal, [0.0], n_iter=50, max_depth=5)
    assert result["mean_tree_depth"] > 0
