"""Tests for moirais.fn.laapx -- Laplace approximation."""

import numpy as np
from moirais.fn.laapx import laplace_approximation


def test_returns_dict():
    result = laplace_approximation(lambda x: -0.5 * float(x @ x), [1.0])
    assert isinstance(result, dict)
    assert "mode" in result


def test_mode_near_zero():
    result = laplace_approximation(lambda x: -0.5 * float(x @ x), [5.0])
    assert abs(result["mode"][0]) < 0.5


def test_converged():
    result = laplace_approximation(lambda x: -0.5 * float(x @ x), [1.0])
    assert result["converged"]
