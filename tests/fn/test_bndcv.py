"""Tests for moirais.fn.bndcv — LOO cross-validation bandwidth."""

import numpy as np
import pytest
from moirais.fn.bndcv import bndcv


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = bndcv(x)
    assert isinstance(result, dict)
    for key in ("h_opt", "cv_scores", "h_grid", "n_obs"):
        assert key in result


def test_h_opt_positive():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = bndcv(x)
    assert result["h_opt"] > 0


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 5"):
        bndcv(np.ones(3))


def test_custom_grid():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = bndcv(x, h_grid=np.array([0.1, 0.5, 1.0]))
    assert len(result["cv_scores"]) == 3
