"""Tests for moirais.fn.bpro — Brownian bridge simulation."""

import numpy as np
import pytest

from moirais.fn.bpro import bpro


def test_endpoints_zero():
    result = bpro(500, seed=42)
    assert result["bridge"][0] == pytest.approx(0.0, abs=1e-12)
    assert result["bridge"][-1] == pytest.approx(0.0, abs=1e-12)


def test_n_points():
    result = bpro(100, seed=1)
    assert len(result["bridge"]) == 100
    assert len(result["t"]) == 100


def test_deterministic():
    r1 = bpro(50, seed=7)
    r2 = bpro(50, seed=7)
    np.testing.assert_array_equal(r1["bridge"], r2["bridge"])


def test_too_few_points_raises():
    with pytest.raises(ValueError, match="n_points must be >= 2"):
        bpro(1)
