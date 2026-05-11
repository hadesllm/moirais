"""Test negbn."""
import numpy as np
import pandas as pd
import pytest
from morie.fn.negbn import negative_binomial_reg


def test_negbn_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    y = rng.poisson(np.exp(0.5 + 0.3 * x))
    df = pd.DataFrame({"y": y, "x": x})
    r = negative_binomial_reg(df, y="y", x="x")
    assert r.method == "Negative Binomial (NB2)"
    assert r.n == 100


def test_negbn_coefficients():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(80)
    y = rng.poisson(np.exp(1.0 + 0.5 * x))
    df = pd.DataFrame({"y": y, "x": x})
    r = negative_binomial_reg(df, y="y", x="x")
    assert "(Intercept)" in r.coefficients
    assert "alpha" in r.extra
