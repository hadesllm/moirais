"""Test firth."""
import numpy as np
import pandas as pd
import pytest
from moirais.fn.firth import firth_logistic


def test_firth_basic():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(80)
    y = (x + rng.standard_normal(80) > 0).astype(float)
    df = pd.DataFrame({"y": y, "x": x})
    r = firth_logistic(df, y="y", x="x")
    assert r.method == "Firth logistic"
    assert r.n == 80


def test_firth_rare_events():
    rng = np.random.default_rng(99)
    x = rng.standard_normal(100)
    y = np.zeros(100)
    y[:5] = 1.0
    df = pd.DataFrame({"y": y, "x": x})
    r = firth_logistic(df, y="y", x="x")
    assert "(Intercept)" in r.coefficients
