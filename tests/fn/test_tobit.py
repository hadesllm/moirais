"""Test tobit."""
import numpy as np
import pandas as pd
import pytest
from moirais.fn.tobit import tobit_model


def test_tobit_basic():
    rng = np.random.default_rng(42)
    y = rng.standard_normal(100).clip(0)
    x = rng.standard_normal(100)
    df = pd.DataFrame({"y": y, "x": x})
    r = tobit_model(df, y="y", x="x", lower=0.0)
    assert r.method == "Tobit"
    assert r.n == 100


def test_tobit_censored_count():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(50).clip(0)
    df = pd.DataFrame({"y": y, "x": rng.standard_normal(50)})
    r = tobit_model(df, y="y", x="x")
    assert r.extra["n_censored"] >= 0
