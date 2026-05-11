"""Tests for fn/dsgwt.py -- Design weights."""
import numpy as np
import pandas as pd
import pytest

from morie.fn.dsgwt import dsgwt, compute_design_weights


def test_dsgwt_returns_series():
    df = pd.DataFrame({"stratum": ["A"] * 10 + ["B"] * 20})
    pop = {"A": 1000, "B": 2000}
    result = dsgwt(df, "stratum", pop)
    assert isinstance(result, pd.Series)
    assert len(result) == 30


def test_dsgwt_correct_values():
    df = pd.DataFrame({"stratum": ["A"] * 10 + ["B"] * 20})
    pop = {"A": 1000, "B": 2000}
    result = compute_design_weights(df, "stratum", pop)
    # A: 1000/10 = 100, B: 2000/20 = 100
    assert np.allclose(result[df["stratum"] == "A"].values, 100.0)
    assert np.allclose(result[df["stratum"] == "B"].values, 100.0)


def test_dsgwt_missing_stratum():
    df = pd.DataFrame({"stratum": ["A", "B", "C"]})
    pop = {"A": 100, "B": 200}  # missing C
    with pytest.raises(KeyError):
        dsgwt(df, "stratum", pop)
