"""Tests for moirais.fn.gate — Group Average Treatment Effect via AIPW within strata."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.gate import estimate_gate


@pytest.fixture()
def synth_data():
    rng = np.random.default_rng(42)
    n = 300
    x1 = rng.standard_normal(n)
    group = rng.choice(["A", "B", "C"], n)
    prob = 1 / (1 + np.exp(-(0.3 * x1)))
    t = rng.binomial(1, prob)
    y = 0.3 * t + 0.2 * x1 + rng.standard_normal(n) * 0.5
    return pd.DataFrame({"x1": x1, "group": group, "treatment": t, "outcome": y})


def test_returns_dataframe(synth_data):
    result = estimate_gate(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1"],
        group_col="group",
    )
    assert isinstance(result, pd.DataFrame)


def test_one_row_per_group(synth_data):
    result = estimate_gate(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1"],
        group_col="group",
    )
    assert len(result) == synth_data["group"].nunique()


def test_has_expected_columns(synth_data):
    result = estimate_gate(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1"],
        group_col="group",
    )
    for col in ("group", "ate", "se", "ci_lower", "ci_upper", "n"):
        assert col in result.columns


def test_group_values_match(synth_data):
    result = estimate_gate(
        synth_data,
        treatment="treatment",
        outcome="outcome",
        covariates=["x1"],
        group_col="group",
    )
    assert set(result["group"].tolist()) == set(synth_data["group"].unique())
