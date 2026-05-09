"""Tests for moirais.fn.rbnd — Rosenbaum bounds sensitivity analysis."""

import numpy as np
import pandas as pd
import pytest

from moirais.fn.rbnd import sensitivity_rosenbaum


@pytest.fixture()
def rbnd_data():
    """Synthetic treatment/control data with a clear treatment effect."""
    rng = np.random.default_rng(42)
    n = 100
    t = np.concatenate([np.ones(n // 2), np.zeros(n // 2)])
    y = np.where(t == 1, rng.normal(3.0, 1.0, n), rng.normal(1.0, 1.0, n))
    x = rng.standard_normal(n)
    return pd.DataFrame({"y": y, "t": t, "x": x})


def test_returns_dataframe(rbnd_data):
    """sensitivity_rosenbaum returns a DataFrame with Gamma/p columns."""
    result = sensitivity_rosenbaum(
        rbnd_data, treatment="t", outcome="y", covariates=["x"],
    )
    assert isinstance(result, pd.DataFrame)
    for col in ("Gamma", "p_lower", "p_upper"):
        assert col in result.columns


def test_gamma_range_correct(rbnd_data):
    """Gamma column spans from 1.0 to 3.0 (default range)."""
    result = sensitivity_rosenbaum(
        rbnd_data, treatment="t", outcome="y", covariates=["x"],
    )
    assert result["Gamma"].iloc[0] == pytest.approx(1.0)
    assert result["Gamma"].iloc[-1] == pytest.approx(3.0)


def test_p_upper_increases_with_gamma(rbnd_data):
    """p_upper should generally increase as Gamma increases (sensitivity)."""
    result = sensitivity_rosenbaum(
        rbnd_data, treatment="t", outcome="y", covariates=["x"],
    )
    # At minimum, final p_upper >= first p_upper
    assert result["p_upper"].iloc[-1] >= result["p_upper"].iloc[0]


def test_invalid_gamma_range_raises(rbnd_data):
    """gamma_range with min < 1.0 should raise ValueError."""
    with pytest.raises(ValueError, match="Minimum Gamma"):
        sensitivity_rosenbaum(
            rbnd_data, treatment="t", outcome="y",
            covariates=["x"], gamma_range=(0.5, 2.0),
        )


def test_custom_n_gamma(rbnd_data):
    """Custom n_gamma should produce the right number of rows."""
    result = sensitivity_rosenbaum(
        rbnd_data, treatment="t", outcome="y",
        covariates=["x"], n_gamma=5,
    )
    assert len(result) == 5
