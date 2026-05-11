"""Tests for morie.fn.cglm — Complex survey GLM."""

import numpy as np
import pandas as pd
import pytest

from morie.fn.cglm import complex_survey_glm


@pytest.fixture()
def glm_data():
    """Synthetic survey data with binary outcome and weights."""
    rng = np.random.default_rng(42)
    n = 200
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)
    linpred = -0.5 + 0.8 * x1 - 0.3 * x2
    prob = 1 / (1 + np.exp(-linpred))
    y = (rng.uniform(size=n) < prob).astype(float)
    w = rng.uniform(1, 5, size=n)
    cluster = rng.choice(range(20), size=n)
    return pd.DataFrame({
        "y": y, "x1": x1, "x2": x2, "w": w, "cluster": cluster,
    })


def test_gaussian_returns_result(glm_data):
    """Gaussian GLM should return a fitted result with params."""
    glm_data["y_cont"] = glm_data["x1"] * 2 + np.random.default_rng(42).standard_normal(len(glm_data))
    result = complex_survey_glm(
        glm_data, formula="y_cont ~ x1 + x2", weight_col="w", family="gaussian",
    )
    assert hasattr(result, "params")
    assert "x1" in result.params.index


def test_binomial_returns_result(glm_data):
    """Binomial (logistic) GLM should return a fitted result."""
    result = complex_survey_glm(
        glm_data, formula="y ~ x1 + x2", weight_col="w", family="binomial",
    )
    assert hasattr(result, "params")
    assert len(result.params) == 3  # Intercept + x1 + x2


def test_cluster_robust_se(glm_data):
    """Cluster-robust SEs should differ from non-cluster SEs."""
    result_no_cluster = complex_survey_glm(
        glm_data, formula="y ~ x1 + x2", weight_col="w", family="binomial",
    )
    result_cluster = complex_survey_glm(
        glm_data, formula="y ~ x1 + x2", weight_col="w",
        family="binomial", cluster_col="cluster",
    )
    # SEs should generally differ (cluster adjustment)
    assert not np.allclose(result_no_cluster.bse.values, result_cluster.bse.values)


def test_unknown_family_raises(glm_data):
    """Unknown family string should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown family"):
        complex_survey_glm(
            glm_data, formula="y ~ x1", weight_col="w", family="tweedie",
        )


def test_missing_weight_col_raises(glm_data):
    """Missing weight column should raise ValueError."""
    with pytest.raises(ValueError, match="not found"):
        complex_survey_glm(
            glm_data, formula="y ~ x1", weight_col="nonexistent",
        )
