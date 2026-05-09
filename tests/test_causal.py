import pandas as pd
import numpy as np
from moirais.causal import calculate_ipw_weights, compute_propensity_scores

def test_calculate_ipw_weights():
    # Setup mock data
    df = pd.DataFrame({
        "treated": [1, 1, 0, 0],
        "ps": [0.8, 0.9, 0.2, 0.1]
    })

    weights = calculate_ipw_weights(df, "treated", "ps")

    # Verify IPW formula: treated weight = 1/ps, control weight = 1/(1-ps)
    for i, row in df.iterrows():
        if row["treated"] == 1:
            expected = 1.0 / row["ps"]
        else:
            expected = 1.0 / (1.0 - row["ps"])
        assert np.isclose(weights[i], expected, atol=1e-10), (
            f"Row {i}: expected {expected:.6f} got {weights[i]:.6f} "
            f"(treated={row['treated']}, ps={row['ps']:.2f})"
        )

    # Verify specific values (1/0.8=1.25, 1/0.9~1.111, 1/0.8=1.25, 1/0.9~1.111)
    assert np.isclose(weights[0], 1.25)
    assert np.isclose(weights[1], 1.0 / 0.9, atol=1e-10)
    assert np.isclose(weights[2], 1.0 / 0.8, atol=1e-10)
    assert np.isclose(weights[3], 1.0 / 0.9, atol=1e-10)

    # All weights must be positive and finite
    assert all(np.isfinite(weights))
    assert all(w > 0 for w in weights)


def test_compute_propensity_scores_returns_probabilities():
    df = pd.DataFrame(
        {
            "treated": [0, 1, 0, 1, 0, 1],
            "age": [21, 22, 23, 24, 25, 26],
            "risk_score": [0.1, 0.7, 0.2, 0.8, 0.3, 0.9],
        }
    )

    propensity_scores = compute_propensity_scores(
        df,
        treatment="treated",
        covariates=["age", "risk_score"],
    )

    assert len(propensity_scores) == len(df)
    assert ((propensity_scores > 0) & (propensity_scores < 1)).all()
