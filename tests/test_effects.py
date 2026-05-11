import pandas as pd

from morie.effects import estimate_ate


def test_estimate_ate_returns_float_estimate_and_standard_error():
    data = pd.DataFrame(
        {
            "outcome": [1.0, 3.0, 2.0, 4.0, 1.5, 4.5],
            "treated": [0, 1, 0, 1, 0, 1],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )

    ate, se = estimate_ate(
        data=data,
        outcome="outcome",
        treatment="treated",
        weights_col="weight",
    )

    assert isinstance(ate, float)
    assert isinstance(se, float)
    assert ate > 0
