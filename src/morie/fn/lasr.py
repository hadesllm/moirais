# morie.fn -- function file (hadesllm/morie)
"""Lasso regression with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from sklearn.linear_model import Lasso


def lasr(X: Union[Sequence, np.ndarray],
         y: Union[Sequence, np.ndarray],
         alpha: float = 1.0,
         fit_intercept: bool = True,
         max_iter: int = 10000):
    """Lasso regression (L1-regularized OLS) - performs variable selection."""
    from ._richresult import RichResult
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)
    model.fit(X, y)
    coefs = model.coef_
    intercept = float(model.intercept_) if fit_intercept else 0.0
    r2 = float(model.score(X, y))
    nonzero = int(np.sum(np.abs(coefs) > 1e-10))
    coef_rows = [[f"x{i+1}", f"{c:.6g}", "selected" if abs(c) > 1e-10 else "zeroed"]
                 for i, c in enumerate(coefs)]
    return RichResult(
        title="Lasso regression",
        summary_lines=[
            ("Alpha (L1)", alpha), ("R^2 (training)", r2),
            ("Intercept", intercept),
            ("Predictors selected", f"{nonzero} of {len(coefs)}"),
            ("n observations", len(y)),
        ],
        tables=[{
            "title": "Coefficients:",
            "headers": ["Predictor", "Coefficient", "Status"],
            "rows": coef_rows,
        }],
        warnings=[] if nonzero > 0 else
                 ["all coefficients zeroed - alpha is too large for this data; "
                  "try a smaller value or use cross-validated LassoCV."],
        payload={"coef": coefs.tolist(), "intercept": intercept,
                 "r2": r2, "alpha": alpha, "nonzero": nonzero},
    )
