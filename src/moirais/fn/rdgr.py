# moirais.fn — function file (hadesllm/moirais)
"""Ridge regression with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from sklearn.linear_model import Ridge


def rdgr(X: Union[Sequence, np.ndarray],
         y: Union[Sequence, np.ndarray],
         alpha: float = 1.0,
         fit_intercept: bool = True):
    """Ridge regression (L2-regularized OLS).

    Minimizes ||y - X.beta||^2 + alpha ||beta||^2.
    """
    from ._richresult import RichResult
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept)
    model.fit(X, y)
    coefs = model.coef_
    intercept = float(model.intercept_) if fit_intercept else 0.0
    r2 = float(model.score(X, y))
    coef_rows = [[f"x{i+1}", f"{c:.6g}"] for i, c in enumerate(coefs)]
    return RichResult(
        title="Ridge regression",
        summary_lines=[
            ("Alpha (L2)", alpha), ("R^2 (training)", r2),
            ("Intercept", intercept), ("n predictors", len(coefs)),
            ("n observations", len(y)),
        ],
        tables=[{
            "title": "Coefficients:",
            "headers": ["Predictor", "Coefficient"],
            "rows": coef_rows,
        }],
        warnings=[] if alpha > 0 else ["alpha=0; behaves like plain OLS."],
        payload={"coef": coefs.tolist(), "intercept": intercept,
                 "r2": r2, "alpha": alpha},
    )
