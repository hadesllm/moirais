# moirais.fn — function file (hadesllm/moirais)
"""Elastic net regression with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from sklearn.linear_model import ElasticNet


def elnetr(X: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray],
           alpha: float = 1.0, l1_ratio: float = 0.5,
           fit_intercept: bool = True, max_iter: int = 10000):
    """Elastic net - convex combination of L1 and L2 penalties."""
    from ._richresult import RichResult
    X = np.asarray(X, dtype=float); y = np.asarray(y, dtype=float)
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                        fit_intercept=fit_intercept, max_iter=max_iter)
    model.fit(X, y)
    coefs = model.coef_
    intercept = float(model.intercept_) if fit_intercept else 0.0
    r2 = float(model.score(X, y))
    nonzero = int(np.sum(np.abs(coefs) > 1e-10))
    coef_rows = [[f"x{i+1}", f"{c:.6g}", "selected" if abs(c) > 1e-10 else "zeroed"]
                 for i, c in enumerate(coefs)]
    return RichResult(
        title="Elastic net regression",
        summary_lines=[
            ("Alpha", alpha), ("L1 ratio (rho)", l1_ratio),
            ("R^2 (training)", r2), ("Intercept", intercept),
            ("Predictors selected", f"{nonzero} of {len(coefs)}"),
            ("n observations", len(y)),
        ],
        tables=[{
            "title": "Coefficients:",
            "headers": ["Predictor", "Coefficient", "Status"],
            "rows": coef_rows,
        }],
        interpretation=(f"l1_ratio={l1_ratio}: " +
                        ("pure Ridge (no selection)" if l1_ratio == 0 else
                         "pure Lasso (full selection)" if l1_ratio == 1 else
                         f"mix - {l1_ratio*100:.0f}% L1 / {(1-l1_ratio)*100:.0f}% L2")),
        payload={"coef": coefs.tolist(), "intercept": intercept,
                 "r2": r2, "alpha": alpha, "l1_ratio": l1_ratio, "nonzero": nonzero},
    )
