# morie.fn -- function file (hadesllm/morie)
"""Principal Component Analysis with R-style verbose result."""

from typing import Optional, Sequence, Union
import numpy as np
from sklearn.decomposition import PCA


def pcaprx(X: Union[Sequence, np.ndarray],
           n_components: Optional[int] = None,
           standardize: bool = True):
    """PCA: orthogonal components ranked by explained variance."""
    from ._richresult import RichResult
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    if standardize:
        mu = X.mean(axis=0); sd = X.std(axis=0, ddof=1)
        sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
    else:
        Xs = X
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    n_for_80 = int(np.argmax(cum_var >= 0.80) + 1) if (cum_var >= 0.80).any() else len(cum_var)
    n_for_95 = int(np.argmax(cum_var >= 0.95) + 1) if (cum_var >= 0.95).any() else len(cum_var)
    rows = [[f"PC{i+1}", f"{ev:.4g}", f"{evr*100:.2f}%", f"{cum*100:.2f}%"]
            for i, (ev, evr, cum) in enumerate(zip(pca.explained_variance_,
                                                    pca.explained_variance_ratio_,
                                                    cum_var))]
    return RichResult(
        title="Principal Component Analysis",
        summary_lines=[
            ("n observations", n), ("p variables", p),
            ("Components retained", len(pca.explained_variance_)),
            ("Standardized", standardize),
            ("Components for >=80% variance", n_for_80),
            ("Components for >=95% variance", n_for_95),
        ],
        tables=[{
            "title": "Variance explained per component:",
            "headers": ["PC", "Eigenvalue", "Var ratio", "Cumulative"],
            "rows": rows,
        }],
        warnings=[] if standardize or all(X.var(axis=0, ddof=1) > 1e-10) else
                 ["raw scaling: variables on different scales dominate PCA. "
                  "Pass standardize=True unless you have a reason."],
        payload={"components": pca.components_.tolist(),
                 "scores": scores.tolist(),
                 "explained_variance": pca.explained_variance_.tolist(),
                 "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                 "n_for_80": n_for_80, "n_for_95": n_for_95},
    )
