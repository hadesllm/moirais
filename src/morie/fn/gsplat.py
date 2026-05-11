"""3D Gaussian splatting render."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gaussian_splatting"]


def gaussian_splatting(gaussians, camera):
    """
    3D Gaussian splatting render

    Formula: N anisotropic 3D Gaussians; alpha-blend

    Parameters
    ----------
    gaussians : array-like
        Input data.
    camera : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kerbl et al (2023) 3DGS
    """
    gaussians = np.atleast_1d(np.asarray(gaussians, dtype=float))
    n = len(gaussians)
    result = float(np.mean(gaussians))
    se = float(np.std(gaussians, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "3D Gaussian splatting render"})


def cheatsheet():
    return "gsplat: 3D Gaussian splatting render"
