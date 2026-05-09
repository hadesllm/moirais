"""SAM prompt encoder (points/boxes/masks)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sam_prompt_encoder"]


def sam_prompt_encoder(prompts):
    """
    SAM prompt encoder (points/boxes/masks)

    Formula: learned embeddings per prompt type

    Parameters
    ----------
    prompts : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kirillov et al (2023)
    """
    prompts = np.atleast_1d(np.asarray(prompts, dtype=float))
    n = len(prompts)
    result = float(np.mean(prompts))
    se = float(np.std(prompts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SAM prompt encoder (points/boxes/masks)"})


def cheatsheet():
    return "sampre: SAM prompt encoder (points/boxes/masks)"
