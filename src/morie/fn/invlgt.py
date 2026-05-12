# morie.fn -- function file (hadesllm/morie)
"""Inverse logit / sigmoid."""

import math
def invlgt(x):
    """Inverse logit: σ(x) = 1 / (1 + exp(-x)).

    The link inverse for logistic regression -- maps ℝ -> (0, 1).
    """
    if isinstance(x, (int, float)):
        return 1.0 / (1.0 + math.exp(-x))
    import numpy as np
    a = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-a))
