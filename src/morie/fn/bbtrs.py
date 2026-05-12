# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""Blackbox_transpose scaling for large-N surveys"""

import numpy as np

from .._spatial_voting import blackbox_scaling
from ._containers import DescriptiveResult


def blackbox_transpose(data=None, *, dims=2, n=50):
    """Blackbox_transpose scaling for large-N survey data.

    Transposes the data matrix before applying Blackbox scaling,
    making it feasible for datasets with more respondents than stimuli.

    Returns
    -------
    DescriptiveResult
    """
    if data is None:
        data = np.random.default_rng(0).standard_normal((n, 5))
    data = np.asarray(data, dtype=float)
    result = blackbox_scaling(data.T, n_dims=dims)
    return DescriptiveResult(
        name="Blackbox transpose",
        value=result.get("fit", 0.0),
        extra={"dims": dims, "n_respondents": data.shape[0], "n_stimuli": data.shape[1], **result},
    )


short = "bbtrs"
alias = "blackbox_transpose"
quote = "The road up and the road down are the same thing. -- Heraclitus"
blackbox_transpose = blackbox_transpose


def cheatsheet() -> str:
    return "blackbox_transpose({}) -> Blackbox_transpose scaling for large-N surveys"
