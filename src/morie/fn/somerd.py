# morie.fn — function file (hadesllm/morie)
"""Somers' D with R-style verbose result."""

from typing import Sequence, Union
import numpy as np
from scipy.stats import somersd as _somersd


def somerd(x: Union[Sequence, np.ndarray],
           y: Union[Sequence, np.ndarray]):
    """Somers' D - asymmetric ordinal-ordinal association."""
    from ._richresult import hypothesis_test_result
    res = _somersd(np.asarray(x), np.asarray(y))
    return hypothesis_test_result(
        test_name="Somers' D (asymmetric ordinal-ordinal)",
        statistic=float(res.statistic), pvalue=float(res.pvalue),
        extra_summary=[
            ("Predictor n", int(np.asarray(x).size)),
            ("Outcome n", int(np.asarray(y).size)),
            ("Strength", "negligible" if abs(res.statistic) < 0.1 else
                         "weak" if abs(res.statistic) < 0.3 else
                         "moderate" if abs(res.statistic) < 0.5 else "strong"),
        ],
        extra_payload={"table": np.asarray(res.table).tolist()},
    )
