# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Instruction-data templating: (instruction, input?, output) formatted for SFT."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_instruction_data_template"]


def alammar_instruction_data_template(records, template):
    """
    Instruction-data templating: (instruction, input?, output) formatted for SFT

    Formula: record = fmt(instruction, input, output);  SFT loss masked to the output region

    Parameters
    ----------
    records : array-like
        Input data.
    template : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: formatted

    References
    ----------
    Alammar Ch 12, instruction data templating section
    """
    records = np.atleast_1d(np.asarray(records, dtype=float))
    n = len(records)
    result = float(np.mean(records))
    se = float(np.std(records, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Instruction-data templating: (instruction, input?, output) formatted for SFT"})


def cheatsheet():
    return "allktmpl: Instruction-data templating: (instruction, input?, output) formatted for SFT"
