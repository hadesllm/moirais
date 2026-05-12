# morie.fn -- function file (hadesllm/morie)
"""Expected frequencies under independence with R-style verbose result."""

from typing import Sequence, Union
import numpy as np


def expfrq(table: Union[Sequence, np.ndarray]):
    """Expected frequency under independence: E_ij = (row_i * col_j) / N."""
    from ._richresult import RichResult
    t = np.asarray(table, dtype=float)
    if t.ndim != 2:
        raise ValueError(f"table must be 2D, got shape {t.shape}.")
    rows = t.sum(axis=1, keepdims=True)
    cols = t.sum(axis=0, keepdims=True)
    n = t.sum()
    if n == 0:
        raise ValueError("table is empty.")
    expected = rows @ cols / n
    low_count_cells = int(np.sum(expected < 5))
    warnings = []
    if low_count_cells > 0:
        pct = 100 * low_count_cells / expected.size
        warnings.append(f"{low_count_cells} of {expected.size} cells "
                        f"({pct:.0f}%) have expected count < 5; "
                        "chi-squared approximation may be poor. "
                        "Consider Fisher's exact (`fishex`).")
    expected_rows = [["Row " + str(i)] + [f"{v:.2f}" for v in expected[i]]
                     for i in range(expected.shape[0])]
    return RichResult(
        title="Expected frequencies (independence)",
        summary_lines=[
            ("Total n", int(n)),
            ("Table shape", expected.shape),
            ("Cells with E < 5", f"{low_count_cells} of {expected.size}"),
            ("Min expected", float(expected.min())),
            ("Max expected", float(expected.max())),
        ],
        tables=[{
            "title": "Expected counts under independence:",
            "headers": [""] + [f"Col {j}" for j in range(expected.shape[1])],
            "rows": expected_rows,
        }],
        warnings=warnings,
        payload={"expected": expected.tolist(), "row_totals": rows.flatten().tolist(),
                 "col_totals": cols.flatten().tolist(), "total": float(n)},
    )
