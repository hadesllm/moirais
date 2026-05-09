# moirais.fn — function file (hadesllm/moirais)
"""It is the mark of an educated mind to entertain a thought without accepting it. — Aristotle"""

from __future__ import annotations

import os

from ._containers import DescriptiveResult

_VALID_FORMATS = {"png", "pdf", "svg", "jpg", "jpeg", "eps", "tiff"}


def save_figure(
    fig: object,
    path: str,
    dpi: int = 150,
    fmt: str = "png",
) -> DescriptiveResult:
    """Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object.
    path : str
        Output file path.
    dpi : int, default 150
    fmt : str, default "png"

    Returns
    -------
    DescriptiveResult
    """
    fmt = fmt.lower().strip(".")
    if fmt not in _VALID_FORMATS:
        raise ValueError(f"Format '{fmt}' not in {sorted(_VALID_FORMATS)}")
    if not path.lower().endswith(f".{fmt}"):
        path = f"{path}.{fmt}"

    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)

    fig.savefig(path, dpi=dpi, format=fmt, bbox_inches="tight")

    return DescriptiveResult(
        name="Save figure",
        value=path,
        extra={"dpi": dpi, "format": fmt, "path": path},
    )


savfg = save_figure


def cheatsheet() -> str:
    return "save_figure({}) -> Save figure helper. 'Great shot kid, that was one in a milli"
