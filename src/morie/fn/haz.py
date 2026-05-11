# morie.fn — function file (hadesllm/morie)
"""Hazard rate λ(t) with R-style verbose result."""


def haz(pdf_t: float, surv_t: float):
    """Hazard rate at time t: f(t) / S(t)."""
    from ._richresult import RichResult
    if surv_t <= 0:
        raise ValueError(f"survival must be positive, got {surv_t}.")
    if pdf_t < 0:
        raise ValueError(f"pdf must be non-negative, got {pdf_t}.")
    h = pdf_t / surv_t
    return RichResult(
        title="Hazard rate λ(t)",
        summary_lines=[
            ("λ(t) = f(t) / S(t)", h),
            ("PDF at t", pdf_t),
            ("Survival at t", surv_t),
            ("Conditional failure prob (1 - S(t)/S(t+))", "instantaneous"),
        ],
        interpretation=("Hazard rate is the instantaneous failure rate at t "
                        "given survival up to t. Constant hazard → exponential. "
                        "Increasing → ageing; decreasing → infant mortality."),
        payload={"value": h, "statistic": h, "pdf": pdf_t, "surv": surv_t},
    )
