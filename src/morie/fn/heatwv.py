# morie.fn — function file (hadesllm/morie)
"""Heat-wave detection via WMO percentile + consecutive-day criteria."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def heat_wave_detect(
    daily_max_C: np.ndarray,
    *,
    percentile: float = 90.0,
    min_consecutive_days: int = 3,
    baseline_window: tuple[int, int] | None = None,
) -> DescriptiveResult:
    """Detect heat-wave episodes in a daily-max-temperature time series.

    Uses the WMO/WHO consensus definition adopted by most national
    meteorological services: a heat wave is a run of N or more
    consecutive days where the daily maximum temperature exceeds a
    local percentile threshold of the historical baseline.

    Parameters
    ----------
    daily_max_C : array-like, shape (n_days,)
        Daily maximum surface air temperature, °C, in chronological
        order. Gaps should be filled (np.nan is not handled — drop or
        interpolate first).
    percentile : float, default 90.0
        Threshold percentile computed from the baseline window.
        Common choices: 90 (WMO ECA&D), 95 (more extreme), 97.5
        (recent heat-dome literature).
    min_consecutive_days : int, default 3
        Minimum run length to qualify as a heat wave. 3 is the
        ECMWF/WMO default; 2 is used for tropical climates; 5 by
        some US sources.
    baseline_window : tuple[int, int], optional
        (start, end) indices into `daily_max_C` defining the reference
        period for the percentile. Defaults to the entire series.
        Pass e.g. (0, 365*30) to use the first 30 years.

    Returns
    -------
    DescriptiveResult
        value = number of heat-wave days detected (integer-valued
        float, matching fn/ convention).
        extra contains:

        - ``threshold_C`` — the percentile-derived temperature cutoff
        - ``n_events`` — number of distinct heat-wave episodes
        - ``event_lengths`` — list of episode lengths in days
        - ``event_peaks_C`` — list of peak T_max within each episode
        - ``hot_day_mask`` — boolean mask same length as input
        - ``event_start_idx`` — start index of each episode

    Examples
    --------
    A series with a clear 5-day spike:

    >>> import numpy as np
    >>> T = np.concatenate([
    ...     np.full(100, 25.0),    # baseline
    ...     np.array([32, 34, 33, 35, 32]),  # heat wave
    ...     np.full(100, 26.0),    # return to normal
    ... ])
    >>> r = heat_wave_detect(T, percentile=90, min_consecutive_days=3)
    >>> r.extra["n_events"]
    1
    >>> r.extra["event_lengths"]
    [5]

    References
    ----------
    Perkins, S. E., & Alexander, L. V. (2013). On the measurement of
    heat waves. Journal of Climate, 26(13), 4500-4517.

    World Meteorological Organization / WHO (2015). Heatwaves and
    Health: Guidance on Warning-System Development. WMO-No. 1142.

    Notes
    -----
    Quote: "The heat is rising."
    """
    T = np.asarray(daily_max_C, dtype=float)
    if T.ndim != 1:
        raise ValueError("daily_max_C must be 1-D (a day-indexed series).")
    if T.size < min_consecutive_days:
        raise ValueError(
            f"Series too short ({T.size}) for min_consecutive_days="
            f"{min_consecutive_days}."
        )
    if not (0 < percentile < 100):
        raise ValueError("percentile must be in (0, 100).")
    if np.any(~np.isfinite(T)):
        raise ValueError("daily_max_C has NaN or inf — fill gaps first.")

    # Threshold from baseline window
    if baseline_window is None:
        baseline = T
    else:
        s, e = baseline_window
        baseline = T[s:e]
        if baseline.size == 0:
            raise ValueError("baseline_window yields empty slice.")
    threshold = float(np.percentile(baseline, percentile))

    # Hot-day mask
    hot = threshold < T

    # Find runs of True ≥ min_consecutive_days
    event_lengths: list[int] = []
    event_peaks: list[float] = []
    event_starts: list[int] = []
    i = 0
    n = len(hot)
    while i < n:
        if hot[i]:
            j = i
            while j < n and hot[j]:
                j += 1
            run_len = j - i
            if run_len >= min_consecutive_days:
                event_lengths.append(run_len)
                event_peaks.append(float(T[i:j].max()))
                event_starts.append(i)
            i = j
        else:
            i += 1

    hw_days = sum(event_lengths)

    return DescriptiveResult(
        name="heat_wave_detect",
        value=float(hw_days),
        extra={
            "threshold_C": threshold,
            "percentile": percentile,
            "min_consecutive_days": min_consecutive_days,
            "n_events": len(event_lengths),
            "event_lengths": event_lengths,
            "event_peaks_C": event_peaks,
            "event_start_idx": event_starts,
            "hot_day_mask": hot.tolist(),
            "total_heat_wave_days": hw_days,
        },
    )


heatwv = heat_wave_detect


def cheatsheet() -> str:
    return "heatwv(Tmax, percentile=90, min_days=3) -> heat-wave episode stats."
