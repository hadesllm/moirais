#' Pan-Tompkins QRS detector -- Rangayyan Ch 6
#'
#' Pan & Tompkins (1985) QRS detector: bandpass 5-15 Hz -> differentiate ->
#' square -> 150-ms moving-window integration -> adaptive threshold (30 %
#' of integrated max, 200-ms refractory).
#'
#' @param x Numeric ECG vector.
#' @param fs Sampling rate (Hz, default 360 -- MIT-BIH).
#' @return Named list `r_peaks` (sample indices), `rr_intervals_ms`,
#'   `heart_rate_bpm`, `integrated`, `fs`.
#' @references Pan & Tompkins (1985), IEEE TBME 32:230. Rangayyan Ch 6.
#' @export
#' @examples
#' \donttest{
#'   if (requireNamespace("signal", quietly = TRUE)) {
#'     fs <- 360; t <- seq(0, 5, length.out = 5 * fs)
#'     ecg <- rowSums(sapply(seq(0.5, 4.5, by = 1.0),
#'                           function(tk) exp(-((t - tk) * 30)^2)))
#'     rgqrs(ecg, fs = fs)$r_peaks
#'   }
#' }
rgqrs <- function(x, fs = 360.0) {
  if (!requireNamespace("signal", quietly = TRUE)) {
    stop("Package 'signal' is required for rgqrs().")
  }
  nyq <- 0.5 * fs
  bf <- signal::butter(3L, c(5, min(15, nyq * 0.95)) / nyq, type = "pass")
  bp <- as.numeric(signal::filtfilt(bf, x))
  der <- numeric(length(bp))
  for (n in 5:length(bp)) {
    der[n] <- (1 / 8) * (2 * bp[n] + bp[n - 1] - bp[n - 3] - 2 * bp[n - 4])
  }
  sq <- der^2
  W <- max(1L, as.integer(round(0.150 * fs)))
  k <- rep(1 / W, W)
  integ <- as.numeric(stats::filter(sq, k, sides = 2))
  integ[is.na(integ)] <- 0
  refractory <- as.integer(round(0.200 * fs))
  thr <- 0.30 * max(integ, na.rm = TRUE)
  # naive peak finder with refractory
  peaks <- integer(0)
  i <- 2L
  N <- length(integ)
  while (i < N) {
    if (integ[i] > thr && integ[i] > integ[i - 1] && integ[i] >= integ[i + 1]) {
      peaks <- c(peaks, i)
      i <- i + refractory
    } else {
      i <- i + 1L
    }
  }
  # refine each to local |bp| max within +/-50 ms
  half <- as.integer(round(0.05 * fs))
  refined <- vapply(peaks, function(p) {
    lo <- max(1L, p - half); hi <- min(N, p + half)
    lo + which.max(abs(bp[lo:hi])) - 1L
  }, integer(1))
  rr_ms <- if (length(refined) > 1) diff(refined) * 1000 / fs else numeric(0)
  hr <- if (length(rr_ms)) 60000 / mean(rr_ms) else NA_real_
  list(r_peaks = refined, rr_intervals_ms = rr_ms,
       heart_rate_bpm = hr, integrated = integ, fs = fs)
}

#' @rdname rgqrs
#' @keywords internal
#' @export
rangayyan_qrs_detect <- rgqrs
