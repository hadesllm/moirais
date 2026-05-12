#' Butterworth IIR filter — Rangayyan Ch 3
#'
#' Zero-phase Butterworth IIR filter via the `signal` package (`butter` +
#' `filtfilt`). Same transfer function as SciPy's `butter` so the Python
#' and R outputs match to machine precision for identical inputs.
#'
#' @param x Numeric vector.
#' @param cutoff Cutoff frequency (Hz). For `"pass"`/`"stop"`, a length-2
#'   numeric vector `c(low, high)`.
#' @param order Filter order (default 4).
#' @param fs Sampling rate (Hz).
#' @param btype One of `"low"`, `"high"`, `"pass"` (bandpass), `"stop"`
#'   (bandstop).
#' @return Named list `signal`, `order`, `cutoff`, `fs`, `btype`.
#' @references Rangayyan Ch 3.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   t <- seq(0, 1, length.out = 500)
#'   x <- sin(2 * pi * 5 * t) + 0.5 * sin(2 * pi * 40 * t)
#'   r <- rgiir(x, cutoff = 10, order = 4, fs = 500, btype = "low")
#'   length(r$signal)
#' }
#' }
rgiir <- function(x, cutoff, order = 4L, fs = 1.0, btype = c("low", "high", "pass", "stop")) {
  btype <- match.arg(btype)
  if (!requireNamespace("signal", quietly = TRUE)) {
    stop("R package 'signal' is required for rgiir().")
  }
  nyq <- 0.5 * fs
  wn  <- cutoff / nyq
  bf <- signal::butter(as.integer(order), wn, type = btype)
  y  <- as.numeric(signal::filtfilt(bf, x))
  list(signal = y, order = as.integer(order), cutoff = cutoff,
       fs = fs, btype = btype)
}

#' @rdname rgiir
#' @keywords internal
#' @export
rangayyan_iir_filter <- rgiir
