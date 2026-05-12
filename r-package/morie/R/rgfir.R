#' FIR lowpass filter (windowed sinc) — Rangayyan Ch 3
#'
#' Designs a linear-phase windowed-sinc FIR lowpass filter and applies it
#' zero-phase via [signal::filtfilt()] (or single-pass [signal::filter()]
#' when the signal is too short to allow forward-backward padding).
#'
#' \deqn{h[n] = w[n] \, 2 f_c \, \mathrm{sinc}(2 f_c (n - M/2))}
#'
#' @param x Numeric vector. Input signal.
#' @param cutoff Cutoff frequency in Hz (if `fs` given) or normalised to
#'   Nyquist=0.5.
#' @param order Number of taps (odd recommended; coerced to odd).
#' @param fs Sampling frequency (Hz). Default 1.
#' @param window Window name: `"hamming"`, `"hann"`, `"blackman"`, `"rectangular"`.
#' @return Named list `signal`, `taps`, `order`, `cutoff`, `fs`, `window`.
#' @references Rangayyan, R.M. *Biomedical Signal Analysis* (Wiley/IEEE, 2015), Ch. 3.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   t <- seq(0, 1, length.out = 200)
#'   x <- sin(2 * pi * 5 * t) + 0.5 * sin(2 * pi * 60 * t)
#'   r <- rgfir(x, cutoff = 10, order = 51, fs = 200)
#'   length(r$signal)
#' }
#' }
rgfir <- function(x, cutoff, order = 51L, fs = 1.0, window = "hamming") {
  order <- as.integer(order)
  if (order < 3L) order <- 3L
  if (order %% 2L == 0L) order <- order + 1L
  nyq <- 0.5 * fs
  fc  <- min(max(cutoff / nyq, 1e-6), 1 - 1e-6)
  if (requireNamespace("signal", quietly = TRUE)) {
    win_fn <- switch(window,
      hamming     = signal::hamming(order),
      hann        = signal::hanning(order),
      blackman    = signal::blackman(order),
      rectangular = rep(1, order),
      signal::hamming(order))
    taps <- signal::fir1(order - 1L, fc, type = "low", window = win_fn)
    padlen <- 3L * order
    if (length(x) > padlen) {
      y <- as.numeric(signal::filtfilt(taps, 1, x))
    } else {
      y <- as.numeric(signal::filter(taps, 1, x))
    }
    return(list(signal = y, taps = taps, order = order,
                cutoff = cutoff, fs = fs, window = window))
  }
  stop("R package 'signal' is required for rgfir().")
}

#' @rdname rgfir
#' @keywords internal
#' @export
rangayyan_fir_filter <- rgfir
