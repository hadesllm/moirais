#' Butterworth lowpass filter
#'
#' Zero-phase Butterworth lowpass filter via the suggested `signal` package's
#' `butter()` + `filtfilt()`. Useful for removing high-frequency noise from
#' biological or geophysical time series.
#'
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param cutoff Cutoff frequency (Hz).
#' @param order Filter order (default 4).
#' @return List with `filtered` (numeric vector), `fs`, `order`, `name`.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   set.seed(1)
#'   t  <- seq(0, 1, length.out = 500)
#'   x  <- sin(2 * pi * 5 * t) + 0.5 * sin(2 * pi * 60 * t)  # 5 Hz + 60 Hz
#'   y  <- buttlp(x, fs = 500, cutoff = 20)
#'   length(y$filtered)  # 500
#' }
#' }
buttlp <- function(x, fs, cutoff, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, cutoff / (fs / 2), type = "low")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_lowpass"))
  }
  .morie_py_call("buttlp", x, fs, cutoff, order)
}

#' Butterworth highpass filter
#'
#' Zero-phase Butterworth highpass filter. Removes low-frequency drift while
#' preserving higher-frequency content; useful for de-trending physiological
#' signals (EEG, ECG) prior to analysis.
#'
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param cutoff Cutoff frequency (Hz).
#' @param order Filter order (default 4).
#' @return List with `filtered` (numeric vector), `fs`, `order`, `name`.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   set.seed(1)
#'   t <- seq(0, 1, length.out = 500)
#'   x <- 5 * t + sin(2 * pi * 10 * t)   # linear drift + 10 Hz signal
#'   y <- butthp(x, fs = 500, cutoff = 1)
#'   length(y$filtered)
#' }
#' }
butthp <- function(x, fs, cutoff, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, cutoff / (fs / 2), type = "high")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_highpass"))
  }
  .morie_py_call("butthp", x, fs, cutoff, order)
}

#' Butterworth bandpass filter
#'
#' Zero-phase Butterworth bandpass filter. Isolates a frequency band of
#' interest (e.g., 0.5--40 Hz for EEG, 25--400 Hz for phonocardiogram).
#'
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param low Lower cutoff (Hz).
#' @param high Upper cutoff (Hz).
#' @param order Filter order (default 4).
#' @return List with `filtered` (numeric vector), `fs`, `order`, `name`.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   set.seed(1)
#'   t <- seq(0, 1, length.out = 1000)
#'   # 2 Hz drift + 10 Hz band of interest + 60 Hz noise
#'   x <- sin(2 * pi * 2 * t) + sin(2 * pi * 10 * t) +
#'        0.3 * sin(2 * pi * 60 * t)
#'   y <- buttbp(x, fs = 1000, low = 5, high = 20)
#'   length(y$filtered)
#' }
#' }
buttbp <- function(x, fs, low, high, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, c(low, high) / (fs / 2), type = "pass")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_bandpass"))
  }
  .morie_py_call("buttbp", x, fs, low, high, order)
}

#' Butterworth bandstop (notch) filter
#'
#' Zero-phase Butterworth bandstop filter. Default 59--61 Hz removes North-
#' American AC mains hum (60 Hz); use 49--51 Hz for European mains.
#'
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param low Lower cutoff (Hz, default 59).
#' @param high Upper cutoff (Hz, default 61).
#' @param order Filter order (default 4).
#' @return List with `filtered` (numeric vector), `fs`, `order`, `name`.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   set.seed(1)
#'   t <- seq(0, 1, length.out = 1000)
#'   x <- sin(2 * pi * 10 * t) + sin(2 * pi * 60 * t)
#'   y <- buttbs(x, fs = 1000)  # remove 60 Hz mains
#'   length(y$filtered)
#' }
#' }
buttbs <- function(x, fs, low = 59, high = 61, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, c(low, high) / (fs / 2), type = "stop")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_bandstop"))
  }
  .morie_py_call("buttbs", x, fs, low, high, order)
}

#' Savitzky-Golay smoothing filter
#'
#' Polynomial-fit smoothing filter. Preserves higher moments (peak heights,
#' shape) better than a moving average and is the standard tool for
#' chromatography, spectroscopy, and biosensor smoothing.
#'
#' @param x Numeric vector.
#' @param window_length Window length (odd integer, default 11).
#' @param polyorder Polynomial order (default 3).
#' @return List with `filtered` (numeric vector) and `name`.
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   set.seed(1)
#'   t <- seq(0, 1, length.out = 200)
#'   x <- sin(2 * pi * 3 * t) + rnorm(200, sd = 0.2)
#'   y <- sgolay_smooth(x, window_length = 11, polyorder = 3)
#'   length(y$filtered)
#' }
#' }
sgolay_smooth <- function(x, window_length = 11L, polyorder = 3L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    sg <- signal::sgolay(p = polyorder, n = window_length)
    y <- signal::filter(sg, x)
    return(list(filtered = as.numeric(y), name = "savitzky_golay"))
  }
  .morie_py_call("sgolay", x, window_length, polyorder)
}

#' Hurst exponent via rescaled-range (R/S) analysis
#'
#' Estimates the Hurst exponent \eqn{H}: a long-memory measure of a time
#' series. \eqn{H = 0.5} indicates uncorrelated (Brownian) increments;
#' \eqn{H > 0.5} indicates persistent (trending) behaviour;
#' \eqn{H < 0.5} indicates anti-persistent (mean-reverting) behaviour.
#'
#' @param x Numeric vector (time series).
#' @return List with `H` (numeric) and `interpretation`
#'   (`"persistent"`/`"anti-persistent"`/`"random"`).
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("pracma", quietly = TRUE)) {
#'   set.seed(1)
#'   x <- cumsum(rnorm(2048))   # Brownian motion, expected H ~ 0.5
#'   res <- hurst_r(x)
#'   res$interpretation
#' }
#' }
hurst_r <- function(x) {
  if (requireNamespace("pracma", quietly = TRUE)) {
    result <- pracma::hurstexp(x, display = FALSE)
    return(list(H = result$Hs, interpretation = ifelse(result$Hs > 0.55, "persistent",
                                                        ifelse(result$Hs < 0.45, "anti-persistent", "random"))))
  }
  .morie_py_call("hurst", x)
}

#' Higuchi fractal dimension
#'
#' Estimates the Higuchi (1988) fractal dimension of a 1-D time series via
#' length scaling across `k` time-lags. Used in EEG and biological-signal
#' complexity analysis. Values typically fall in \[1, 2\]; higher values
#' indicate greater signal complexity.
#'
#' @param x Numeric vector.
#' @param kmax Maximum k (default 10).
#' @return List with the fractal-dimension value.
#' @export
#' @examples
#' \donttest{
#' set.seed(1)
#' x <- cumsum(rnorm(1000))
#' hfd(x, kmax = 10)
#' }
hfd <- function(x, kmax = 10L) {
  .morie_py_call("hfd", x, kmax)
}

#' Phonocardiogram (PCG) bandpass filter
#'
#' Convenience preset wrapping [buttbp()] with the standard PCG band
#' (25--400 Hz at 2000 Hz sampling). Removes baseline drift below 25 Hz and
#' anti-aliased high-frequency noise above 400 Hz.
#'
#' @param x Numeric vector (PCG signal).
#' @param fs Sampling frequency (Hz, default 2000).
#' @param low Lower cutoff (Hz, default 25).
#' @param high Upper cutoff (Hz, default 400).
#' @return List with filtered signal (see [buttbp()]).
#' @export
#' @examples
#' \donttest{
#' if (requireNamespace("signal", quietly = TRUE)) {
#'   set.seed(1)
#'   x <- rnorm(2000)            # 1 second of white-noise PCG-like input
#'   y <- pcg_filter(x)
#'   length(y$filtered)
#' }
#' }
pcg_filter <- function(x, fs = 2000, low = 25, high = 400) {
  buttbp(x, fs, low, high)
}

.morie_py_call <- function(fn_name, ...) {
  args <- list(...)
  arg_str <- paste(sapply(args, function(a) {
    if (is.numeric(a) && length(a) > 1) {
      paste0("[", paste(a, collapse = ","), "]")
    } else {
      as.character(a)
    }
  }), collapse = " ")
  cmd <- paste(fn_name, arg_str)
  out <- system2("python3", c("-m", "morie.stat_bridge", "exec", cmd), stdout = TRUE, stderr = TRUE)
  paste(out, collapse = "\n")
}
