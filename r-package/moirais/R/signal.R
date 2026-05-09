#' Butterworth lowpass filter
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param cutoff Cutoff frequency (Hz).
#' @param order Filter order (default 4).
#' @return List with filtered signal and metadata.
#' @export
buttlp <- function(x, fs, cutoff, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, cutoff / (fs / 2), type = "low")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_lowpass"))
  }
  .moirais_py_call("buttlp", x, fs, cutoff, order)
}

#' Butterworth highpass filter
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param cutoff Cutoff frequency (Hz).
#' @param order Filter order (default 4).
#' @return List with filtered signal and metadata.
#' @export
butthp <- function(x, fs, cutoff, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, cutoff / (fs / 2), type = "high")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_highpass"))
  }
  .moirais_py_call("butthp", x, fs, cutoff, order)
}

#' Butterworth bandpass filter
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param low Lower cutoff (Hz).
#' @param high Upper cutoff (Hz).
#' @param order Filter order (default 4).
#' @return List with filtered signal and metadata.
#' @export
buttbp <- function(x, fs, low, high, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, c(low, high) / (fs / 2), type = "pass")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_bandpass"))
  }
  .moirais_py_call("buttbp", x, fs, low, high, order)
}

#' Butterworth bandstop (notch) filter
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param low Lower cutoff (Hz, default 59).
#' @param high Upper cutoff (Hz, default 61).
#' @param order Filter order (default 4).
#' @return List with filtered signal and metadata.
#' @export
buttbs <- function(x, fs, low = 59, high = 61, order = 4L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    bf <- signal::butter(order, c(low, high) / (fs / 2), type = "stop")
    y <- signal::filtfilt(bf, x)
    return(list(filtered = y, fs = fs, order = order, name = "butter_bandstop"))
  }
  .moirais_py_call("buttbs", x, fs, low, high, order)
}

#' Savitzky-Golay smoothing filter
#' @param x Numeric vector.
#' @param window_length Window length (odd integer, default 11).
#' @param polyorder Polynomial order (default 3).
#' @return List with filtered signal.
#' @export
sgolay_smooth <- function(x, window_length = 11L, polyorder = 3L) {
  if (requireNamespace("signal", quietly = TRUE)) {
    sg <- signal::sgolay(p = polyorder, n = window_length)
    y <- signal::filter(sg, x)
    return(list(filtered = as.numeric(y), name = "savitzky_golay"))
  }
  .moirais_py_call("sgolay", x, window_length, polyorder)
}

#' Hurst exponent via R/S analysis
#' @param x Numeric vector (time series).
#' @return List with H exponent and interpretation.
#' @export
hurst_r <- function(x) {
  if (requireNamespace("pracma", quietly = TRUE)) {
    result <- pracma::hurstexp(x, display = FALSE)
    return(list(H = result$Hs, interpretation = ifelse(result$Hs > 0.55, "persistent",
                                                        ifelse(result$Hs < 0.45, "anti-persistent", "random"))))
  }
  .moirais_py_call("hurst", x)
}

#' Higuchi fractal dimension
#' @param x Numeric vector.
#' @param kmax Maximum k (default 10).
#' @return List with fractal dimension value.
#' @export
hfd <- function(x, kmax = 10L) {
  .moirais_py_call("hfd", x, kmax)
}

#' PCG bandpass filter
#' @param x Numeric vector (PCG signal).
#' @param fs Sampling frequency (Hz, default 2000).
#' @param low Lower cutoff (Hz, default 25).
#' @param high Upper cutoff (Hz, default 400).
#' @return List with filtered signal.
#' @export
pcg_filter <- function(x, fs = 2000, low = 25, high = 400) {
  buttbp(x, fs, low, high)
}

.moirais_py_call <- function(fn_name, ...) {
  args <- list(...)
  arg_str <- paste(sapply(args, function(a) {
    if (is.numeric(a) && length(a) > 1) {
      paste0("[", paste(a, collapse = ","), "]")
    } else {
      as.character(a)
    }
  }), collapse = " ")
  cmd <- paste(fn_name, arg_str)
  out <- system2("python3", c("-m", "moirais.stat_bridge", "exec", cmd), stdout = TRUE, stderr = TRUE)
  paste(out, collapse = "\n")
}
