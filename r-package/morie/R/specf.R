# SPDX-License-Identifier: MIT OR Apache-2.0

#' Welch power spectral density
#'
#' @param x Numeric univariate series.
#' @param fs Sampling frequency. Default 1.
#' @param nperseg Segment length. Default max(n/4, 8).
#' @return Named list with \code{frequencies, psd, n_segments, nperseg,
#'   fs, n, method}.
#' @export
spectral_density <- function(x, fs = 1, nperseg = NULL) {
  r <- as.numeric(x); n <- length(r)
  if (n < 8) stop("Need >=8 obs.")
  if (is.null(nperseg)) nperseg <- max(n %/% 4, 8)
  nperseg <- min(nperseg, n)
  step <- max(nperseg %/% 2, 1)
  win <- 0.5 - 0.5 * cos(2 * pi * (0:(nperseg - 1)) / max(nperseg - 1, 1))
  U <- sum(win^2)
  nfreq <- nperseg %/% 2 + 1
  S <- numeric(nfreq); nseg <- 0; start <- 1
  while (start + nperseg - 1 <= n) {
    seg <- (r[start:(start + nperseg - 1)] -
            mean(r[start:(start + nperseg - 1)])) * win
    Fk <- fft(seg)[1:nfreq]
    S <- S + Mod(Fk)^2
    nseg <- nseg + 1
    start <- start + step
  }
  S <- S / (nseg * U * fs)
  freqs <- seq(0, fs / 2, length.out = nfreq)
  list(frequencies = freqs, psd = S, n_segments = nseg,
       nperseg = nperseg, fs = fs, n = n,
       method = "Welch PSD (Hann window, 50% overlap, base R)")
}
