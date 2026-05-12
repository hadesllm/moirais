# SPDX-License-Identifier: GPL-2.0-only

#' Magnitude-squared coherence between two time series
#'
#' @param x Numeric vector.
#' @param y Numeric vector (same length).
#' @param nperseg Segment length. Default n/4.
#' @param fs Sampling frequency. Default 1.
#' @return Named list with \code{frequencies, coherence, n_segments,
#'   nperseg, fs, n, method}.
#' @export
coherence <- function(x, y, nperseg = NULL, fs = 1) {
  x <- as.numeric(x); y <- as.numeric(y)
  if (length(x) != length(y)) stop("Length mismatch.")
  n <- length(x); if (n < 8) stop("Need >=8 obs.")
  if (is.null(nperseg)) nperseg <- max(n %/% 4, 4)
  nperseg <- min(nperseg, n)
  step <- nperseg %/% 2
  nfreq <- nperseg %/% 2 + 1
  Sxx <- numeric(nfreq); Syy <- numeric(nfreq)
  Sxy <- complex(nfreq); nseg <- 0; start <- 1
  while (start + nperseg - 1 <= n) {
    xs <- x[start:(start + nperseg - 1)] - mean(x[start:(start + nperseg - 1)])
    ys <- y[start:(start + nperseg - 1)] - mean(y[start:(start + nperseg - 1)])
    fx <- fft(xs)[1:nfreq]; fy <- fft(ys)[1:nfreq]
    Sxx <- Sxx + Mod(fx)^2
    Syy <- Syy + Mod(fy)^2
    Sxy <- Sxy + fx * Conj(fy)
    nseg <- nseg + 1
    start <- start + step
  }
  Sxx <- Sxx / nseg; Syy <- Syy / nseg; Sxy <- Sxy / nseg
  denom <- pmax(Sxx * Syy, 1e-15)
  coh <- Mod(Sxy)^2 / denom
  freqs <- seq(0, fs / 2, length.out = nfreq)
  list(frequencies = freqs, coherence = coh,
       n_segments = nseg, nperseg = nperseg,
       fs = fs, n = n,
       method = "Magnitude-squared coherence (Welch, base R)")
}
