#' Short-time Fourier transform -- Rangayyan Ch 4
#'
#' Sliding-window FFT spectrogram (one-sided, PSD-scaled). Mirrors
#' `scipy.signal.spectrogram` with `scaling="density"`.
#'
#' @param x Numeric vector.
#' @param fs Sampling rate (Hz).
#' @param nperseg Window length in samples (default 256).
#' @param noverlap Overlap in samples (default `nperseg/2`).
#' @param window `"hann"` (default), `"hamming"`, `"boxcar"`.
#' @return Named list `freqs`, `times`, `Sxx` (rows=freqs, cols=times),
#'   `nperseg`, `noverlap`, `fs`.
#' @references Rangayyan Ch 4.
#' @export
#' @examples
#' \donttest{
#'   t <- seq(0, 10, length.out = 1024); x <- sin(2 * pi * 10 * t)
#'   r <- rgstf(x, fs = 100, nperseg = 128); dim(r$Sxx)
#' }
rgstf <- function(x, fs = 1.0, nperseg = 256L, noverlap = NULL,
                  window = "hann") {
  nperseg <- as.integer(min(nperseg, length(x)))
  if (is.null(noverlap)) noverlap <- nperseg %/% 2L
  step <- nperseg - as.integer(noverlap)
  starts <- seq(1, length(x) - nperseg + 1, by = step)
  if (length(starts) < 1) starts <- 1
  w <- switch(window,
              hann    = 0.5 - 0.5 * cos(2 * pi * (seq_len(nperseg) - 1) / (nperseg - 1)),
              hamming = 0.54 - 0.46 * cos(2 * pi * (seq_len(nperseg) - 1) / (nperseg - 1)),
              boxcar  = rep(1, nperseg),
              rep(1, nperseg))
  W <- sum(w^2)
  nfreq <- nperseg %/% 2L + 1L
  freqs <- seq(0, fs / 2, length.out = nfreq)
  Sxx <- matrix(0, nrow = nfreq, ncol = length(starts))
  times <- numeric(length(starts))
  for (i in seq_along(starts)) {
    s <- starts[i]
    seg <- (x[s:(s + nperseg - 1)] - mean(x[s:(s + nperseg - 1)])) * w
    X <- stats::fft(seg)[1:nfreq]
    pxx <- (Mod(X)^2) / (fs * W)
    if (length(pxx) > 2) pxx[2:(length(pxx) - 1)] <- 2 * pxx[2:(length(pxx) - 1)]
    Sxx[, i] <- pxx
    times[i] <- (s + nperseg / 2 - 1) / fs
  }
  list(freqs = freqs, times = times, Sxx = Sxx,
       nperseg = nperseg, noverlap = as.integer(noverlap), fs = fs)
}

#' @rdname rgstf
#' @keywords internal
#' @export
rangayyan_stft <- rgstf
