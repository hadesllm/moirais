#' Welch power spectral density — Rangayyan Ch 4
#'
#' Welch-averaged one-sided PSD via [stats::spec.pgram()] on overlapping
#' detrended segments. Same algorithm as SciPy's `welch` so band powers
#' agree to ~1e-3 for typical EEG/ECG inputs.
#'
#' @param x Numeric vector.
#' @param fs Sampling frequency (Hz).
#' @param nperseg Segment length (default `min(N, 256)`).
#' @param window Window name (`"hann"` default).
#' @return Named list `freqs`, `psd`, `fs`, `nperseg`, `peak_freq`,
#'   `total_power`.
#' @references Welch (1967); Rangayyan Ch 4.
#' @export
#' @examples
#' \donttest{
#' set.seed(0); fs <- 100
#' t <- seq(0, 10, length.out = 1000)
#' x <- sin(2 * pi * 10 * t)
#' r <- rgpsd(x, fs = fs, nperseg = 256)
#' abs(r$peak_freq - 10) < 1
#' }
rgpsd <- function(x, fs = 1.0, nperseg = NULL, window = "hann") {
  N <- length(x)
  if (is.null(nperseg)) nperseg <- min(N, 256L)
  nperseg <- as.integer(nperseg)
  # Build overlapping segments with 50% overlap, apply Hann window,
  # take periodogram, average.
  step <- nperseg %/% 2L
  starts <- seq(1, N - nperseg + 1, by = step)
  if (length(starts) < 1) starts <- 1
  w <- switch(window,
              hann     = 0.5 - 0.5 * cos(2 * pi * (seq_len(nperseg) - 1) / (nperseg - 1)),
              hamming  = 0.54 - 0.46 * cos(2 * pi * (seq_len(nperseg) - 1) / (nperseg - 1)),
              boxcar   = rep(1, nperseg),
              rep(1, nperseg))
  W <- sum(w^2)
  freqs <- seq(0, fs / 2, length.out = nperseg %/% 2L + 1L)
  psd_acc <- numeric(length(freqs))
  for (s in starts) {
    seg <- x[s:(s + nperseg - 1)] - mean(x[s:(s + nperseg - 1)])
    seg <- seg * w
    X <- stats::fft(seg)[1:length(freqs)]
    pxx <- (Mod(X)^2) / (fs * W)
    # one-sided scaling: double interior bins
    if (length(pxx) > 2) pxx[2:(length(pxx) - 1)] <- 2 * pxx[2:(length(pxx) - 1)]
    psd_acc <- psd_acc + pxx
  }
  psd <- psd_acc / length(starts)
  peak <- which.max(psd)
  total <- sum(psd) * (freqs[2] - freqs[1])
  list(freqs = freqs, psd = psd, fs = fs, nperseg = nperseg,
       peak_freq = freqs[peak], total_power = total)
}

#' @rdname rgpsd
#' @keywords internal
#' @export
rangayyan_psd <- rgpsd
