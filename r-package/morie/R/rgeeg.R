#' EEG band power (δ θ α β γ) via Welch — Rangayyan Ch 9
#'
#' Absolute and relative power in the clinical EEG bands using the
#' Welch PSD ([rgpsd()]).
#'
#' Default bands: delta 0.5–4, theta 4–8, alpha 8–13, beta 13–30,
#' gamma 30–100 Hz.
#'
#' @param x Numeric EEG vector.
#' @param fs Sampling rate (Hz).
#' @param bands Optional named list of `c(lo, hi)` band edges.
#' @param nperseg Welch segment length.
#' @return Named list `absolute` (named numeric), `relative` (named
#'   numeric, sum ≤ 1), `total_power`, `freqs`, `psd`.
#' @references Rangayyan Ch 9.
#' @export
#' @examples
#' \donttest{
#'   set.seed(0); fs <- 256
#'   t <- seq(0, 8, length.out = 2048)
#'   x <- sin(2 * pi * 10 * t) + 0.3 * rnorm(length(t))
#'   r <- rgeeg(x, fs = fs); r$relative[["alpha"]] > r$relative[["gamma"]]
#' }
rgeeg <- function(x, fs, bands = NULL, nperseg = NULL) {
  if (is.null(bands)) {
    bands <- list(delta = c(0.5, 4), theta = c(4, 8),
                  alpha = c(8, 13), beta = c(13, 30),
                  gamma = c(30, 100))
  }
  if (is.null(nperseg)) nperseg <- max(16, min(length(x), as.integer(4 * fs)))
  ps <- rgpsd(x, fs = fs, nperseg = nperseg)
  freqs <- ps$freqs; psd <- ps$psd
  df <- diff(freqs[1:2])
  total <- sum(psd) * df
  absolute <- vapply(names(bands), function(nm) {
    rng <- bands[[nm]]
    mask <- freqs >= rng[1] & freqs < rng[2]
    if (any(mask)) sum(psd[mask]) * df else 0
  }, numeric(1))
  names(absolute) <- names(bands)
  relative <- if (total > 0) absolute / total else absolute * 0
  list(absolute = absolute, relative = relative,
       total_power = total, freqs = freqs, psd = psd)
}

#' @rdname rgeeg
#' @keywords internal
#' @export
rangayyan_eeg_bands <- rgeeg
