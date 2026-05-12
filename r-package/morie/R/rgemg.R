#' EMG sliding-window RMS envelope — Rangayyan Ch 8
#'
#' \deqn{\mathrm{RMS}[n] = \sqrt{ (1/W) \sum_{k=n-W+1}^{n} x[k]^2 }}
#'
#' @param x Numeric vector.
#' @param window Window length in samples (default 64).
#' @param fs Sampling rate (Hz, for reporting only).
#' @return Named list `rms`, `window`, `fs`, `mean_rms`.
#' @references Rangayyan Ch 8.
#' @export
#' @examples
#' set.seed(0); r <- rgemg(rnorm(500), window = 32); length(r$rms)
rgemg <- function(x, window = 64L, fs = 1.0) {
  W <- as.integer(window)
  if (W < 1) stop("window must be >= 1")
  N <- length(x)
  sq <- x^2
  csum <- c(0, cumsum(sq))
  rms <- rep(NA_real_, N)
  for (i in W:N) rms[i] <- sqrt((csum[i + 1] - csum[i + 1 - W]) / W)
  if (N >= W) rms[seq_len(W - 1L)] <- rms[W]
  list(rms = rms, window = W, fs = fs, mean_rms = mean(rms, na.rm = TRUE))
}

#' @rdname rgemg
#' @keywords internal
#' @export
rangayyan_emg_rms <- rgemg
