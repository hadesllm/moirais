#' Wavelet denoising (universal threshold) — Rangayyan Ch 10
#'
#' Donoho-Johnstone soft/hard thresholding via the `wavelets` package
#' (Daubechies DWT). Noise scale estimated from the finest-scale detail
#' as σ = MAD / 0.6745; universal threshold T = σ sqrt(2 ln N).
#'
#' Falls back to a 5-tap moving-average smoother (with a warning) if
#' `wavelets` is not installed — keeps R↔Py parity to ~5 % on smooth
#' signals.
#'
#' @param x Numeric vector.
#' @param wavelet Wavelet filter name (default `"d8"` ≈ db4).
#' @param level Decomposition depth (default `min(4, max_level)`).
#' @param mode `"soft"` or `"hard"`.
#' @return Named list `signal`, `threshold`, `sigma`, `wavelet`,
#'   `level`, `mode`.
#' @references Donoho & Johnstone (1994); Rangayyan Ch 10.
#' @export
#' @examples
#' \donttest{
#'   set.seed(0); t <- seq(0, 1, length.out = 256)
#'   x <- sin(2 * pi * 3 * t) + 0.3 * rnorm(256)
#'   if (requireNamespace("wavelets", quietly = TRUE)) {
#'     r <- rgwav(x, level = 3); length(r$signal)
#'   }
#' }
rgwav <- function(x, wavelet = "d8", level = NULL, mode = c("soft", "hard")) {
  mode <- match.arg(mode)
  if (!requireNamespace("wavelets", quietly = TRUE)) {
    warning("'wavelets' not available; using 5-tap MA fallback.")
    k <- rep(1 / 5, 5)
    y <- as.numeric(stats::filter(x, k, sides = 2))
    y[is.na(y)] <- x[is.na(y)]
    return(list(signal = y, threshold = NA_real_, sigma = NA_real_,
                wavelet = wavelet, level = 0L, mode = "MA-fallback"))
  }
  # Pad to next power of two (wavelets::dwt requires it)
  N0 <- length(x)
  N  <- 2^ceiling(log2(N0))
  xp <- c(x, rep(0, N - N0))
  if (is.null(level)) {
    level <- min(4L, as.integer(log2(N)) - 1L)
  }
  level <- as.integer(level)
  fit <- wavelets::dwt(xp, filter = wavelet, n.levels = level)
  # MAD on the finest detail coefficients (W[[1]])
  d1 <- as.numeric(fit@W[[1]])
  sigma <- stats::median(abs(d1)) / 0.6745
  T <- sigma * sqrt(2 * log(N))
  thresh <- function(d, T, mode) {
    if (mode == "soft") sign(d) * pmax(abs(d) - T, 0)
    else d * (abs(d) > T)
  }
  for (i in seq_along(fit@W)) {
    fit@W[[i]][, 1] <- thresh(as.numeric(fit@W[[i]]), T, mode)
  }
  y <- as.numeric(wavelets::idwt(fit))[seq_len(N0)]
  list(signal = y, threshold = T, sigma = sigma,
       wavelet = wavelet, level = level, mode = mode)
}

#' @rdname rgwav
#' @keywords internal
#' @export
rangayyan_wavelet_denoise <- rgwav
