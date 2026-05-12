#' LMS adaptive noise canceller -- Rangayyan Ch 11
#'
#' Widrow-Hoff least-mean-squares adaptive filter. Returns the cleaned
#' residual `signal = x - LMS(reference)` and the noise estimate.
#'
#' @param x Numeric vector (target + correlated noise).
#' @param reference Numeric vector of reference noise (same length).
#' @param mu Step size (0 < mu).
#' @param order Number of taps (default 16).
#' @return Named list `signal`, `noise_estimate`, `weights`, `mu`, `order`.
#' @references Widrow & Stearns (1985); Rangayyan Ch 11.
#' @export
#' @examples
#' set.seed(0); n <- rnorm(200)
#' x <- sin(2 * pi * seq_len(200) / 20) + n
#' r <- rgadp(x, reference = n, mu = 0.01, order = 8); length(r$signal)
rgadp <- function(x, reference, mu = 0.01, order = 16L) {
  if (length(x) != length(reference)) stop("x and reference must have equal length.")
  M <- as.integer(order); N <- length(x)
  w <- numeric(M); y <- numeric(N); e <- numeric(N)
  for (n in M:N) {
    rv <- reference[(n - M + 1):n][seq.int(M, 1L)]
    y[n] <- sum(w * rv)
    e[n] <- x[n] - y[n]
    w <- w + 2 * mu * e[n] * rv
  }
  list(signal = e, noise_estimate = y, weights = w,
       mu = mu, order = M)
}

#' @rdname rgadp
#' @keywords internal
#' @export
rangayyan_adaptive_filter <- rgadp
