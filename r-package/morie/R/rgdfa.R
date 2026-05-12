#' Detrended fluctuation analysis -- Rangayyan Ch 7
#'
#' DFA-alpha scaling exponent (Peng et al. 1994).
#'
#' @param x Numeric vector.
#' @param scales Optional integer box sizes (default geometric 4..N/4).
#' @param order Detrending polynomial order (default 1 = DFA-1).
#' @return Named list `alpha`, `scales`, `F`, `log_scales`, `log_F`.
#' @references Peng et al. (1994), Phys Rev E 49:1685; Rangayyan Ch 7.
#' @export
#' @examples
#' set.seed(0); rgdfa(rnorm(500))$alpha
rgdfa <- function(x, scales = NULL, order = 1L) {
  N <- length(x)
  if (N < 32) stop("DFA needs at least 32 samples.")
  if (is.null(scales)) {
    log_n <- seq(log(4), log(max(8, N %/% 4)), length.out = 12)
    scales <- unique(round(exp(log_n)))
    scales <- scales[scales >= 4]
  }
  scales <- as.integer(scales)
  y <- cumsum(x - mean(x))
  F <- numeric(length(scales))
  for (j in seq_along(scales)) {
    n <- scales[j]
    nseg <- N %/% n
    if (nseg < 1) { F[j] <- NA_real_; next }
    rms <- numeric(nseg)
    for (k in seq_len(nseg)) {
      seg <- y[((k - 1) * n + 1):(k * n)]
      t_ <- seq_len(n)
      p <- stats::lm(seg ~ stats::poly(t_, order, raw = TRUE))
      rms[k] <- mean(stats::residuals(p)^2)
    }
    F[j] <- sqrt(mean(rms))
  }
  mask <- is.finite(F) & F > 0
  log_n <- log(scales[mask]); log_F <- log(F[mask])
  fit <- stats::lm(log_F ~ log_n)
  list(alpha = unname(stats::coef(fit)[2]),
       scales = scales, F = F,
       log_scales = log_n, log_F = log_F)
}

#' @rdname rgdfa
#' @keywords internal
#' @export
rangayyan_dfa <- rgdfa
