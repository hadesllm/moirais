#' Higuchi fractal dimension -- Rangayyan Ch 7
#'
#' Higuchi (1988) fractal dimension via curve-length scaling.
#'
#' @param x Numeric vector.
#' @param kmax Maximum lag (default 10).
#' @return Named list `HFD`, `log_L`, `log_inv_k`, `kmax`.
#' @references Higuchi (1988), Physica D 31:277. Rangayyan Ch 7.
#' @export
#' @examples
#' set.seed(0); rghfd(rnorm(500), kmax = 8)$HFD
rghfd <- function(x, kmax = 10L) {
  N <- length(x)
  if (N < 4 || kmax < 2) stop("Need length(x) >= 4 and kmax >= 2.")
  kmax <- as.integer(min(kmax, N %/% 2L))
  L <- numeric(kmax)
  for (k in seq_len(kmax)) {
    lk <- numeric(0)
    for (m in 0:(k - 1)) {
      idx <- seq(m + 1L, N, by = k)
      if (length(idx) < 2) next
      diffs <- sum(abs(diff(x[idx])))
      norm  <- (N - 1) / (k * floor((N - m) / k))
      lk <- c(lk, (diffs / k) * norm)
    }
    L[k] <- if (length(lk)) mean(lk) else NA_real_
  }
  ks <- seq_len(kmax)
  log_L <- log(L); log_inv_k <- log(1 / ks)
  fit <- stats::lm(log_L ~ log_inv_k)
  list(HFD = unname(stats::coef(fit)[2]),
       intercept = unname(stats::coef(fit)[1]),
       log_L = log_L, log_inv_k = log_inv_k, kmax = kmax)
}

#' @rdname rghfd
#' @keywords internal
#' @export
rangayyan_higuchi_fd <- rghfd
