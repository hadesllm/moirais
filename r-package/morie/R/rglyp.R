#' Largest Lyapunov exponent (Rosenstein) — Rangayyan Ch 7
#'
#' Rosenstein et al. (1993) algorithm via delay embedding and
#' nearest-neighbour divergence tracking.
#'
#' @param x Numeric vector.
#' @param m Embedding dimension (default 3).
#' @param tau Embedding lag (default 1).
#' @param max_t Maximum forward step (default `min(M/4, 100)`).
#' @param theiler Theiler-window exclusion (default 10).
#' @return Named list `lyapunov`, `divergence_curve`, `t`.
#' @references Rosenstein et al. (1993), Physica D 65:117.
#' @export
#' @examples
#' set.seed(0); rglyp(rnorm(200), m = 3, tau = 1, max_t = 20)$lyapunov
rglyp <- function(x, m = 3L, tau = 1L, max_t = NULL, theiler = 10L) {
  N <- length(x)
  M <- N - (m - 1L) * tau
  if (M < 10) stop("Series too short for embedding.")
  Y <- matrix(0, nrow = M, ncol = m)
  for (i in seq_len(m)) Y[, i] <- x[((i - 1L) * tau + 1L):((i - 1L) * tau + M)]
  if (is.null(max_t)) max_t <- min(as.integer(M / 4), 100L)
  d <- as.matrix(stats::dist(Y))
  iv <- seq_len(M)
  mask <- abs(outer(iv, iv, "-")) <= theiler
  d[mask] <- Inf
  nn <- max.col(-d, ties.method = "first")
  div <- rep(NA_real_, max_t)
  for (t in seq_len(max_t)) {
    t0 <- t - 1L
    ok <- (iv + t0 <= M) & (nn + t0 <= M)
    if (!any(ok)) next
    diffs <- sqrt(rowSums((Y[iv[ok] + t0, , drop = FALSE] -
                            Y[nn[ok] + t0, , drop = FALSE])^2))
    diffs <- diffs[diffs > 0]
    if (length(diffs)) div[t] <- mean(log(diffs))
  }
  ts <- which(is.finite(div))
  if (length(ts) < 3) {
    lam <- NA_real_
  } else {
    half <- max(3L, length(ts) %/% 2L)
    lam <- stats::coef(stats::lm(div[ts[seq_len(half)]] ~ ts[seq_len(half)]))[2]
  }
  list(lyapunov = unname(lam), divergence_curve = div,
       t = seq_len(max_t))
}

#' @rdname rglyp
#' @keywords internal
#' @export
rangayyan_lyapunov <- rglyp
