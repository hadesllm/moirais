#' Correlation dimension (Grassberger-Procaccia) — Rangayyan Ch 7
#'
#' Slope of \eqn{\log C(r)} vs \eqn{\log r} in the middle scaling region.
#'
#' @param x Numeric vector.
#' @param m Embedding dimension (default 3).
#' @param tau Embedding lag (default 1).
#' @param n_r Number of radii (default 20).
#' @return Named list `D2`, `log_r`, `log_C`, `m`, `tau`.
#' @references Grassberger & Procaccia (1983), Physica D 9:189.
#' @export
#' @examples
#' set.seed(0); rgcrl(rnorm(200), m = 3, tau = 1, n_r = 15)$D2
rgcrl <- function(x, m = 3L, tau = 1L, n_r = 20L) {
  N <- length(x)
  M <- N - (m - 1L) * tau
  if (M < 10) stop("Series too short for embedding.")
  Y <- matrix(0, nrow = M, ncol = m)
  for (i in seq_len(m)) Y[, i] <- x[((i - 1L) * tau + 1L):((i - 1L) * tau + M)]
  dist <- as.numeric(stats::dist(Y))
  if (length(dist) == 0) stop("No pairwise distances.")
  pos <- dist[dist > 0]
  rmin <- max(if (length(pos)) min(pos) else 1e-12, 1e-12)
  rmax <- max(dist)
  rs <- 10^seq(log10(rmin), log10(rmax), length.out = n_r)
  C <- vapply(rs, function(r) mean(dist <= r), numeric(1))
  mask <- C > 0 & is.finite(C)
  log_r <- log(rs[mask]); log_C <- log(C[mask])
  if (length(log_r) < 3) {
    D2 <- NA_real_
  } else {
    n  <- length(log_r)
    lo <- max(1L, n %/% 5L)
    hi <- max(lo + 2L, n - n %/% 5L)
    D2 <- unname(stats::coef(stats::lm(log_C[lo:hi] ~ log_r[lo:hi]))[2])
  }
  list(D2 = D2, log_r = log_r, log_C = log_C, m = m, tau = tau)
}

#' @rdname rgcrl
#' @keywords internal
#' @export
rangayyan_correlation_dimension <- rgcrl
