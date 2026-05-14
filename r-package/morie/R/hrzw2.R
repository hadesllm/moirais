# SPDX-License-Identifier: MIT OR Apache-2.0

#' Wild-bootstrap MISE bandwidth selection for NW regression
#'
#' @param x Numeric covariate vector.
#' @param y Numeric response vector.
#' @param B Integer number of bootstrap replications (default 50).
#' @param n_h Integer bandwidth grid size (default 15).
#' @param seed Integer RNG seed (default 0).
#' @return Named list with estimate (h_star), h_silverman, mise_curve, h_grid, n, B, method.
#' @keywords internal
hrzw2 <- function(x, y, B = 50, n_h = 15, seed = 0) {
  x <- as.numeric(x); y <- as.numeric(y); n <- length(x)
  if (n < 30 || length(y) != n)
    return(list(estimate = NA_real_, n = n,
                method = "bw-bootstrap (insufficient data)"))
  nw_fit <- function(x_train, y_train, x_eval, h) {
    u <- outer(x_eval, x_train, `-`) / h
    w <- exp(-0.5 * u^2); s <- rowSums(w); safe <- ifelse(s > 0, s, 1)
    as.numeric((w %*% y_train) / safe)
  }
  h_sil <- .hrz_silverman(x)
  h_grid <- seq(0.5 * h_sil, 2.5 * h_sil, length.out = n_h)
  m_pilot <- nw_fit(x, y, x, h_sil)
  r <- y - m_pilot
  set.seed(seed)
  mise <- numeric(n_h)
  for (j in seq_along(h_grid)) {
    ise <- 0
    for (b in 1:B) {
      v <- sample(c(-1, 1), n, replace = TRUE)
      y_star <- m_pilot + r * v
      m_star <- nw_fit(x, y_star, x, h_grid[j])
      ise <- ise + mean((m_star - m_pilot)^2)
    }
    mise[j] <- ise / B
  }
  j_star <- which.min(mise)
  list(estimate = as.numeric(h_grid[j_star]), h_silverman = as.numeric(h_sil),
       mise_curve = mise, h_grid = h_grid, n = n, B = B,
       method = "Wild-bootstrap MISE bandwidth selection (Faraway-Jhun)")
}

# canonical full-name alias (Py<->R API parity)
#' @rdname hrzw2
#' @keywords internal
#' @export
horowitz_bandwidth_bootstrap <- hrzw2
