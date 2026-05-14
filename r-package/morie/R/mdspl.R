# SPDX-License-Identifier: MIT OR Apache-2.0

#' Classical MDS for spatial map of legislators (Armstrong Ch 7)
#'
#' Torgerson double-centring followed by top-k eigen-decomposition.
#' Reports Stress-1 = sqrt(sum (d - hat_d)^2 / sum d^2).
#'
#' @param x Either an (n by p) configuration matrix OR an (n by n)
#'   symmetric distance matrix.
#' @param k Number of MDS dimensions (default 2).
#' @return Named list with `coords`, `eigenvalues`, `stress`, `k`,
#'   `n`, `method`.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
mdspl <- function(x, k = 2L) {
  X <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  n <- nrow(X)
  if (n < 2L)
    return(list(coords = matrix(0, n, k), eigenvalues = rep(0, k),
                stress = NA_real_, k = k, n = n,
                method = "mds_classical"))
  is_dist <- (nrow(X) == ncol(X)) &&
             isTRUE(all.equal(X, t(X))) &&
             all(abs(diag(X)) < 1e-9)
  D <- if (is_dist) X else as.matrix(stats::dist(X))
  D2 <- D^2
  J <- diag(n) - matrix(1 / n, n, n)
  B <- -0.5 * J %*% D2 %*% J
  e <- eigen((B + t(B)) / 2, symmetric = TRUE)
  ev <- e$values; vec <- e$vectors
  k_eff <- min(k, n - 1L)
  pos <- pmax(ev[seq_len(k_eff)], 0)
  coords <- vec[, seq_len(k_eff), drop = FALSE] *
            matrix(sqrt(pos), n, k_eff, byrow = TRUE)
  Dh <- as.matrix(stats::dist(coords))
  denom <- sum(D^2)
  stress <- if (denom > 0) sqrt(sum((D - Dh)^2) / denom) else NA_real_
  list(coords = coords, eigenvalues = ev, stress = stress,
       k = k_eff, n = n, method = "mds_classical")
}

#' @keywords internal
#' @rdname mdspl
#' @export
mds_spatial_map <- mdspl
