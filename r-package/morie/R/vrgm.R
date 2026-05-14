# SPDX-License-Identifier: MIT OR Apache-2.0
#' Empirical (Matheron) variogram estimation.
#'
#' \deqn{\hat\gamma(h) = \frac{1}{2|N(h)|}
#'        \sum_{(i,j) \in N(h)} (Z(s_i) - Z(s_j))^2}
#'
#' @param x Numeric vector, length n.
#' @param coords Numeric matrix (n by d) of coordinates.
#' @param n_bins Number of distance bins (default 10).
#' @param max_dist Upper distance cutoff. Default max(dist)/2.
#' @return Named list with estimate (sub-list bins, gamma, n_pairs),
#'   n, method.
#' @references Matheron (1962); Schabenberger & Gotway (2005), Ch 3.
#' @export
vrgm <- function(x, coords, n_bins = 10, max_dist = NULL) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  if (nrow(coords) != n) stop("coords rows must match length(x)")
  if (n < 2) stop("need at least 2 points")
  D <- as.matrix(stats::dist(coords))
  iu <- which(upper.tri(D), arr.ind = TRUE)
  dists <- D[iu]
  diffs2 <- (x[iu[, 1]] - x[iu[, 2]]) ^ 2
  if (is.null(max_dist)) max_dist <- max(dists) / 2
  edges <- seq(0, max_dist, length.out = n_bins + 1)
  mids <- 0.5 * (edges[-1] + edges[-(n_bins + 1)])
  gamma <- rep(NA_real_, n_bins); npairs <- integer(n_bins)
  for (k in seq_len(n_bins)) {
    m <- dists > edges[k] & dists <= edges[k + 1]
    npairs[k] <- sum(m)
    if (npairs[k] > 0) gamma[k] <- 0.5 * mean(diffs2[m])
  }
  list(
    estimate = list(bins = mids, gamma = gamma, n_pairs = npairs),
    n = n, method = "Empirical (Matheron) variogram"
  )
}

# CANONICAL TEST
# vrgm(c(1,2,3,4,5), matrix(0:4, ncol=1), n_bins=4, max_dist=4)

#' @rdname vrgm
#' @keywords internal
#' @export
variogram_estimation <- vrgm
