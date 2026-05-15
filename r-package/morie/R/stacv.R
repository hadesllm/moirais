# SPDX-License-Identifier: AGPL-3.0-or-later
#' Empirical spatiotemporal autocovariance.
#'
#' \deqn{\hat C(h, u) = \frac{1}{|N(h, u)|}
#'       \sum_{(i,j) \in N(h,u)} (Z_i - \bar Z)(Z_j - \bar Z)}.
#'
#' @param x Numeric vector.
#' @param coords Coord matrix (n by d_s).
#' @param times Numeric times vector.
#' @param n_spatial_bins,n_temporal_bins Integer bin counts.
#' @param max_spatial,max_temporal Upper cutoffs (default: max/2).
#' @return Named list: estimate (C, spatial_bins, temporal_bins, counts),
#'   n, method.
#' @references Cressie & Huang (1999); Schabenberger & Gotway (2005), Ch 8.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
stacv <- function(x, coords, times,
                  n_spatial_bins = 6, n_temporal_bins = 6,
                  max_spatial = NULL, max_temporal = NULL) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  t <- as.numeric(times)
  if (nrow(coords) != n || length(t) != n)
    stop("shape mismatch among x, coords, times")
  xbar <- mean(x)
  sd_full <- as.matrix(stats::dist(coords))
  td_full <- abs(outer(t, t, "-"))
  iu <- which(upper.tri(sd_full), arr.ind = TRUE)
  sd_f <- sd_full[iu]; td_f <- td_full[iu]
  prods <- (x[iu[, 1]] - xbar) * (x[iu[, 2]] - xbar)
  if (is.null(max_spatial))
    max_spatial <- if (max(sd_f) > 0) max(sd_f) / 2 else 1
  if (is.null(max_temporal))
    max_temporal <- if (max(td_f) > 0) max(td_f) / 2 else 1
  s_edges <- seq(0, max_spatial, length.out = n_spatial_bins + 1)
  t_edges <- seq(0, max_temporal, length.out = n_temporal_bins + 1)
  C <- matrix(NA_real_, n_spatial_bins, n_temporal_bins)
  counts <- matrix(0L, n_spatial_bins, n_temporal_bins)
  for (i in seq_len(n_spatial_bins)) {
    for (j in seq_len(n_temporal_bins)) {
      m <- sd_f > s_edges[i] & sd_f <= s_edges[i + 1] &
           td_f > t_edges[j] & td_f <= t_edges[j + 1]
      k <- sum(m)
      counts[i, j] <- as.integer(k)
      if (k > 0) C[i, j] <- mean(prods[m])
    }
  }
  s_mids <- 0.5 * (s_edges[-1] + s_edges[-(n_spatial_bins + 1)])
  t_mids <- 0.5 * (t_edges[-1] + t_edges[-(n_temporal_bins + 1)])
  list(estimate = list(C = C, spatial_bins = s_mids,
                       temporal_bins = t_mids, counts = counts),
       n = n, method = "Empirical spatiotemporal autocovariance")
}

# CANONICAL TEST  (3x3 spatiotemporal grid of x = i+t)

#' @rdname stacv
#' @keywords internal
#' @export
spatiotemporal_autocovariance <- stacv
