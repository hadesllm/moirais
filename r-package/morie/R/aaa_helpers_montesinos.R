# SPDX-License-Identifier: GPL-2.0-only

# Shared Montesinos-suite helpers.
#
# Sourced before gblpf.R / mtgbl.R / mrkvr.R so callers don't depend on R's
# alphabetical load order. Loaded first via the leading underscore in the
# filename and via the explicit Collate: field in DESCRIPTION.

#' VanRaden Genomic Relationship Matrix
#'
#' Computes G = ZZ' / (2 sum p_j(1-p_j)) for method 1 (default), or the
#' per-locus-scaled variant for method 2.
#'
#' @param markers Numeric (n x m) genotype matrix coded (coded 0/1/2).
#' @param method  1 or 2 (VanRaden 2008).
#' @return Named list with estimate (G matrix), diag_mean, off_mean, p, n, m, method.
#' @references VanRaden (2008) J Dairy Sci 91:4414. Montesinos Lopez Ch 3.
#' @export
grm_vanraden <- function(markers, method = 1) {
  M <- as.matrix(markers); storage.mode(M) <- "double"
  n <- nrow(M); m <- ncol(M)
  p <- colMeans(M) / 2
  Z <- sweep(M, 2, 2 * p, "-")
  if (identical(method, 2)) {
    s <- sqrt(2 * p * (1 - p)); s[s <= 0] <- 1
    Z <- sweep(Z, 2, s, "/")
    denom <- m
    method_str <- "VanRaden method 2 (per-locus scaled)"
  } else {
    denom <- 2 * sum(p * (1 - p)); if (denom <= 0) denom <- 1
    method_str <- "VanRaden method 1 (sum-2pq)"
  }
  G <- tcrossprod(Z) / denom
  diag_mean <- mean(diag(G))
  off <- G; diag(off) <- 0
  off_mean <- if (n > 1) sum(off) / (n * (n - 1)) else 0
  list(estimate = G, diag_mean = diag_mean, off_mean = off_mean,
       p = p, n = n, m = m, method = method_str)
}

# CANONICAL TEST
# set.seed(0); M <- matrix(sample(0:2, 20, TRUE), 4, 5)
# grm_vanraden(M)$diag_mean  # ~1 in expectation
