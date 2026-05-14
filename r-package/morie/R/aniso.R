# SPDX-License-Identifier: MIT OR Apache-2.0
#' Anisotropy detection via Levene comparison of directional pair-difference
#' distributions.
#'
#' For each of `n_dirs` directions in [0, pi), select pairs whose connecting
#' vector falls within `tol_deg` of the direction and apply Levene's test
#' across direction groups.
#'
#' @param x Numeric vector.
#' @param coords Coord matrix (n by 2).
#' @param n_dirs Number of directional sectors (default 4).
#' @param tol_deg Half-width tolerance (degrees, default 22.5).
#' @return Named list: statistic, p_value, directional_gamma,
#'   directions_deg, n, method.
#' @references Goovaerts (1997); Schabenberger & Gotway (2005), Ch 3.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
aniso <- function(x, coords, n_dirs = 4, tol_deg = 22.5) {
  x <- as.numeric(x); n <- length(x)
  coords <- if (is.matrix(coords)) coords else
    matrix(as.numeric(unlist(coords)), nrow = n)
  if (nrow(coords) != n) stop("coords rows must match length(x)")
  if (ncol(coords) == 1) {
    return(list(statistic = 0, p_value = 1, n = n,
                method = "Anisotropy test (1D: trivially isotropic)"))
  }
  tol <- tol_deg * pi / 180
  angles <- seq(0, pi, length.out = n_dirs + 1)[-(n_dirs + 1)]
  iu <- which(upper.tri(matrix(0, n, n)), arr.ind = TRUE)
  dv <- coords[iu[, 2], , drop = FALSE] - coords[iu[, 1], , drop = FALSE]
  ang <- atan2(dv[, 2], dv[, 1]) %% pi
  groups <- list(); means <- numeric(); kept_angles <- numeric()
  for (a in angles) {
    diff <- abs(ang - a)
    diff <- pmin(diff, pi - diff)
    mask <- diff <= tol
    if (sum(mask) < 2) next
    d2 <- (x[iu[mask, 1]] - x[iu[mask, 2]]) ^ 2
    groups[[length(groups) + 1]] <- d2
    means <- c(means, 0.5 * mean(d2))
    kept_angles <- c(kept_angles, a * 180 / pi)
  }
  if (length(groups) < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, n = n,
                method = "Anisotropy test (insufficient pairs)"))
  }
  # Levene's test using base R via group-wise abs-from-median
  med_all <- vapply(groups, stats::median, numeric(1))
  abs_dev <- mapply(function(g, m) abs(g - m), groups, med_all,
                    SIMPLIFY = FALSE)
  combined <- unlist(abs_dev)
  grp_idx <- rep(seq_along(abs_dev), vapply(abs_dev, length, integer(1)))
  fit <- stats::aov(combined ~ factor(grp_idx))
  s <- summary(fit)[[1]]
  list(statistic = s$`F value`[1], p_value = s$`Pr(>F)`[1],
       directional_gamma = means, directions_deg = kept_angles,
       n = n,
       method = sprintf("Anisotropy test (Levene across %d directions)", n_dirs))
}

#' @rdname aniso
#' @keywords internal
#' @export
anisotropy_test <- aniso
