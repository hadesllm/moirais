# SPDX-License-Identifier: MIT OR Apache-2.0

#' Kendall's coefficient of concordance W (Gibbons Ch 12.5)
#'
#' Supports incomplete rankings via NA entries.  For complete
#' rankings, W = 12 S / (k^2 (n^3 - n)) where S is the sum of
#' squared deviations of object rank-sums from their mean.
#' Significance via chi-square approximation k(n-1) W ~ chi-square with n-1 df.
#'
#' @param x Matrix (n objects rows x k rankers cols); NA = not ranked.
#' @return Named list: statistic (W), p_value, df, chi2, n, k.
#' @importFrom stats pchisq
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
concordance_incomplete <- function(x) {
  X <- as.matrix(x)
  storage.mode(X) <- "numeric"
  n <- nrow(X); k <- ncol(X)
  if (n < 2 || k < 2) {
    return(list(statistic = NA_real_, p_value = NA_real_, df = n - 1L,
                chi2 = NA_real_, n = n, k = k,
                method = "Kendall's coefficient of concordance W"))
  }
  R <- matrix(NA_real_, n, k)
  for (j in seq_len(k)) {
    col <- X[, j]
    mask <- !is.na(col)
    if (sum(mask) >= 2) R[mask, j] <- rank(col[mask])
  }
  Ri <- rowSums(R, na.rm = TRUE)
  ki <- rowSums(!is.na(R))
  if (all(ki == k)) {
    Rbar <- mean(Ri)
    S <- sum((Ri - Rbar)^2)
    W <- 12 * S / (k^2 * (n^3 - n))
  } else {
    expected <- (n + 1) / 2
    S <- 0; norm <- 0
    for (i in seq_len(n)) {
      ri <- R[i, !is.na(R[i, ])]
      if (length(ri) > 0) {
        S <- S + sum((ri - expected)^2)
        norm <- norm + length(ri)
      }
    }
    max_S <- norm * (n^2 - 1) / 12
    W <- if (max_S > 0) S / max_S else NA_real_
  }
  df <- n - 1
  chi2 <- k * df * W
  p <- 1 - stats::pchisq(chi2, df)
  list(
    statistic = W,
    p_value = p,
    df = df,
    chi2 = chi2,
    n = n,
    k = k,
    method = "Kendall's coefficient of concordance W (incomplete rankings)"
  )
}
