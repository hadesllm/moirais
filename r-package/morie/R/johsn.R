# SPDX-License-Identifier: MIT OR Apache-2.0

#' Johansen trace test for cointegration
#'
#' @param x Numeric matrix (T x k) of I(1) candidate series.
#' @param k_ar_diff Number of lagged differences. Default 1.
#' @return Named list with \code{eigenvalues, trace_stat, crit_values,
#'   rank, n, k, method}.
#' @examples
#' \dontrun{
#'   # See the package vignettes for usage examples:
#'   #   vignette(package = "morie")
#' }
#' @export
johansen_cointegration <- function(x, k_ar_diff = 1) {
  Y <- as.matrix(x); if (nrow(Y) < ncol(Y)) Y <- t(Y)
  Tt <- nrow(Y); k <- ncol(Y)
  if (Tt < 20 || k < 2) stop("Need T>=20, k>=2.")
  if (requireNamespace("urca", quietly = TRUE)) {
    jres <- urca::ca.jo(Y, type = "trace", ecdet = "none",
                        K = max(k_ar_diff + 1, 2))
    return(list(eigenvalues = jres@lambda,
                trace_stat = jres@teststat,
                crit_values = jres@cval,
                rank = sum(jres@teststat > jres@cval[, "5pct"]),
                n = Tt, k = k,
                method = "Johansen trace test via urca::ca.jo"))
  }
  dY <- diff(Y)
  rows <- nrow(dY) - k_ar_diff
  Z0 <- dY[(k_ar_diff + 1):nrow(dY), , drop = FALSE]
  Z1 <- Y[(k_ar_diff + 1):(k_ar_diff + rows), , drop = FALSE]
  Z2 <- matrix(1, rows, 1)
  if (k_ar_diff > 0) {
    for (i in seq_len(k_ar_diff))
      Z2 <- cbind(Z2, dY[(k_ar_diff - i + 1):(k_ar_diff - i + rows), ,
                          drop = FALSE])
  }
  P <- Z2 %*% solve(crossprod(Z2)) %*% t(Z2)
  R0 <- Z0 - P %*% Z0; R1 <- Z1 - P %*% Z1
  S00 <- crossprod(R0) / rows; S01 <- crossprod(R0, R1) / rows
  S11 <- crossprod(R1) / rows
  M <- solve(S11) %*% t(S01) %*% solve(S00) %*% S01
  eig <- sort(Re(eigen(M, only.values = TRUE)$values), decreasing = TRUE)
  eig <- pmax(pmin(eig, 1 - 1e-12), 0)
  trace_stat <- sapply(0:(k - 1), function(r) -rows * sum(log(1 - eig[(r + 1):k])))
  crit_table <- list(
    `1` = c(2.7055, 3.8415, 6.6349), `2` = c(13.4294, 15.4943, 19.9349),
    `3` = c(27.0669, 29.7961, 35.4628), `4` = c(44.4929, 47.8545, 54.6815),
    `5` = c(65.8202, 69.8189, 77.8202)
  )
  crit_values <- do.call(rbind, lapply(seq_len(k),
    function(r) crit_table[[as.character(k - r + 1)]] %||% c(NA, NA, NA)))
  rank <- sum(trace_stat > crit_values[, 2])
  list(eigenvalues = eig, trace_stat = trace_stat,
       crit_values = crit_values, rank = rank,
       n = Tt, k = k,
       method = "Johansen trace test (reduced-rank regression, base R)")
}
