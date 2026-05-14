# SPDX-License-Identifier: MIT OR Apache-2.0
#' Moran's I spatial autocorrelation (Schabenberger Ch 1).
#'
#' Computes the global Moran's I and a two-sided p-value under the
#' Cliff-Ord (1981) randomization variance.
#'
#' Formula:
#' \deqn{I = (n / S_0) \sum_{ij} w_{ij}(x_i - \bar x)(x_j - \bar x)
#'           / \sum_i (x_i - \bar x)^2}
#'
#' @param x Numeric vector of length n (observed values).
#' @param w n-by-n numeric matrix of spatial weights.
#' @return Named list: statistic, p_value, expectation, variance,
#'   z_score, n, method.
#' @references Cliff & Ord (1981). Schabenberger & Gotway (2005), Ch 1.
#' @export
sptau <- function(x, w) {
  x <- as.numeric(x); n <- length(x)
  W <- as.matrix(w)
  if (!all(dim(W) == c(n, n)))
    stop("w must be an n-by-n matrix")
  if (n < 3) {
    return(list(statistic = NA_real_, p_value = NA_real_,
                expectation = NA_real_, variance = NA_real_,
                z_score = NA_real_, n = n,
                method = "Moran's I (spatial autocorrelation)"))
  }
  xbar <- mean(x); z <- x - xbar
  S0 <- sum(W)
  num <- as.numeric(t(z) %*% W %*% z)
  den <- sum(z ^ 2)
  if (S0 == 0 || den == 0)
    return(list(statistic = NA_real_, p_value = NA_real_,
                expectation = NA_real_, variance = NA_real_,
                z_score = NA_real_, n = n,
                method = "Moran's I (spatial autocorrelation)"))
  I <- (n / S0) * (num / den)
  EI <- -1 / (n - 1)
  S1 <- 0.5 * sum((W + t(W)) ^ 2)
  S2 <- sum((rowSums(W) + colSums(W)) ^ 2)
  m2 <- mean(z ^ 2); m4 <- mean(z ^ 4)
  b2 <- if (m2 > 0) m4 / (m2 ^ 2) else 3
  A <- n * ((n ^ 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ^ 2)
  B <- b2 * ((n ^ 2 - n) * S1 - 2 * n * S2 + 6 * S0 ^ 2)
  Cc <- (n - 1) * (n - 2) * (n - 3) * S0 ^ 2
  if (Cc <= 0) {
    var_I <- NA_real_; z_sc <- NA_real_; p <- NA_real_
  } else {
    var_I <- (A - B) / Cc - EI ^ 2
    if (is.na(var_I) || var_I <= 0) {
      z_sc <- NA_real_; p <- NA_real_
    } else {
      z_sc <- (I - EI) / sqrt(var_I)
      p <- 2 * (1 - stats::pnorm(abs(z_sc)))
    }
  }
  list(statistic = I, p_value = p, expectation = EI,
       variance = var_I, z_score = z_sc, n = n,
       method = "Moran's I (spatial autocorrelation)")
}

# CANONICAL TEST
# x <- c(1,2,3,4,5); n <- 5; W <- matrix(0, n, n)
# for (i in 1:(n-1)) { W[i,i+1] <- 1; W[i+1,i] <- 1 }
# sptau(x, W)$statistic   # expect 0.5

#' @rdname sptau
#' @keywords internal
#' @export
spatial_autocorrelation <- sptau
