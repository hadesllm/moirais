# SPDX-License-Identifier: MIT OR Apache-2.0

#' Fauzi: Hoeffding (H-) decomposition of a degree-2 U-statistic (Ch 5)
#'
#' For a symmetric kernel g(x1,x2),
#' \eqn{U_n = \binom{n}{2}^{-1}\sum_{i<j} g(X_i,X_j)},
#' \eqn{\sigma_1^2 = \mathrm{Var}\, g_1(X)} (Hajek projection),
#' \eqn{\mathrm{Var}(U_n) \approx 4\sigma_1^2/n}.
#'
#' @param x Numeric vector.
#' @param kernel Function(a,b); default 0.5*(a-b)^2 (estimates variance).
#' @param max_pairs Cap on pairs evaluated.
#' @param seed RNG seed for subsampling pairs.
#' @return Named list with estimate, sigma1_sq, sigma2_sq, se, n, n_pairs, method.
#' @importFrom utils combn
#' @importFrom stats var
#' @export
fzhdc <- function(x, kernel = NULL, max_pairs = 2000L, seed = 0L) {
  x <- as.numeric(x); n <- length(x)
  if (n < 4L) return(list(estimate = NA_real_, n = n,
                           method = "fzhdc - too few obs"))
  if (is.null(kernel)) kernel <- function(a, b) 0.5 * (a - b)^2
  total <- n * (n - 1) / 2
  if (total <= max_pairs) {
    pairs <- utils::combn(n, 2)
  } else {
    set.seed(seed)
    seen <- character(); pairs_list <- list()
    while (length(pairs_list) < max_pairs) {
      ij <- sample.int(n, 2)
      i <- min(ij); j <- max(ij)
      k <- paste(i, j, sep = "-")
      if (!(k %in% seen)) {
        seen <- c(seen, k); pairs_list[[length(pairs_list) + 1]] <- c(i, j)
      }
    }
    pairs <- do.call(cbind, pairs_list)
  }
  g_vals <- vapply(seq_len(ncol(pairs)),
                   function(p) kernel(x[pairs[1, p]], x[pairs[2, p]]),
                   numeric(1))
  theta <- mean(g_vals)
  sigma2 <- stats::var(g_vals)
  g1 <- numeric(n); cnt <- numeric(n)
  for (p in seq_len(ncol(pairs))) {
    i <- pairs[1, p]; j <- pairs[2, p]; v <- g_vals[p]
    g1[i] <- g1[i] + v; cnt[i] <- cnt[i] + 1
    g1[j] <- g1[j] + v; cnt[j] <- cnt[j] + 1
  }
  cnt[cnt == 0] <- 1; g1 <- g1 / cnt - theta
  sigma1_sq <- stats::var(g1)
  var_U <- 4 * sigma1_sq / n
  list(estimate = theta, sigma1_sq = sigma1_sq, sigma2_sq = sigma2,
       se = sqrt(max(var_U, 0)), n = n, n_pairs = ncol(pairs),
       method = "Fauzi H-decomposition of degree-2 U-statistic (Ch 5)")
}

# CANONICAL TEST
# set.seed(0); x <- rnorm(200); r <- fzhdc(x)
# stopifnot(r$estimate > 0.7 && r$estimate < 1.3)

#' @rdname fzhdc
#' @keywords internal
#' @export
fauzi_h_decomposition <- fzhdc
