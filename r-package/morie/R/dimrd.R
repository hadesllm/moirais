# SPDX-License-Identifier: MIT OR Apache-2.0

#' Dimensionality test via Kaiser scree (Armstrong Ch 7)
#'
#' Counts eigenvalues exceeding `threshold` on the symmetric matrix
#' formed from x (input matrix is used directly if symmetric; else
#' uses the column correlation matrix).
#'
#' @param x Numeric matrix (n by m).
#' @param threshold Eigenvalue cut-off (default 1, Kaiser 1960).
#' @return Named list with `n_dims`, `eigenvalues`, `threshold`,
#'   `scree_gap`, `method`.
#' @export
dimrd <- function(x, threshold = 1) {
  M <- if (is.matrix(x)) x else matrix(as.numeric(x), ncol = 1L)
  n <- nrow(M); m <- ncol(M)
  is_sym <- (n == m) && isTRUE(all.equal(M, t(M)))
  S <- if (is_sym) (M + t(M)) / 2
       else {
         if (m < 2L)
           return(list(n_dims = 0L, eigenvalues = numeric(0),
                       threshold = threshold, scree_gap = NA_integer_,
                       method = "dimensionality_test"))
         S0 <- suppressWarnings(stats::cor(M))
         S0[is.na(S0)] <- 0; diag(S0) <- 1; S0
       }
  ev <- sort(eigen((S + t(S)) / 2, symmetric = TRUE,
                   only.values = TRUE)$values, decreasing = TRUE)
  n_dims <- sum(ev > threshold)
  gaps <- -diff(ev)
  scree <- if (length(gaps)) which.max(gaps) else 0L
  list(n_dims = as.integer(n_dims), eigenvalues = ev,
       threshold = threshold, scree_gap = as.integer(scree),
       method = "dimensionality_test")
}

#' @keywords internal
#' @rdname dimrd
#' @export
dimensionality_test <- dimrd
